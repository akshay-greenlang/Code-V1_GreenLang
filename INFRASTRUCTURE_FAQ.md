# GreenLang Infrastructure FAQ

**Frequently Asked Questions About GreenLang-First Architecture**

Version: 1.0.0 | Last Updated: November 9, 2025

---

## General Questions

### Q1: What is the GreenLang-First Architecture Policy?

**A:** It's a development principle that requires developers to use existing GreenLang infrastructure components before writing custom code. The policy states: "Always use GreenLang infrastructure. Never build custom when infrastructure exists."

**Why?** To achieve 70-95% code reduction, 60-80% time savings, and eliminate technical debt.

---

### Q2: When can I write custom code?

**A:** Custom code is allowed in these cases:
1. **Business logic** unique to your application (domain-specific calculations)
2. **UI/UX code** (user interfaces, frontend components)
3. **Integration glue** (connecting infrastructure components)
4. **After ADR approval** (Architecture Decision Record required)

**Process:** Search infrastructure → Request enhancement (if close) → Write ADR → Get approval → Implement

---

### Q3: How do I find the right infrastructure component?

**A:** Follow this workflow:
1. **Quick Reference** ([INFRASTRUCTURE_QUICK_REF.md](INFRASTRUCTURE_QUICK_REF.md)) - 1-page cheat sheet
2. **Search Catalog** ([GREENLANG_INFRASTRUCTURE_CATALOG.md](GREENLANG_INFRASTRUCTURE_CATALOG.md)) - Ctrl+F keywords
3. **Check Examples** (GL-CBAM-APP, GL-CSRD-APP, GL-VCCI-APP)
4. **Ask Team** (Discord #infrastructure)

**90% of use cases are in the Quick Reference!**

---

### Q4: What if infrastructure doesn't do exactly what I need?

**A:** You have 3 options:
1. **Request enhancement** - File GitHub issue tagged `infrastructure`, `enhancement`
2. **Adapt your approach** - Can you solve the problem differently using existing infrastructure?
3. **Write ADR** - Justify custom code with Architecture Decision Record

**Infrastructure team prioritizes enhancement requests!**

---

### Q5: What is IUM (Infrastructure Usage Metric)?

**A:** IUM measures what percentage of your application code uses GreenLang infrastructure:

```
IUM = (Infrastructure LOC / Total Application LOC) × 100
```

**Target:** 80%
**Current average:** 85%

**Why it matters:** Higher IUM = less custom code = faster development + less technical debt

---

## LLM Infrastructure

### Q6: When should I use LLM/AI?

**A:** Use LLM for:
- ✅ Narrative generation (reports, explanations)
- ✅ Text classification (categorize spend, waste types)
- ✅ Entity resolution (fuzzy match supplier names)
- ✅ Natural language queries
- ✅ Insights and recommendations

**Never use LLM for:**
- ❌ Numeric calculations (use Python arithmetic)
- ❌ Compliance decisions (use deterministic rules)
- ❌ Data validation (use ValidationFramework)
- ❌ Critical business logic

**Rule:** Zero hallucination for calculations = database + Python only

---

### Q7: Why can't I use LLM for calculations?

**A:** Because LLMs are non-deterministic:
- Same input ≠ same output
- Can hallucinate numbers ("approximately 125.5")
- Cannot be certified for regulatory compliance
- Expensive ($0.01+ per calculation)
- Slow (500-2000ms vs <1μs for Python)

**Example:**
```python
# WRONG
result = llm.complete("What is 100 * 0.18?")  # Might say "18" or "approximately 18"

# CORRECT
result = 100 * 0.18  # Always 18.0
```

---

### Q8: How do I switch between OpenAI and Anthropic?

**A:** ChatSession is provider-agnostic:

```python
from greenlang.intelligence import ChatSession

# OpenAI
session = ChatSession(provider="openai", model="gpt-4")

# Anthropic
session = ChatSession(provider="anthropic", model="claude-3-opus")

# Same API for both!
response = session.complete(prompt="Hello")
```

Configuration in `config/llm_config.yaml`:

```yaml
llm:
  default_provider: openai  # Change to "anthropic"
```

---

## Agent Framework

### Q9: Do all agents need to inherit from Agent base class?

**A:** **Yes, absolutely.** All agents MUST inherit from `greenlang.sdk.base.Agent`.

**Why?**
- Standardized error handling
- Automatic provenance tracking
- Automatic telemetry
- Automatic retry logic
- Input/output validation
- Consistent agent API

**Exception:** None. This is a hard requirement enforced in code review.

---

### Q10: What's the difference between Agent and AsyncAgent?

**A:**
- **Agent** - Synchronous execution (default, simpler)
- **AsyncAgent** - Asynchronous execution (for I/O-bound operations)

**Use Agent when:**
- CPU-bound operations (calculations)
- Simple workflows
- No concurrent I/O

**Use AsyncAgent when:**
- Multiple API calls
- Multiple database queries
- File I/O operations
- Need to run operations concurrently

**Example:**
```python
# Synchronous
class MyAgent(Agent):
    def execute(self, input_data):
        return do_work()

# Asynchronous
class MyAsyncAgent(AsyncAgent):
    async def execute(self, input_data):
        results = await asyncio.gather(
            api_call_1(),
            api_call_2(),
            api_call_3()
        )
        return results
```

---

## Data & Validation

### Q11: Where should I store configuration?

**A:** **Never hardcode configuration.** Use ConfigManager:

```python
from greenlang.config import ConfigManager

config = ConfigManager(
    config_file="config/app_config.yaml",
    environment="production"
)

db_host = config.get("database.host")
api_key = config.get_secret("openai.api_key")  # From environment
```

**config/app_config.yaml:**
```yaml
database:
  host: ${DB_HOST:localhost}
  port: ${DB_PORT:5432}

openai:
  api_key: ${OPENAI_API_KEY}  # From environment
```

**Secrets:** Always from environment variables, never in code or config files.

---

### Q12: How do I validate input data?

**A:** Use ValidationFramework with declarative rules:

```python
from greenlang.validation import ValidationFramework, ValidationRule

rules = [
    ValidationRule(field="value", rule_type="positive"),
    ValidationRule(field="date", rule_type="date_format", parameters={"format": "%Y-%m-%d"}),
    ValidationRule(field="status", rule_type="enum", parameters={"allowed": ["active", "inactive"]})
]

validator = ValidationFramework(rules=rules)
result = validator.validate(data)

if not result.is_valid:
    for error in result.errors:
        print(f"Error: {error.field} - {error.message}")
```

**50+ built-in rules:** positive, enum, range, regex, date_format, email, url, etc.

---

### Q13: Should I use Cache or Database?

**A:** Use both, for different purposes:

**CacheManager (Redis):**
- Temporary data (TTL)
- Frequently accessed data (emission factors)
- Session storage
- Rate limiting counters
- LLM response caching

**DatabaseManager (PostgreSQL):**
- Permanent data
- Transactional data
- Historical records
- Audit trails
- Relational data

**Pattern:** Database as source of truth, cache for performance

```python
# Check cache first
value = cache.get(f"emission_factor:{fuel_type}")
if value is None:
    # Cache miss - query database
    value = db.query("SELECT * FROM emission_factors WHERE fuel_type = %s", [fuel_type])
    # Cache for 1 hour
    cache.set(f"emission_factor:{fuel_type}", value, ttl=3600)
return value
```

---

## Security & Compliance

### Q14: How do I handle secrets?

**A:** **Zero hardcoded secrets. Period.**

**Use environment variables:**
```bash
export OPENAI_API_KEY=sk-...
export DB_PASSWORD=...
```

**Access via ConfigManager:**
```python
config = ConfigManager()
api_key = config.get_secret("openai.api_key")  # Never logged
```

**Never:**
- ❌ Hardcode in code
- ❌ Store in config files (YAML, JSON)
- ❌ Commit to git
- ❌ Log to console/files

**Pre-commit hooks prevent committing secrets!**

---

### Q15: How do I ensure regulatory compliance?

**A:** Follow the Zero Hallucination Architecture:

1. **Never use LLM for calculations** - Database + Python only
2. **Complete provenance tracking** - SHA-256 hash every calculation
3. **Deterministic operations** - Same input = same output
4. **Audit trails** - Log every operation
5. **Validation rules** - 50+ compliance checks

**Example (CBAM/CSRD compliant):**
```python
# Get emission factor from database (deterministic)
factor = db.query("SELECT kgco2_per_kwh FROM factors WHERE fuel='natural_gas'")

# Calculate emissions (deterministic arithmetic)
emissions = consumption * factor

# Track provenance
provenance_hash = hashlib.sha256(f"{consumption}{factor}".encode()).hexdigest()

# Audit log
audit_log(operation="calculate_emissions", input_hash=..., output_hash=..., timestamp=...)
```

---

## Deployment & Operations

### Q16: How do I deploy my application?

**A:** Use the provided Kubernetes manifests:

```bash
# Local development
gl deploy local

# Kubernetes (staging)
gl deploy kubernetes --namespace staging

# Kubernetes (production)
gl deploy kubernetes --namespace production
```

**Infrastructure includes:**
- 77 Kubernetes YAML manifests
- HorizontalPodAutoscaler (autoscaling)
- ConfigMaps and Secrets
- NetworkPolicies
- IngressRoutes

**Location:** `infrastructure/kubernetes/`

---

### Q17: How do I monitor my application?

**A:** TelemetryManager provides automatic monitoring:

**Metrics (Prometheus):**
```python
from greenlang.monitoring import TelemetryManager

telemetry = TelemetryManager(service_name="my-app")

# Counters
telemetry.record_counter("emissions_calculated")

# Histograms (latency)
telemetry.record_histogram("calculation_time_ms", 150)

# Gauges (current value)
telemetry.record_gauge("active_suppliers", 1234)
```

**Traces (OpenTelemetry):**
```python
with telemetry.trace_span("calculate_emissions") as span:
    span.set_attribute("supplier_id", "SUP-001")
    result = calculate()
    span.set_attribute("result", result)
```

**Logs (Structured):**
```python
telemetry.log_info("Calculation complete", extra={
    "supplier_id": "SUP-001",
    "emissions_tco2": 100.5
})
```

**Dashboards:** Grafana dashboards auto-generated from metrics

---

## Performance & Optimization

### Q18: How do I optimize LLM costs?

**A:** Multiple strategies:

1. **Cache responses:**
```python
cache_key = hashlib.sha256(prompt.encode()).hexdigest()
cached = cache.get(f"llm:{cache_key}")
if cached:
    return cached  # Reuse
response = llm.complete(prompt)
cache.set(f"llm:{cache_key}", response, ttl=3600)
```

2. **Use temperature=0:**
```python
session = ChatSession(temperature=0.0)  # Deterministic = cacheable
```

3. **Batch requests:**
```python
# Instead of 100 individual calls
for item in items:
    llm.complete(f"Categorize: {item}")

# Batch into 1 call
llm.complete(f"Categorize these items: {items}")
```

4. **Choose cheaper models:**
- GPT-4: $0.03/1K tokens (expensive, high quality)
- GPT-3.5-Turbo: $0.002/1K tokens (cheap, good quality)
- Use GPT-3.5 when possible, GPT-4 only when needed

---

### Q19: What if infrastructure is too slow?

**A:** First, measure to understand the bottleneck:

```python
with telemetry.trace_span("operation") as span:
    result = operation()
# Check span duration in Jaeger
```

**Common optimizations:**

1. **Enable caching** (CacheManager)
2. **Use batch operations** (bulk inserts, batch LLM calls)
3. **Use async agents** (for I/O-bound operations)
4. **Optimize database queries** (indexes, EXPLAIN ANALYZE)
5. **Request infrastructure enhancement** (if bottleneck is in infrastructure)

**Example:**
```python
# Slow: 100 individual DB queries
for id in ids:
    result = db.query("SELECT * FROM table WHERE id = %s", [id])

# Fast: 1 query with IN clause
results = db.query("SELECT * FROM table WHERE id IN %s", [tuple(ids)])
```

---

## Development Workflow

### Q20: How do I request new infrastructure features?

**A:** File a GitHub issue:

1. **Tag:** `infrastructure`, `enhancement`
2. **Title:** Clear description (e.g., "Add support for LSTM forecasting")
3. **Description:**
   - Use case (what are you trying to do?)
   - Desired API (how would you like to use it?)
   - Example code (show ideal usage)
   - Alternatives considered (what doesn't work?)

**Template:**
```markdown
## Use Case
I need to forecast emissions using LSTM neural networks for complex patterns.

## Desired API
from greenlang.agents import ForecastAgentLSTM
agent = ForecastAgentLSTM(lookback_periods=12)
forecast = agent.run({"historical_data": df, "forecast_periods": 6})

## Alternatives Considered
- SARIMA: Doesn't handle non-linear patterns well
- Custom implementation: 800+ LOC, 2 weeks of work
```

**Infrastructure team will:**
- Review within 1 business day
- Prioritize based on impact
- Implement or suggest alternatives
- Notify you when ready

---

### Q21: How do I contribute to infrastructure?

**A:** We welcome contributions!

**Process:**
1. **Discuss first:** File issue or ask in #infrastructure
2. **Design review:** Propose API design
3. **Implement:** Write code + tests + docs
4. **Submit PR:** Include tests, docs, examples
5. **Code review:** Infrastructure team reviews
6. **Update catalog:** Add to GREENLANG_INFRASTRUCTURE_CATALOG.md

**Quality requirements:**
- 90%+ test coverage
- Complete API documentation
- Code examples
- Performance benchmarks
- Migration guide (if replacing existing pattern)

---

### Q22: What if I disagree with the policy?

**A:** Healthy debate is encouraged!

**Process:**
1. **Understand the rationale:** Read this FAQ, ask in #infrastructure
2. **Discuss your concern:** Team meeting or #infrastructure channel
3. **Propose alternative:** Write RFC (Request for Comments)
4. **Consensus decision:** Team votes

**Recent policy changes:**
- Q3 2025: Increased IUM target from 70% to 80% (based on actual usage)
- Q2 2025: Added AsyncAgent for I/O-bound operations (community request)
- Q1 2025: Added RAG infrastructure (multiple teams needed it)

**Policy evolves based on team needs!**

---

## Troubleshooting

### Q23: My code fails with "AgentExecutionError"

**A:** Agent base class caught an exception. Check the error details:

```python
try:
    result = agent.run(input_data)
except AgentExecutionError as e:
    print(f"Agent: {e.agent_name}")
    print(f"Error: {e.message}")
    print(f"Details: {e.details}")
    print(f"Traceback: {e.traceback}")
```

**Common causes:**
1. **Invalid input:** Check Pydantic validation errors
2. **Database connection:** Check database is running
3. **API timeout:** Increase timeout in config
4. **Rate limit:** LLM provider rate limit exceeded

**Solution:** Check `e.details` for specific cause

---

### Q24: How do I debug LLM prompts?

**A:** Enable prompt logging (development only!):

```yaml
# config/llm_config.yaml
llm:
  provenance:
    log_prompts: true  # WARNING: Logs prompts to console
    log_responses: true
```

**In code:**
```python
session = ChatSession(provider="openai", temperature=0.0)
response = session.complete(prompt="...")

# Check logged prompt/response in console
print(f"Prompt: {session.last_prompt}")
print(f"Response: {session.last_response}")
print(f"Tokens: {session.total_tokens}")
print(f"Cost: ${session.total_cost:.4f}")
```

**Production:** NEVER log prompts (may contain PII, secrets)

---

### Q25: How do I handle migration from custom code?

**A:** Follow the migration pattern in the infrastructure catalog:

**Step 1:** Identify infrastructure component
**Step 2:** Read migration guide for that component
**Step 3:** Implement side-by-side (old + new)
**Step 4:** Test thoroughly (unit + integration)
**Step 5:** Switch traffic (gradual rollout)
**Step 6:** Remove old code
**Step 7:** Measure improvement (LOC, performance, cost)

**Example:**
```python
# Phase 1: Side-by-side
from my_custom_agent import OldAgent
from greenlang.sdk.base import Agent

class NewAgent(Agent):
    def execute(self, input_data):
        # New implementation
        pass

# Route some traffic to new agent
if use_new_agent:
    result = NewAgent().run(input_data)
else:
    result = OldAgent().run(input_data)

# Phase 2: All traffic to new agent
result = NewAgent().run(input_data)

# Phase 3: Remove old agent
# Delete my_custom_agent.py
```

**See:** [MIGRATION_TO_INFRASTRUCTURE.md](GL-CBAM-APP/MIGRATION_TO_INFRASTRUCTURE.md) for detailed guides

---

## Resources

- **Quick Reference:** [INFRASTRUCTURE_QUICK_REF.md](INFRASTRUCTURE_QUICK_REF.md)
- **Full Catalog:** [GREENLANG_INFRASTRUCTURE_CATALOG.md](GREENLANG_INFRASTRUCTURE_CATALOG.md)
- **Onboarding:** [DEVELOPER_ONBOARDING.md](DEVELOPER_ONBOARDING.md)
- **Tutorial:** [INFRASTRUCTURE_TUTORIAL.md](INFRASTRUCTURE_TUTORIAL.md)
- **Examples:** `examples/` directory

**Questions not answered here?**
- Discord: #infrastructure
- GitHub: Issues tagged `infrastructure`, `question`
- Email: infrastructure@greenlang.io
- Office Hours: Tuesdays 2-3pm PT

---

**Last Updated:** November 9, 2025
**Version:** 1.0.0
**Maintainer:** GreenLang Infrastructure Team
