# Developer Onboarding Guide

**Welcome to GreenLang! Your Guide to the GreenLang-First Architecture**

Version: 1.0.0 | Last Updated: November 9, 2025

---

## Welcome!

You're joining a team that's building the Climate Operating System. This guide will teach you the **GreenLang-First Architecture** principle that powers our 8-10x faster development velocity.

**Goal:** By the end of this guide, you'll understand how to build climate-aware applications using GreenLang infrastructure instead of writing custom code.

---

## Table of Contents

1. [The GreenLang-First Principle](#the-greenlang-first-principle)
2. [How to Search for Infrastructure](#how-to-search-for-infrastructure)
3. [How to Use the Infrastructure Catalog](#how-to-use-the-infrastructure-catalog)
4. [Common Patterns DO vs DON'T](#common-patterns-do-vs-dont)
5. [When Custom Code is Allowed](#when-custom-code-is-allowed)
6. [How to Create an ADR](#how-to-create-an-adr)
7. [Code Examples](#code-examples)
8. [Your First Week Checklist](#your-first-week-checklist)
9. [Getting Help](#getting-help)

---

## The GreenLang-First Principle

### Core Philosophy

**"Always use GreenLang infrastructure. Never build custom when infrastructure exists."**

This isn't just a guideline — it's a **requirement** enforced through:
- Pre-commit hooks
- Code review checklists
- Quarterly audits
- Architecture Decision Records (ADRs)

### Why This Matters

**Development Velocity:**
- **70-95% code reduction** (boilerplate eliminated)
- **60-80% time savings** (don't reinvent the wheel)
- **Consistent quality** (battle-tested infrastructure)
- **Zero technical debt** (maintained by infrastructure team)

**Example:**
- Custom agent implementation: 400 LOC, 3 days
- Using `greenlang.sdk.base.Agent`: 50 LOC, 4 hours
- **Savings:** 350 LOC (87%), 2.5 days (83%)

### The Numbers

- **Infrastructure Available:** 100+ components, 50,000+ LOC
- **Target IUM (Infrastructure Usage Metric):** 80%
- **Current Average IUM:** 85% (we're crushing it!)

---

## How to Search for Infrastructure

### Step-by-Step Process

**Before writing ANY code, follow this workflow:**

1. **Check the Quick Reference** ([INFRASTRUCTURE_QUICK_REF.md](INFRASTRUCTURE_QUICK_REF.md))
   - 1-page cheat sheet of common tasks
   - "Need X? → Use Y" format
   - Covers 90% of use cases

2. **Search the Infrastructure Catalog** ([GREENLANG_INFRASTRUCTURE_CATALOG.md](GREENLANG_INFRASTRUCTURE_CATALOG.md))
   - Ctrl+F for keywords
   - 5000+ lines of documentation
   - Complete API docs, examples, migration guides

3. **Check Existing Apps**
   - GL-CBAM-APP
   - GL-CSRD-APP
   - GL-VCCI-Carbon-APP
   - See how they use infrastructure

4. **Ask the Team**
   - Discord: #infrastructure channel
   - GitHub: Search issues tagged `infrastructure`
   - Team member: Ping infrastructure team

### Search Keywords

**If you need to...**

- **Call an LLM:** Search "LLM", "ChatSession", "GPT", "Claude"
- **Build an agent:** Search "Agent", "base class", "execute"
- **Cache data:** Search "cache", "Redis", "CacheManager"
- **Validate data:** Search "validation", "rules", "ValidationFramework"
- **Authenticate users:** Search "auth", "JWT", "AuthManager"
- **Access database:** Search "database", "PostgreSQL", "DatabaseManager"
- **Generate reports:** Search "report", "PDF", "Excel", "ReportGenerator"
- **Forecast emissions:** Search "forecast", "SARIMA", "time series"
- **Detect anomalies:** Search "anomaly", "outlier", "IForest"

---

## How to Use the Infrastructure Catalog

### Structure

The catalog is organized into 20 sections:

1. LLM Infrastructure (ChatSession, RAG, Embeddings)
2. Agent Framework (Agent base, AsyncAgent, AgentSpec)
3. Data Storage & Caching (CacheManager, DatabaseManager)
4. Authentication & Authorization (AuthManager)
5. API Frameworks (FastAPI integration)
6. Validation & Security (ValidationFramework)
7. Monitoring & Telemetry (TelemetryManager)
8. Configuration Management (ConfigManager)
9. Pipeline & Orchestration (PipelineOrchestrator)
10. Data Processing (DataTransformer, ETL)
11. Reporting & Output (ReportGenerator)
12. ERP Connectors (SAP, Oracle, Workday)
13. Emissions & Climate Data (EmissionFactorDatabase)
14. Machine Learning (Forecasting, Anomaly Detection)
15. Testing Infrastructure (Fixtures, Mocks)
16. Deployment & Infrastructure (Kubernetes, Docker)
17. CLI Framework (Typer commands)
18. Pack System (Modular components)
19. Migration Patterns (Before/After examples)
20. Support & Resources

### Each Component Includes

- **Purpose:** What it does, why it exists
- **Use Cases:** When to use it
- **When to Use vs. Build Custom:** Decision criteria
- **API Documentation:** Complete API reference
- **Code Examples:** Copy-paste examples
- **Configuration:** How to configure
- **Migration Guide:** How to migrate from custom code
- **Performance:** Latency, throughput, cost
- **Best Practices:** DOs and DON'Ts
- **Related Components:** What works well together

### Reading Tips

1. **Use Ctrl+F:** Don't read sequentially, search for your need
2. **Read Purpose First:** Understand what the component does
3. **Check Examples:** Copy and adapt to your use case
4. **Review Best Practices:** Avoid common pitfalls
5. **Check Related Components:** You might need multiple pieces

---

## Common Patterns: DO vs DON'T

### Pattern 1: LLM Integration

#### ❌ DON'T: Custom OpenAI Code

```python
import openai

openai.api_key = "sk-..."  # Hardcoded secret!

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.0
)

content = response.choices[0].message.content
cost = estimate_cost(response.usage)  # Custom function

# No provenance tracking
# No error handling
# No retry logic
# Hardcoded secrets
```

#### ✅ DO: Use ChatSession

```python
from greenlang.intelligence import ChatSession

session = ChatSession(
    provider="openai",
    model="gpt-4",
    temperature=0.0,
    track_provenance=True  # Automatic
)

response = session.complete(
    prompt=prompt,
    system="You are a helper."
)

content = response.content
cost = response.cost  # Tracked automatically

# ✅ Automatic provenance tracking
# ✅ Automatic error handling
# ✅ Automatic retry on transient failures
# ✅ Secrets from environment (ConfigManager)
# ✅ Cost and token tracking
# ✅ Latency monitoring
```

**Why Better:**
- 200+ LOC saved (error handling, retry, tracking)
- Secrets never hardcoded
- Consistent across all agents
- Centralized monitoring

---

### Pattern 2: Agent Development

#### ❌ DON'T: Custom Agent Class

```python
class MyAgent:
    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(name)

    def run(self, input_data):
        try:
            self.logger.info(f"Starting {self.name}")
            start_time = time.time()

            # Validate input
            if not self._validate_input(input_data):
                raise ValueError("Invalid input")

            # Execute logic
            result = self._execute(input_data)

            # Validate output
            if not self._validate_output(result):
                raise ValueError("Invalid output")

            # Log metrics
            duration = time.time() - start_time
            self.logger.info(f"Completed in {duration}s")

            # Track provenance
            self._save_provenance(input_data, result, duration)

            return result

        except Exception as e:
            self.logger.error(f"Failed: {e}")
            raise

    def _validate_input(self, data):
        # Custom validation
        pass

    def _execute(self, data):
        # Business logic
        pass

    def _validate_output(self, result):
        # Custom validation
        pass

    def _save_provenance(self, input_data, result, duration):
        # Custom provenance tracking
        pass

# 100+ LOC of boilerplate for EVERY agent!
```

#### ✅ DO: Inherit from Agent Base

```python
from greenlang.sdk.base import Agent
from pydantic import BaseModel

class MyAgentInput(BaseModel):
    value: float
    threshold: float = 0.5

class MyAgentOutput(BaseModel):
    result: float

class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            name="my-agent",
            version="1.0.0",
            description="Does amazing things"
        )

    def execute(self, input_data: MyAgentInput) -> MyAgentOutput:
        # Only business logic here!
        result = input_data.value * input_data.threshold
        return MyAgentOutput(result=result)

# ✅ Input validation automatic (Pydantic)
# ✅ Output validation automatic (Pydantic)
# ✅ Error handling automatic
# ✅ Logging automatic
# ✅ Provenance automatic
# ✅ Retry automatic
# ✅ Metrics automatic

# 20 LOC vs 100+ LOC custom!
```

**Why Better:**
- 80+ LOC saved per agent
- Consistent error messages
- Automatic provenance tracking
- Automatic telemetry
- Input/output validation with Pydantic

---

### Pattern 3: Data Validation

#### ❌ DON'T: Custom Validation Logic

```python
def validate_emissions_data(data):
    errors = []

    # Check required fields
    if "emissions_tco2" not in data:
        errors.append("Missing emissions_tco2")

    # Check positive values
    if data.get("emissions_tco2", 0) < 0:
        errors.append("emissions_tco2 must be positive")

    # Check date format
    try:
        datetime.strptime(data["date"], "%Y-%m-%d")
    except (ValueError, KeyError):
        errors.append("Invalid date format")

    # Check enum values
    if data.get("fuel_type") not in ["natural_gas", "electricity", "diesel"]:
        errors.append("Invalid fuel_type")

    # ... 50 more checks ...

    return {
        "is_valid": len(errors) == 0,
        "errors": errors
    }

# 200+ LOC for validation!
# Duplicate logic across apps!
```

#### ✅ DO: Use ValidationFramework

```python
from greenlang.validation import ValidationFramework, ValidationRule

# Define rules declaratively (or load from YAML)
rules = [
    ValidationRule(
        field="emissions_tco2",
        rule_type="positive",
        message="emissions_tco2 must be positive"
    ),
    ValidationRule(
        field="date",
        rule_type="date_format",
        parameters={"format": "%Y-%m-%d"},
        message="Invalid date format"
    ),
    ValidationRule(
        field="fuel_type",
        rule_type="enum",
        parameters={"allowed": ["natural_gas", "electricity", "diesel"]},
        message="Invalid fuel_type"
    )
]

validator = ValidationFramework(rules=rules)

result = validator.validate(data)

# ✅ 20 LOC vs 200+ LOC
# ✅ 50+ built-in validation rules
# ✅ Reusable across apps
# ✅ Can load from YAML
# ✅ Detailed error messages
```

**Why Better:**
- 180+ LOC saved
- Declarative (rules in YAML, not code)
- Reusable validation rules
- Consistent error messages

---

### Pattern 4: LLM for Calculations (NEVER!)

#### ❌ DON'T: LLM for Math

```python
from greenlang.intelligence import ChatSession

session = ChatSession(provider="openai")

# WRONG WRONG WRONG!
response = session.complete(
    prompt=f"Calculate: {consumption} kg × {factor} kgCO2/kg"
)

# Try to parse result
result = float(response.content.split()[0])  # Brittle!

# Problems:
# - LLM might hallucinate: "approximately 125.5"
# - Non-deterministic: same input ≠ same output
# - Slow: 500-2000ms
# - Expensive: $0.01 per calculation
# - No audit trail
# - Cannot certify for regulatory compliance
```

#### ✅ DO: Python Arithmetic

```python
# CORRECT
result = consumption * factor

# ✅ Deterministic: same input = same output
# ✅ Fast: <1μs
# ✅ Free: $0
# ✅ Complete audit trail
# ✅ Regulatory compliant
```

**Rule:** **NEVER use LLM for numeric calculations or compliance decisions.**

---

### Pattern 5: Configuration Management

#### ❌ DON'T: Hardcoded Configuration

```python
# WRONG
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "greenlang_prod"
DB_PASSWORD = "password123"  # SECRET IN CODE!

OPENAI_API_KEY = "sk-abc123..."  # SECRET IN CODE!

MAX_RETRIES = 3
TIMEOUT = 30

# Problems:
# - Secrets in code (security risk!)
# - No environment-specific configs
# - Hard to change without redeploying
# - No centralized management
```

#### ✅ DO: Use ConfigManager

```python
from greenlang.config import ConfigManager

config = ConfigManager(
    config_file="config/app_config.yaml",
    environment="production"  # or "development", "staging"
)

db_host = config.get("database.host")
db_port = config.get("database.port")
db_name = config.get("database.name")
db_password = config.get_secret("database.password")  # From environment

api_key = config.get_secret("openai.api_key")  # Never logged

max_retries = config.get("retry.max_attempts", default=3)

# ✅ Secrets from environment (never in code)
# ✅ Environment-specific configs
# ✅ Change config without redeploying
# ✅ Centralized management
# ✅ Validation on load
```

**config/app_config.yaml:**
```yaml
database:
  host: ${DB_HOST:localhost}
  port: ${DB_PORT:5432}
  name: greenlang_${ENV}
  password: ${DB_PASSWORD}  # From environment

openai:
  api_key: ${OPENAI_API_KEY}
  model: gpt-4

retry:
  max_attempts: 3
  timeout: 30
```

**Why Better:**
- Zero hardcoded secrets
- Environment-specific configs
- Centralized management
- Validation on load

---

## When Custom Code is Allowed

### Approved Use Cases

1. **Business Logic**
   - Domain-specific calculations unique to your app
   - Industry-specific algorithms
   - Example: CBAM carbon intensity calculations

2. **UI/UX Code**
   - User interface components
   - Frontend logic
   - Example: React components, dashboards

3. **Integration Glue**
   - Connecting infrastructure components
   - App-specific orchestration
   - Example: Pipeline that uses multiple agents

4. **After ADR Approval**
   - Architecture Decision Record required
   - Justification for why infrastructure can't be used
   - Example: Specialized XBRL processing for CSRD

### Process for Custom Code

**Step 1: Search Infrastructure**
- Check Quick Reference
- Check Infrastructure Catalog
- Ask team in #infrastructure

**Step 2: Request Enhancement** (if infrastructure is close)
- File GitHub issue tagged `infrastructure`, `enhancement`
- Describe your use case
- Suggest API design
- Infrastructure team will prioritize

**Step 3: Write ADR** (if truly custom is needed)
- Use template: `.greenlang/adr/TEMPLATE.md`
- Justify why infrastructure can't be used
- Evaluate alternatives
- Document trade-offs

**Step 4: Get Approval**
- Submit ADR for review
- Architecture team approves/rejects
- If approved, proceed with custom code

**Step 5: Document**
- Add to app README under "Custom Components"
- Explain why custom was necessary
- Document maintenance plan

---

## How to Create an ADR

### ADR (Architecture Decision Record)

An ADR documents a significant architectural decision, including context, alternatives, and consequences.

### When to Write an ADR

- You want to write custom code instead of using infrastructure
- You want to use a new external library/service
- You want to change how infrastructure is used
- You want to deviate from GreenLang-First principle

### ADR Template

See: `.greenlang/adr/TEMPLATE.md`

**Sections:**
1. **Status:** Proposed | Accepted | Rejected
2. **Context:** Why is this decision needed?
3. **Infrastructure Evaluation:** What infrastructure exists? Why can't it be used?
4. **Decision:** What did we decide to do?
5. **Consequences:** What are the trade-offs?
6. **Alternatives Considered:** What else did we evaluate?

### Example ADR

See: `.greenlang/adr/EXAMPLE-001-zero-hallucination-exemption.md`

**Summary:**
- **Context:** CBAM calculations require 100% deterministic results
- **Infrastructure:** ChatSession exists but uses LLM (non-deterministic)
- **Decision:** Use database lookups + Python arithmetic (zero hallucination)
- **Consequence:** Cannot use LLM for numeric calculations
- **Approval:** ✅ Accepted (regulatory requirement)

---

## Code Examples

### Example 1: Build Your First Agent

**Goal:** Create an agent that calculates carbon emissions from fuel consumption.

```python
from greenlang.sdk.base import Agent
from pydantic import BaseModel
from typing import Literal

# Define input/output schemas
class FuelInput(BaseModel):
    fuel_type: Literal["natural_gas", "electricity", "diesel"]
    consumption: float  # Must be positive
    unit: Literal["kwh", "therms", "gallons"]

    # Pydantic validators (automatic)
    class Config:
        validate_assignment = True

class EmissionOutput(BaseModel):
    emissions_tco2: float
    emission_factor: float
    source: str
    provenance_hash: str

# Create agent
class FuelEmissionsAgent(Agent):
    def __init__(self):
        super().__init__(
            name="fuel-emissions-agent",
            version="1.0.0",
            description="Calculate emissions from fuel consumption"
        )

        # Load emission factors (cached)
        from greenlang.data import EmissionFactorDatabase
        self.efdb = EmissionFactorDatabase()

    def execute(self, input_data: FuelInput) -> EmissionOutput:
        # Step 1: Get emission factor (deterministic database lookup)
        factor = self.efdb.get_factor(
            activity=f"{input_data.fuel_type}_combustion",
            region="US",
            year=2024
        )

        # Step 2: Convert units
        consumption_kwh = self._convert_to_kwh(
            input_data.consumption,
            input_data.unit
        )

        # Step 3: Calculate emissions (deterministic arithmetic)
        emissions_kg = consumption_kwh * factor.kgco2_per_kwh
        emissions_tco2 = emissions_kg / 1000

        # Step 4: Generate provenance hash
        import hashlib
        provenance = hashlib.sha256(
            f"{input_data}{factor}".encode()
        ).hexdigest()

        return EmissionOutput(
            emissions_tco2=emissions_tco2,
            emission_factor=factor.kgco2_per_kwh,
            source=factor.source,
            provenance_hash=provenance
        )

    def _convert_to_kwh(self, value: float, unit: str) -> float:
        """Convert various units to kWh"""
        conversions = {
            "kwh": 1.0,
            "therms": 29.3,  # 1 therm = 29.3 kWh
            "gallons": 33.7  # 1 gallon diesel = 33.7 kWh
        }
        return value * conversions[unit]

# Usage
agent = FuelEmissionsAgent()

result = agent.run(FuelInput(
    fuel_type="natural_gas",
    consumption=1000,
    unit="therms"
))

print(f"Emissions: {result.emissions_tco2:.2f} tCO2")
print(f"Factor: {result.emission_factor} kgCO2/kWh")
print(f"Source: {result.source}")
print(f"Provenance: {result.provenance_hash[:16]}...")

# Automatic features:
# - Input validation (Pydantic)
# - Output validation (Pydantic)
# - Error handling (try/catch in base class)
# - Provenance tracking (execution history)
# - Telemetry (metrics sent to Prometheus)
# - Retry logic (on transient failures)
```

**What You Get for Free:**
- Input validation: Pydantic checks types, enums automatically
- Output validation: Ensures return value matches schema
- Error handling: Base class catches exceptions, logs details
- Provenance: Execution history tracked automatically
- Telemetry: Metrics sent to Prometheus
- Retry: Automatic retry on network failures

**Lines of Code:**
- Custom approach: ~400 LOC (with all boilerplate)
- Infrastructure approach: ~60 LOC (business logic only)
- **Savings: 340 LOC (85%)**

---

### Example 2: Build a Multi-Agent Pipeline

**Goal:** Create a pipeline that ingests data, validates it, calculates emissions, and generates a report.

```python
from greenlang.pipeline import PipelineOrchestrator, PipelineStep
from greenlang.sdk.base import Agent

# Assume you have these agents (inherit from Agent base)
from my_agents import DataIntakeAgent, ValidationAgent, EmissionsCalculatorAgent, ReportGeneratorAgent

# Create pipeline
pipeline = PipelineOrchestrator(
    name="emissions-reporting-pipeline",
    version="1.0.0",
    description="End-to-end emissions reporting"
)

# Step 1: Intake (no dependencies)
pipeline.add_step(
    PipelineStep(
        name="intake",
        agent=DataIntakeAgent(),
        dependencies=[],  # Runs first
        retry={"max_attempts": 3, "backoff_factor": 2}
    )
)

# Step 2: Validate (depends on intake)
pipeline.add_step(
    PipelineStep(
        name="validate",
        agent=ValidationAgent(),
        dependencies=["intake"],  # Runs after intake
        retry={"max_attempts": 1}  # Don't retry validation
    )
)

# Step 3: Calculate (depends on validate)
pipeline.add_step(
    PipelineStep(
        name="calculate",
        agent=EmissionsCalculatorAgent(),
        dependencies=["validate"],
        retry={"max_attempts": 3}
    )
)

# Step 4: Report (depends on calculate)
pipeline.add_step(
    PipelineStep(
        name="report",
        agent=ReportGeneratorAgent(),
        dependencies=["calculate"],
        retry={"max_attempts": 2}
    )
)

# Execute pipeline
result = pipeline.run(input_data={
    "data_file": "emissions_data.csv",
    "company_profile": "company.json",
    "output_dir": "output/"
})

# Access results by step
intake_result = result["intake"]
validation_result = result["validate"]
emissions_result = result["calculate"]
report_result = result["report"]

print(f"Pipeline completed in {result.total_duration_ms}ms")
print(f"Steps executed: {len(result.steps)}")
print(f"Final report: {report_result.report_path}")

# Automatic features:
# - Dependency management (DAG execution)
# - Parallel execution (independent steps run concurrently)
# - Error handling (step failures, rollback)
# - Checkpointing (resume from failure)
# - Provenance (complete lineage tracking)
```

**What You Get for Free:**
- Dependency management: DAG execution
- Parallel execution: Independent steps run concurrently
- Error handling: Step failures, rollback
- Checkpointing: Resume from failure
- Provenance: Complete lineage

---

## Your First Week Checklist

### Day 1: Setup & Orientation

- [ ] Clone GreenLang repo
- [ ] Set up development environment
- [ ] Read this onboarding guide
- [ ] Read Quick Reference ([INFRASTRUCTURE_QUICK_REF.md](INFRASTRUCTURE_QUICK_REF.md))
- [ ] Skim Infrastructure Catalog ([GREENLANG_INFRASTRUCTURE_CATALOG.md](GREENLANG_INFRASTRUCTURE_CATALOG.md))
- [ ] Join Discord #infrastructure channel
- [ ] Attend team standup

### Day 2: Learn by Example

- [ ] Read GL-CBAM-APP README (see infrastructure usage)
- [ ] Read GL-CSRD-APP README (see infrastructure usage)
- [ ] Explore `examples/` directory
- [ ] Run example: `python examples/quickstart/basic_agent.py`
- [ ] Review ADR examples (`.greenlang/adr/EXAMPLE-*.md`)

### Day 3: Hands-On Practice

- [ ] Build your first agent (Example 1 above)
- [ ] Add validation to your agent
- [ ] Add caching to your agent
- [ ] Run tests: `pytest tests/`
- [ ] Pair with senior developer

### Day 4: Understand the Stack

- [ ] Read about LLM infrastructure (ChatSession, RAG)
- [ ] Read about Agent framework (Agent base class)
- [ ] Read about Data infrastructure (Cache, Database)
- [ ] Read about Monitoring (Telemetry)
- [ ] Explore `greenlang/` directory structure

### Day 5: Contribute

- [ ] Pick a starter task from backlog
- [ ] Search infrastructure before coding
- [ ] Submit your first PR
- [ ] Participate in code review
- [ ] Ask questions in #infrastructure

---

## Getting Help

### Resources

- **Quick Reference:** [INFRASTRUCTURE_QUICK_REF.md](INFRASTRUCTURE_QUICK_REF.md) (1 page)
- **Full Catalog:** [GREENLANG_INFRASTRUCTURE_CATALOG.md](GREENLANG_INFRASTRUCTURE_CATALOG.md) (5000+ lines)
- **Tutorial:** [INFRASTRUCTURE_TUTORIAL.md](INFRASTRUCTURE_TUTORIAL.md) (step-by-step)
- **FAQ:** [INFRASTRUCTURE_FAQ.md](INFRASTRUCTURE_FAQ.md) (20+ common questions)
- **Examples:** `examples/` directory (30+ examples)

### Channels

- **Discord:** #infrastructure (real-time help)
- **GitHub:** Issues tagged `infrastructure`
- **Email:** infrastructure@greenlang.io
- **Office Hours:** Tuesdays 2-3pm PT

### Who to Ask

- **Infrastructure questions:** Infrastructure team (@infra-team)
- **Agent design:** Architecture team (@arch-team)
- **LLM integration:** AI team (@ai-team)
- **Deployment:** DevOps team (@devops-team)

---

## Final Tips

1. **Search First, Build Last** - 90% of what you need exists
2. **When in Doubt, Ask** - #infrastructure channel is your friend
3. **Read Examples** - Learn from existing apps
4. **Use ADRs** - Document significant decisions
5. **Contribute Back** - Enhance infrastructure for everyone

---

**Welcome to GreenLang! Let's build the Climate Operating System together.**

Questions? Ping us in #infrastructure or email infrastructure@greenlang.io

Happy coding!
