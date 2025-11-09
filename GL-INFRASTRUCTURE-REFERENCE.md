# GreenLang Infrastructure Reference Guide
## The Complete Guide to Building Climate Applications

**Version:** 2.0.0
**Last Updated:** November 9, 2025
**Status:** Production Ready
**Target Audience:** Developers building climate/sustainability applications

---

## 📋 TABLE OF CONTENTS

### PART I: GETTING STARTED
1. [Introduction](#1-introduction)
2. [30-Minute Quick Start](#2-30-minute-quick-start)
3. [Application Architecture Patterns](#3-application-architecture-patterns)

### PART II: INFRASTRUCTURE CATALOG
4. [AI & Intelligence Layer](#4-ai--intelligence-layer)
5. [Agent Framework](#5-agent-framework)
6. [Data Infrastructure](#6-data-infrastructure)
7. [Security & Authentication](#7-security--authentication)
8. [Observability & Monitoring](#8-observability--monitoring)
9. [Shared Services](#9-shared-services)

### PART III: BUILD YOUR APPLICATION
10. [Step-by-Step Application Guide](#10-step-by-step-application-guide)
11. [Testing & Quality Assurance](#11-testing--quality-assurance)
12. [Production Deployment](#12-production-deployment)

### PART IV: ADVANCED TOPICS
13. [Performance Optimization](#13-performance-optimization)
14. [Troubleshooting Guide](#14-troubleshooting-guide)
15. [Best Practices & Anti-Patterns](#15-best-practices--anti-patterns)

### PART V: REFERENCE
16. [Quick Reference Tables](#16-quick-reference-tables)
17. [Import Cheat Sheet](#17-import-cheat-sheet)
18. [Decision Trees](#18-decision-trees)

---

# PART I: GETTING STARTED

## 1. Introduction

### What is GreenLang Infrastructure?

GreenLang Infrastructure is a **complete, production-ready platform** for building climate and sustainability applications. Think of it as:

- **AWS** for climate intelligence
- **Django/Rails** but for sustainability apps
- **Kubernetes** for climate workflows

### Infrastructure Stats

| Metric | Value |
|--------|-------|
| **Total Infrastructure Code** | 172,338 LOC |
| **Core Infrastructure** | 81,370 LOC |
| **Reusable Components** | 200+ |
| **Production Modules** | 11 |
| **Applications Built** | 3 (CSRD, CBAM, VCCI) |
| **Development Speed** | 8-10x faster |
| **Cost Reduction** | 77% |
| **Infrastructure Usage** | 82% average |

### Who Should Use This Guide?

- ✅ Developers building new climate applications
- ✅ Teams migrating from custom implementations
- ✅ Architects designing sustainability platforms
- ✅ Product managers scoping climate tech features

### The Golden Rule

**"Always use GreenLang infrastructure. Never build custom when infrastructure exists."**

---

## 2. 30-Minute Quick Start

### Build Your First Climate Application

Let's build a simple **Carbon Footprint Calculator** application in 30 minutes.

#### Step 1: Setup (5 minutes)

```bash
# Create project directory
mkdir my-climate-app
cd my-climate-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install GreenLang
pip install greenlang

# Verify installation
python -c "from greenlang.sdk.base import Agent; print('✓ GreenLang Ready')"
```

#### Step 2: Create Data Intake Agent (5 minutes)

```python
# agents/intake_agent.py
from greenlang.sdk.base import Agent, Result, Metadata
from greenlang.validation import ValidationFramework
import pandas as pd

class CarbonDataIntakeAgent(Agent):
    """Ingest carbon activity data with validation."""

    def __init__(self):
        super().__init__(
            name="carbon-data-intake",
            version="1.0.0",
            description="Ingest and validate carbon activity data"
        )

        # Setup validation
        self.validator = ValidationFramework()
        self.validator.add_validator(
            name="positive_values",
            func=lambda data: all(data['consumption'] > 0),
            config={"severity": "ERROR"}
        )

    def execute(self, input_data: dict) -> Result:
        # Read input file
        df = pd.read_csv(input_data['file_path'])

        # Validate data
        validation_result = self.validator.validate(df.to_dict('records'))

        if not validation_result.is_valid:
            return Result(
                success=False,
                data=None,
                metadata=Metadata(errors=validation_result.errors)
            )

        return Result(
            success=True,
            data=df.to_dict('records'),
            metadata=Metadata(
                rows_processed=len(df),
                validation_status="passed"
            )
        )
```

#### Step 3: Create Calculator Agent (10 minutes)

```python
# agents/calculator_agent.py
from greenlang.sdk.base import Agent, Result
from greenlang.services import FactorBroker, FactorRequest
from greenlang.provenance import ProvenanceTracker

class CarbonCalculatorAgent(Agent):
    """Calculate carbon emissions using emission factors."""

    def __init__(self):
        super().__init__(
            name="carbon-calculator",
            version="1.0.0"
        )
        self.factor_broker = FactorBroker()
        self.tracker = ProvenanceTracker("carbon-calculation")

    async def execute(self, input_data: dict) -> Result:
        results = []

        with self.tracker.track_operation("batch-calculation"):
            for activity in input_data['data']:
                # Get emission factor
                factor_request = FactorRequest(
                    fuel_type=activity['fuel_type'],
                    region=activity.get('region', 'US'),
                    year=activity.get('year', 2024)
                )

                factor = await self.factor_broker.resolve(factor_request)

                # Calculate emissions
                emissions_kg = activity['consumption'] * factor.kgco2e_per_unit
                emissions_tco2 = emissions_kg / 1000

                # Track provenance
                self.tracker.track_data_transformation(
                    operation="emission_calculation",
                    inputs={
                        "consumption": activity['consumption'],
                        "factor": factor.kgco2e_per_unit
                    },
                    outputs={"emissions_tco2": emissions_tco2}
                )

                results.append({
                    **activity,
                    'emissions_tco2': emissions_tco2,
                    'emission_factor': factor.kgco2e_per_unit,
                    'factor_source': factor.source
                })

        # Save provenance
        self.tracker.save_record("provenance.json")

        return Result(
            success=True,
            data=results,
            metadata={'total_emissions': sum(r['emissions_tco2'] for r in results)}
        )
```

#### Step 4: Create Reporting Agent (5 minutes)

```python
# agents/reporting_agent.py
from greenlang.sdk.base import Agent, Result
import pandas as pd

class CarbonReportingAgent(Agent):
    """Generate carbon footprint reports."""

    def __init__(self):
        super().__init__(
            name="carbon-reporting",
            version="1.0.0"
        )

    def execute(self, input_data: dict) -> Result:
        df = pd.DataFrame(input_data['data'])

        # Generate summary
        summary = {
            'total_emissions_tco2': df['emissions_tco2'].sum(),
            'by_fuel_type': df.groupby('fuel_type')['emissions_tco2'].sum().to_dict(),
            'by_region': df.groupby('region')['emissions_tco2'].sum().to_dict(),
            'activity_count': len(df)
        }

        # Export to Excel
        output_path = input_data.get('output_path', 'carbon_report.xlsx')
        df.to_excel(output_path, index=False)

        return Result(
            success=True,
            data=summary,
            metadata={'report_path': output_path}
        )
```

#### Step 5: Create Pipeline (5 minutes)

```python
# pipeline.py
from greenlang.sdk.base import Pipeline
from agents.intake_agent import CarbonDataIntakeAgent
from agents.calculator_agent import CarbonCalculatorAgent
from agents.reporting_agent import CarbonReportingAgent
import asyncio

async def main():
    # Create pipeline
    pipeline = Pipeline(
        name="carbon-footprint-calculator",
        version="1.0.0"
    )

    # Add agents
    pipeline.add_agent(CarbonDataIntakeAgent(), name="intake")
    pipeline.add_agent(CarbonCalculatorAgent(), name="calculator")
    pipeline.add_agent(CarbonReportingAgent(), name="reporting")

    # Run pipeline
    result = await pipeline.run({
        'file_path': 'data/activities.csv',
        'output_path': 'reports/carbon_footprint.xlsx'
    })

    print(f"✓ Pipeline completed!")
    print(f"Total emissions: {result.metadata['total_emissions']:.2f} tCO2")
    print(f"Report saved: {result.metadata['report_path']}")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Step 6: Run Your Application

```bash
# Create sample data
cat > data/activities.csv << EOF
fuel_type,consumption,region,year
natural_gas,1000,US,2024
electricity,5000,US,2024
diesel,500,US,2024
EOF

# Run pipeline
python pipeline.py

# Output:
# ✓ Pipeline completed!
# Total emissions: 3.45 tCO2
# Report saved: reports/carbon_footprint.xlsx
```

**🎉 Congratulations!** You just built a complete carbon footprint calculator using GreenLang infrastructure in 30 minutes!

### What You Got for Free

- ✅ **Validation Framework** - Automatic data quality checks
- ✅ **Emission Factors** - 100K+ factors from authoritative sources
- ✅ **Provenance Tracking** - Complete audit trail
- ✅ **Error Handling** - Automatic retry, circuit breakers
- ✅ **Telemetry** - Metrics, logging, tracing (when configured)
- ✅ **Type Safety** - Pydantic models throughout
- ✅ **Async Support** - High-performance I/O

---

## 3. Application Architecture Patterns

### Pattern 1: Data Intake → Calculate → Report

**Use Case:** CBAM Importer, Scope 3 Calculator, Carbon Accounting

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│ IntakeAgent │ ───> │ CalcAgent    │ ───> │ ReportAgent  │
└─────────────┘      └──────────────┘      └──────────────┘
```

**Components:**
- `greenlang.agents.templates.IntakeAgent` - Multi-format data ingestion
- `greenlang.agents.templates.CalculatorAgent` - Zero-hallucination calculations
- `greenlang.agents.templates.ReportingAgent` - Multi-format export

**Example Apps:** GL-CBAM-APP, GL-VCCI-APP (Scope 3 categories)

### Pattern 2: AI-Powered Analysis Pipeline

**Use Case:** CSRD Materiality Assessment, ESG Reporting, Supplier Risk

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│ IntakeAgent │ ───> │ AI Agent     │ ───> │ ReportAgent  │
│             │      │ (LLM + RAG)  │      │              │
└─────────────┘      └──────────────┘      └──────────────┘
```

**Components:**
- `greenlang.intelligence.ChatSession` - LLM orchestration
- `greenlang.intelligence.rag.RAGEngine` - Knowledge retrieval
- `greenlang.services.entity_mdm.EntityResolver` - Entity matching

**Example Apps:** GL-CSRD-APP (materiality assessment)

### Pattern 3: Multi-Agent Parallel Processing

**Use Case:** Value Chain Carbon Intelligence, Multi-Scope Calculations

```
                    ┌─> Category1Agent ─┐
┌─────────────┐     │                   │     ┌──────────────┐
│ IntakeAgent │ ──┬─┼─> Category2Agent ─┼──> │ AggregatorAgent
│             │   │ │                   │     │              │
└─────────────┘   │ └─> Category3Agent ─┘     └──────────────┘
                  │
                  └─> CategoryNAgent
```

**Components:**
- `greenlang.sdk.base.Pipeline` - Orchestration with parallelization
- `greenlang.cache.CacheManager` - Shared caching across agents
- `greenlang.telemetry` - Distributed tracing

**Example Apps:** GL-VCCI-APP (15 Scope 3 categories)

### Pattern 4: Real-Time Monitoring Dashboard

**Use Case:** Live emissions tracking, Alert systems

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│ Connectors  │ ───> │ StreamAgent  │ ───> │ Dashboard    │
│ (SAP/IoT)   │      │ (WebSocket)  │      │ (GraphQL)    │
└─────────────┘      └──────────────┘      └──────────────┘
```

**Components:**
- `greenlang.connectors` - ERP/IoT integration
- `greenlang.api.websocket` - Real-time streaming
- `greenlang.monitoring` - Dashboard orchestrator

**Example Apps:** Real-time CBAM monitoring (future)

---

# PART II: INFRASTRUCTURE CATALOG

## 4. AI & Intelligence Layer

### 4.1 ChatSession - LLM Orchestration

**Location:** `greenlang.intelligence.ChatSession`
**Lines of Code:** 28,056 (entire intelligence module)
**Status:** ✅ Production Ready

#### When to Use
- AI-powered text generation (narratives, recommendations)
- Natural language understanding (spend categorization)
- Entity resolution (fuzzy matching)
- Q&A over documents (RAG)

#### Quick Start

```python
from greenlang.intelligence import ChatSession, ChatMessage, Role

# Initialize session
session = ChatSession(
    provider="openai",  # or "anthropic"
    model="gpt-4o",
    temperature=0.0,  # Reproducibility
    max_tokens=2000
)

# Simple completion
response = await session.complete(
    prompt="Explain GHG Protocol Scope 3 in 2 sentences",
    system="You are a climate expert"
)

print(response.content)
# Automatic cost tracking
print(f"Cost: ${response.cost:.4f}")
```

#### Advanced: Tool Calling (Zero Hallucination)

```python
# Define extraction tool
extraction_tool = {
    "type": "function",
    "function": {
        "name": "extract_emissions",
        "parameters": {
            "type": "object",
            "properties": {
                "emissions_tco2": {"type": "number"},
                "scope": {"type": "string", "enum": ["Scope 1", "Scope 2", "Scope 3"]}
            },
            "required": ["emissions_tco2", "scope"]
        }
    }
}

# Extract structured data (no hallucination)
response = await session.complete_with_tools(
    prompt=f"Extract emissions from: {invoice_text}",
    tools=[extraction_tool]
)

# Get structured result
emissions = response.tool_calls[0].arguments["emissions_tco2"]  # Guaranteed number!
```

#### RAG (Retrieval-Augmented Generation)

```python
from greenlang.intelligence.rag import RAGEngine

# Initialize RAG
rag = RAGEngine(
    vector_store="weaviate",
    embedding_model="text-embedding-3-large"
)

# Index knowledge base
await rag.ingest_documents([
    {"text": "GHG Protocol scope 3 covers...", "source": "GHG Protocol"},
    {"text": "ESRS E1 requires...", "source": "EFRAG"}
])

# Query with citations
result = await rag.query(
    question="What are the 15 Scope 3 categories?",
    top_k=3
)

print(result.answer)  # Answer with citations
print(result.sources)  # Source documents
```

#### Best Practices

✅ **DO:**
- Use `temperature=0.0` for reproducibility
- Use tool calling for structured data extraction
- Track costs with `track_provenance=True`
- Set `max_tokens` to prevent runaway costs

❌ **DON'T:**
- Never use LLM for numeric calculations (use Python!)
- Never parse unstructured text (use tools!)
- Never skip error handling
- Never use high temperature for compliance work

#### Performance
- **Latency:** 500-5000ms depending on model
- **Cost:** $0.01-0.20 per request
- **Throughput:** 100+ req/min with rate limiting

---

### 4.2 Semantic Cache

**Location:** `greenlang.intelligence.SemanticCache`
**Status:** ✅ Production Ready

#### Use Case
Cache LLM responses for similar (not identical) prompts to reduce costs by 80-90%.

```python
from greenlang.intelligence import SemanticCache, ChatSession

cache = SemanticCache(
    similarity_threshold=0.95,
    embedding_model="text-embedding-3-small"
)

session = ChatSession(
    provider="openai",
    semantic_cache=cache
)

# First call: Full LLM cost
r1 = await session.complete("Explain ESRS E1 Climate Change")  # $0.10, 2s

# Second call: Similar prompt, cache hit!
r2 = await session.complete("Describe ESRS E1 on climate change")  # $0.00, 10ms
# Semantic similarity: 0.96 > 0.95 = cache hit!
```

**Savings:** 80-90% cost reduction for similar queries

---

## 5. Agent Framework

### 5.1 Agent Base Class

**Location:** `greenlang.sdk.base.Agent`
**Status:** ✅ Production Ready

#### The Foundation

**Every agent in your application should inherit from `Agent`**. You get:

- ✅ Input/output validation (Pydantic)
- ✅ Error handling with retry logic
- ✅ Provenance tracking (automatic)
- ✅ Telemetry (metrics, logs, traces)
- ✅ Lifecycle hooks
- ✅ Status management

#### Basic Agent

```python
from greenlang.sdk.base import Agent, Result, Metadata
from pydantic import BaseModel

class MyInput(BaseModel):
    value: float
    unit: str

class MyOutput(BaseModel):
    result: float
    confidence: float

class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            name="my-agent",
            version="1.0.0",
            description="Does something useful"
        )

    def execute(self, input_data: MyInput) -> Result:
        # Your business logic here
        result = input_data.value * 2

        return Result(
            success=True,
            data=MyOutput(result=result, confidence=0.95),
            metadata=Metadata(execution_time_ms=150)
        )

# Usage
agent = MyAgent()
result = agent.run(MyInput(value=100, unit="kwh"))
print(result.data.result)  # 200
```

#### Agent with Dependencies

```python
from greenlang.cache import get_cache_manager
from greenlang.db import get_session
from greenlang.telemetry import get_logger

class AdvancedAgent(Agent):
    def __init__(self):
        super().__init__(name="advanced-agent", version="1.0.0")

        # Infrastructure dependencies
        self.cache = get_cache_manager()
        self.db = get_session()
        self.logger = get_logger(__name__)

    async def execute(self, input_data: dict) -> Result:
        # Use cache
        cached = await self.cache.get(f"key:{input_data['id']}")
        if cached:
            self.logger.info("Cache hit", id=input_data['id'])
            return Result(success=True, data=cached)

        # Query database
        data = await self.db.query("SELECT * FROM table WHERE id = :id",
                                    {"id": input_data['id']})

        # Cache result
        await self.cache.set(f"key:{input_data['id']}", data, ttl=3600)

        return Result(success=True, data=data)
```

---

### 5.2 Pre-Built Agent Templates

#### IntakeAgent - Data Ingestion

```python
from greenlang.agents.templates import IntakeAgent, DataFormat

agent = IntakeAgent()

# Supports: CSV, JSON, Excel, XML, Parquet, Avro, ORC, Feather
result = await agent.ingest(
    file_path="data/emissions.csv",
    format=DataFormat.CSV,
    validate=True,  # Automatic validation
    resolve_entities=True  # Entity resolution
)

print(f"Loaded {result.rows_read} rows")
print(f"Quality score: {result.quality_score}")
```

**Streaming for large files:**
```python
async for chunk in agent.ingest_streaming(
    file_path="large_file.csv",
    chunk_size=10000
):
    process_chunk(chunk.data)
```

#### CalculatorAgent - Zero-Hallucination Calculations

```python
from greenlang.agents.templates import CalculatorAgent

agent = CalculatorAgent()

# Register formulas
agent.register_formula(
    name="scope1_emissions",
    formula=lambda inputs: inputs['fuel_consumption'] * inputs['emission_factor'],
    required_inputs=['fuel_consumption', 'emission_factor']
)

# Calculate
result = await agent.calculate(
    formula_name="scope1_emissions",
    inputs={'fuel_consumption': 1000, 'emission_factor': 2.5}
)

print(result.value)  # 2500 (deterministic!)
print(result.provenance)  # Complete calculation trail
```

**Parallel batch processing:**
```python
results = await agent.batch_calculate_parallel(
    formula_name="scope1_emissions",
    inputs_list=[
        {'fuel_consumption': 1000, 'emission_factor': 2.5},
        {'fuel_consumption': 2000, 'emission_factor': 3.0},
        # ... thousands more
    ],
    use_processes=True  # Process pool for CPU-bound
)
```

#### ReportingAgent - Multi-Format Export

```python
from greenlang.agents.templates import ReportingAgent, ReportFormat

agent = ReportingAgent()

# Generate report
result = await agent.generate_report(
    data=emissions_data,
    format=ReportFormat.EXCEL,
    output_path="reports/emissions_2024.xlsx",
    check_compliance=[ComplianceFramework.GHG_PROTOCOL]
)

# With charts
result = await agent.generate_with_charts(
    data=emissions_data,
    chart_configs=[
        {"type": "bar", "x": "category", "y": "emissions"},
        {"type": "pie", "values": "emissions", "labels": "category"}
    ],
    format=ReportFormat.HTML,
    output_path="reports/dashboard.html"
)
```

---

### 5.3 Pipeline - Orchestration

```python
from greenlang.sdk.base import Pipeline

pipeline = Pipeline(
    name="emissions-calculation",
    version="1.0.0"
)

# Add agents (sequential by default)
pipeline.add_agent(IntakeAgent(), name="intake")
pipeline.add_agent(CalculatorAgent(), name="calculator")
pipeline.add_agent(ReportingAgent(), name="reporting")

# Run pipeline
result = await pipeline.run({
    'file_path': 'data.csv',
    'output_path': 'report.xlsx'
})

# Access individual step results
intake_result = result.steps['intake']
calc_result = result.steps['calculator']
```

**Parallel execution:**
```python
# Add agents in parallel
pipeline.add_agents_parallel([
    (CategoryAgent1(), "cat1"),
    (CategoryAgent2(), "cat2"),
    (CategoryAgent3(), "cat3")
])

# Then aggregate
pipeline.add_agent(AggregatorAgent(), name="aggregate")
```

---

## 6. Data Infrastructure

### 6.1 Validation Framework

**Location:** `greenlang.validation.ValidationFramework`
**Status:** ✅ Production Ready

```python
from greenlang.validation import ValidationFramework, SchemaValidator, RulesEngine

# Create framework
validator = ValidationFramework()

# Add schema validation (JSON Schema)
validator.add_validator(
    name="schema",
    func=SchemaValidator(schema={
        "type": "object",
        "properties": {
            "emissions": {"type": "number", "minimum": 0},
            "fuel_type": {"type": "string", "enum": ["gas", "electricity", "diesel"]}
        },
        "required": ["emissions", "fuel_type"]
    }),
    config={"severity": "ERROR"}
)

# Add business rules
rules_engine = RulesEngine()
rules_engine.add_rule({
    "name": "reasonable_emissions",
    "condition": "emissions < 1000000",
    "message": "Emissions exceed reasonable threshold",
    "severity": "WARNING"
})

validator.add_validator(
    name="business_rules",
    func=rules_engine,
    config={"severity": "WARNING"}
)

# Validate data
result = validator.validate(data)

if not result.is_valid:
    for error in result.errors:
        print(f"{error.severity}: {error.message}")
```

---

### 6.2 Cache Manager

**Location:** `greenlang.cache.CacheManager`
**Status:** ✅ Production Ready

**3-Tier Caching Architecture:**

```python
from greenlang.cache import CacheManager, initialize_cache_manager

# Initialize (once at app startup)
initialize_cache_manager(
    enable_l1=True,  # In-memory LRU
    enable_l2=True,  # Redis distributed
    enable_l3=True,  # Disk persistence
    redis_url="redis://localhost:6379"
)

# Use cache
cache = get_cache_manager()

# Simple get/set
await cache.set("key", "value", ttl=3600)
value = await cache.get("key")

# Get or compute pattern
emission_factor = await cache.get_or_compute(
    key=f"factor:{fuel_type}:{region}",
    compute_fn=lambda: fetch_emission_factor(fuel_type, region),
    ttl=86400  # 24 hours
)
```

**Performance:**
- L1 (memory): <1ms latency, 99%+ hit rate for hot data
- L2 (Redis): 1-5ms latency, 85%+ hit rate for warm data
- L3 (disk): 5-10ms latency, 100% persistence

---

### 6.3 Database Manager

**Location:** `greenlang.db`
**Status:** ✅ Production Ready

```python
from greenlang.db import get_engine, get_session, Base
from sqlalchemy.orm import Session

# Get session (connection pooled)
async with get_session() as session:
    # Query
    results = await session.execute(
        "SELECT * FROM emission_factors WHERE fuel_type = :type",
        {"type": "natural_gas"}
    )

    # Insert
    session.add(EmissionFactor(
        fuel_type="electricity",
        region="US",
        kgco2e_per_kwh=0.385
    ))
    await session.commit()

# Connection pooling is automatic
# - Pool size: 20
# - Max overflow: 10
# - Automatic retry on transient errors
```

**Query Optimization:**
```python
from greenlang.db import QueryOptimizer

optimizer = QueryOptimizer()

# Automatic query caching
result = await optimizer.execute_cached(
    query="SELECT * FROM large_table WHERE category = :cat",
    params={"cat": "scope3"},
    ttl=3600
)
```

---

## 7. Security & Authentication

### 7.1 Authentication Manager

**Location:** `greenlang.auth.AuthManager`
**Status:** ✅ Production Ready

```python
from greenlang.auth import AuthManager

auth = AuthManager()

# Create user
user = await auth.create_user(
    username="analyst@company.com",
    password="secure_password",
    roles=["analyst", "viewer"]
)

# Authenticate
token = await auth.authenticate(
    username="analyst@company.com",
    password="secure_password"
)

# Verify token in API endpoint
payload = await auth.verify_token(token.access_token)

# Check permission
has_access = await auth.check_permission(
    user_id=payload['user_id'],
    resource="emissions_data",
    action="read"
)
```

### 7.2 Enterprise SSO

**SAML 2.0:**
```python
from greenlang.auth.sso import SAMLProvider

saml = SAMLProvider(
    entity_id="https://myapp.com",
    idp_metadata_url="https://idp.com/metadata"
)

# Handle SAML response
user = await saml.process_response(saml_response)
```

**OAuth 2.0/OIDC:**
```python
from greenlang.auth.sso import OAuthProvider

oauth = OAuthProvider(
    client_id="your-client-id",
    client_secret="your-client-secret",
    authorization_url="https://idp.com/oauth/authorize"
)

# Get authorization URL
auth_url = oauth.get_authorization_url(redirect_uri="https://myapp.com/callback")
```

---

## 8. Observability & Monitoring

### 8.1 Structured Logging

**Location:** `greenlang.telemetry.StructuredLogger`
**Status:** ✅ Production Ready

```python
from greenlang.telemetry import get_logger

logger = get_logger(__name__)

# Structured logging (JSON output)
logger.info(
    "Emissions calculated",
    emissions_tco2=123.45,
    supplier_id="SUP-001",
    category="scope3_cat1"
)

# Automatic correlation IDs
# Automatic timestamp, level, module
# Output: {"timestamp": "2024-11-09T10:30:00Z", "level": "INFO", ...}

# Context manager for request context
with logger.context(request_id="req-123", user_id="user-456"):
    logger.info("Processing request")
    # All logs in this block get request_id and user_id
```

### 8.2 Metrics Collection

```python
from greenlang.telemetry import get_metrics_collector

metrics = get_metrics_collector()

# Counter
metrics.increment("emissions_calculated", labels={"category": "scope1"})

# Gauge
metrics.gauge("active_suppliers", 1234)

# Histogram (for latencies)
metrics.record("calculation_duration_ms", 150, labels={"agent": "calculator"})

# Metrics auto-exported to Prometheus
# Access at http://localhost:9090/metrics
```

### 8.3 Distributed Tracing

```python
from greenlang.telemetry import TracingManager

tracer = TracingManager()

# Trace operation
with tracer.create_span("calculate_emissions") as span:
    span.set_attribute("supplier_id", "SUP-001")
    span.set_attribute("category", "scope3_cat1")

    result = calculate_emissions()

    span.set_attribute("emissions_tco2", result)

# Traces sent to Jaeger/Zipkin for visualization
```

---

## 9. Shared Services

### 9.1 Factor Broker Service

**Location:** `greenlang.services.FactorBroker`
**Lines of Code:** 5,530
**Status:** ✅ Production Ready

**The authoritative source for emission factors.**

```python
from greenlang.services import FactorBroker, FactorRequest

broker = FactorBroker()

# Request emission factor
request = FactorRequest(
    fuel_type="natural_gas",
    region="US",
    year=2024,
    gwp_standard="AR6"  # IPCC AR6 100-year GWP
)

factor = await broker.resolve(request)

print(f"Factor: {factor.kgco2e_per_unit}")
print(f"Source: {factor.source}")  # e.g., "EPA 2024"
print(f"Uncertainty: ±{factor.uncertainty_pct}%")
print(f"Data quality: {factor.dqi_score}/5")
```

**Multi-source cascade:**
1. ecoinvent (highest quality)
2. DESNZ UK
3. EPA US
4. IPCC defaults

**Features:**
- 100K+ emission factors
- Automatic source selection
- License compliance (24h cache for ecoinvent)
- Complete provenance
- P95 latency <50ms
- 85%+ cache hit rate

---

### 9.2 Entity MDM (Master Data Management)

**Location:** `greenlang.services.EntityResolver`
**Status:** ✅ Production Ready

**ML-powered entity resolution for supplier deduplication.**

```python
from greenlang.services import EntityResolver

resolver = EntityResolver()

# Match entities (fuzzy matching)
matches = await resolver.find_matches(
    entity_name="Microsoft Corp",
    entity_type="supplier",
    threshold=0.85
)

for match in matches:
    print(f"{match.name}: {match.confidence:.0%} match")
    # Microsoft Corporation: 98% match
    # MSFT: 95% match

# Resolve to canonical entity
canonical = await resolver.resolve(
    entity_name="Microsoft Corp"
)

print(canonical.canonical_name)  # "Microsoft Corporation"
print(canonical.lei_code)  # Legal Entity Identifier
print(canonical.duns_number)  # D-U-N-S Number
```

**Two-stage matching:**
1. **Vector similarity** (fast, 1000 entities/sec)
2. **BERT cross-encoder** (precise, 98%+ accuracy)

**Performance:**
- 85%+ auto-match rate (no human review)
- 98%+ accuracy
- <100ms P95 latency

---

### 9.3 Methodologies Service

**Location:** `greenlang.services.methodologies`
**Lines of Code:** 7,007
**Status:** ✅ Production Ready

**Uncertainty quantification and data quality assessment.**

#### Pedigree Matrix (ILCD)

```python
from greenlang.services.methodologies import PedigreeMatrixEvaluator, PedigreeScore

evaluator = PedigreeMatrixEvaluator()

# Assess data quality
score = PedigreeScore(
    reliability=5,  # Verified data
    completeness=4,  # Representative data
    temporal_correlation=5,  # Less than 3 years
    geographical_correlation=4,  # Area under study
    technological_correlation=5  # Identical technology
)

dqi = evaluator.calculate_dqi(score)
uncertainty = evaluator.calculate_uncertainty(score)

print(f"DQI: {dqi:.2f}/5")  # Data Quality Indicator
print(f"Uncertainty: ±{uncertainty:.1%}")
```

#### Monte Carlo Simulation

```python
from greenlang.services.methodologies import MonteCarloSimulator

simulator = MonteCarloSimulator()

# Simulate uncertainty propagation
result = await simulator.simulate(
    value=1000,  # Baseline value
    uncertainty=0.15,  # ±15% uncertainty
    distribution="lognormal",
    iterations=10000
)

print(f"Mean: {result.mean:.2f}")
print(f"P50 (median): {result.p50:.2f}")
print(f"P95: {result.p95:.2f}")
print(f"P99: {result.p99:.2f}")
```

**Supported distributions:**
- Normal, Lognormal, Triangular, Uniform, Beta, Gamma

**Performance:**
- 10,000 iterations in <1 second
- Parallel processing support

---

### 9.4 PCF Exchange Service

**Location:** `greenlang.services.PCFExchangeService`
**Status:** ✅ Production Ready

**Product Carbon Footprint exchange via PACT Pathfinder protocol.**

```python
from greenlang.services import PCFExchangeService

pcf_service = PCFExchangeService()

# Request PCF from supplier
pcf = await pcf_service.request_pcf(
    product_id="PROD-12345",
    supplier_id="SUP-001",
    protocol="pact_pathfinder_v2"
)

print(f"PCF: {pcf.carbon_footprint} kg CO2e")
print(f"Scope 1: {pcf.scope1_emissions}")
print(f"Scope 2: {pcf.scope2_emissions}")
print(f"Scope 3: {pcf.scope3_emissions}")
print(f"Data quality: {pcf.dqi_score}/5")

# Validate PCF
validation = pcf_service.validate_pcf(pcf)
if validation.is_valid:
    print("✓ PCF validated")
```

**Supported protocols:**
- PACT Pathfinder v2.0
- Catena-X (automotive)
- SAP SDX (planned)

---

# PART III: BUILD YOUR APPLICATION

## 10. Step-by-Step Application Guide

### Building a Complete CBAM Application

Let's build a **Carbon Border Adjustment Mechanism (CBAM)** declaration application step-by-step.

#### Step 1: Project Structure

```
cbam-app/
├── agents/
│   ├── intake_agent.py
│   ├── emissions_calculator_agent.py
│   └── reporting_agent.py
├── config/
│   └── app_config.yaml
├── data/
│   └── cbam_declarations.csv
├── tests/
│   ├── test_intake.py
│   ├── test_calculator.py
│   └── test_integration.py
├── pipeline.py
├── requirements.txt
└── README.md
```

#### Step 2: Configuration (config/app_config.yaml)

```yaml
application:
  name: cbam-importer
  version: 1.0.0
  environment: production

database:
  provider: postgresql
  host: ${DB_HOST:localhost}
  port: ${DB_PORT:5432}
  database: cbam
  pool_size: 20

cache:
  enable_l1: true
  enable_l2: true
  redis_url: ${REDIS_URL:redis://localhost:6379}
  ttl: 86400  # 24 hours

llm:
  provider: openai
  model: gpt-4o
  temperature: 0.0
  max_tokens: 2000

telemetry:
  enable_metrics: true
  enable_tracing: true
  prometheus_port: 9090
```

#### Step 3: Intake Agent (agents/intake_agent.py)

```python
from greenlang.sdk.base import Agent, Result, Metadata
from greenlang.validation import ValidationFramework, SchemaValidator
from greenlang.provenance import ProvenanceTracker
from greenlang.telemetry import get_logger
from pydantic import BaseModel
import pandas as pd

class CBAMDeclarationInput(BaseModel):
    file_path: str
    declaration_period: str

class CBAMDeclarationOutput(BaseModel):
    declarations: list
    rows_processed: int
    validation_errors: list

class CBAMIntakeAgent(Agent):
    def __init__(self):
        super().__init__(
            name="cbam-intake",
            version="1.0.0",
            description="Ingest CBAM shipment declarations"
        )

        self.logger = get_logger(__name__)
        self.tracker = ProvenanceTracker("cbam-intake")

        # Setup validation
        self.validator = ValidationFramework()
        self.validator.add_validator(
            name="cbam_schema",
            func=SchemaValidator(schema={
                "type": "object",
                "properties": {
                    "shipment_id": {"type": "string"},
                    "product_cn_code": {"type": "string", "pattern": "^[0-9]{8}$"},
                    "country_of_origin": {"type": "string", "minLength": 2, "maxLength": 2},
                    "quantity": {"type": "number", "minimum": 0},
                    "unit": {"type": "string"}
                },
                "required": ["shipment_id", "product_cn_code", "country_of_origin", "quantity"]
            }),
            config={"severity": "ERROR"}
        )

    def execute(self, input_data: CBAMDeclarationInput) -> Result:
        with self.tracker.track_operation("intake"):
            # Read file
            self.tracker.track_file_input(input_data.file_path)
            df = pd.read_csv(input_data.file_path)

            self.logger.info(
                "File loaded",
                file_path=input_data.file_path,
                rows=len(df)
            )

            # Validate
            declarations = df.to_dict('records')
            validation_result = self.validator.validate_batch(declarations)

            valid_declarations = []
            errors = []

            for i, (decl, val_result) in enumerate(zip(declarations, validation_result)):
                if val_result.is_valid:
                    valid_declarations.append(decl)
                else:
                    errors.append({
                        "row": i + 1,
                        "errors": [e.message for e in val_result.errors]
                    })

            self.logger.info(
                "Validation complete",
                valid=len(valid_declarations),
                invalid=len(errors)
            )

            # Save provenance
            self.tracker.save_record("provenance/intake.json")

            return Result(
                success=len(errors) == 0,
                data=CBAMDeclarationOutput(
                    declarations=valid_declarations,
                    rows_processed=len(df),
                    validation_errors=errors
                ),
                metadata=Metadata(
                    rows_valid=len(valid_declarations),
                    rows_invalid=len(errors)
                )
            )
```

#### Step 4: Emissions Calculator Agent (agents/emissions_calculator_agent.py)

```python
from greenlang.sdk.base import Agent, Result
from greenlang.services import FactorBroker, FactorRequest
from greenlang.cache import get_cache_manager
from greenlang.telemetry import get_logger, get_metrics_collector
from pydantic import BaseModel

class EmissionsInput(BaseModel):
    declarations: list

class EmissionsOutput(BaseModel):
    declarations_with_emissions: list
    total_emissions_tco2: float

class CBAMEmissionsCalculatorAgent(Agent):
    def __init__(self):
        super().__init__(
            name="cbam-emissions-calculator",
            version="1.0.0"
        )

        self.factor_broker = FactorBroker()
        self.cache = get_cache_manager()
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()

    async def execute(self, input_data: EmissionsInput) -> Result:
        results = []
        total_emissions = 0.0

        for declaration in input_data.declarations:
            # Map CN code to product type
            product_type = self._map_cn_code_to_product(
                declaration['product_cn_code']
            )

            # Get emission factor (with caching)
            factor_key = f"factor:{product_type}:{declaration['country_of_origin']}"
            factor = await self.cache.get_or_compute(
                key=factor_key,
                compute_fn=lambda: self.factor_broker.resolve(
                    FactorRequest(
                        product_type=product_type,
                        region=declaration['country_of_origin'],
                        standard="CBAM"
                    )
                ),
                ttl=86400  # 24 hours
            )

            # Calculate emissions
            emissions_kg = declaration['quantity'] * factor.kgco2e_per_unit
            emissions_tco2 = emissions_kg / 1000

            total_emissions += emissions_tco2

            results.append({
                **declaration,
                'emissions_tco2': emissions_tco2,
                'emission_factor': factor.kgco2e_per_unit,
                'factor_source': factor.source,
                'data_quality_score': factor.dqi_score
            })

            # Metrics
            self.metrics.increment(
                "cbam_declarations_processed",
                labels={"country": declaration['country_of_origin']}
            )
            self.metrics.record(
                "emissions_calculated",
                emissions_tco2,
                labels={"product": product_type}
            )

        self.logger.info(
            "Emissions calculated",
            declarations=len(results),
            total_emissions_tco2=total_emissions
        )

        return Result(
            success=True,
            data=EmissionsOutput(
                declarations_with_emissions=results,
                total_emissions_tco2=total_emissions
            )
        )

    def _map_cn_code_to_product(self, cn_code: str) -> str:
        """Map 8-digit CN code to CBAM product category."""
        # Simplified mapping (full mapping would be extensive)
        cn_prefix = cn_code[:4]
        mapping = {
            "7208": "steel",
            "7210": "steel",
            "2523": "cement",
            "7601": "aluminium",
            "3102": "fertilizer"
        }
        return mapping.get(cn_prefix, "other")
```

#### Step 5: Reporting Agent (agents/reporting_agent.py)

```python
from greenlang.sdk.base import Agent, Result
from greenlang.agents.templates import ReportingAgent, ReportFormat, ComplianceFramework
from pydantic import BaseModel
import pandas as pd

class ReportInput(BaseModel):
    declarations_with_emissions: list
    output_path: str

class CBAMReportingAgent(Agent):
    def __init__(self):
        super().__init__(
            name="cbam-reporting",
            version="1.0.0"
        )
        self.reporter = ReportingAgent()

    async def execute(self, input_data: ReportInput) -> Result:
        df = pd.DataFrame(input_data.declarations_with_emissions)

        # Generate Excel report
        result = await self.reporter.generate_report(
            data=df,
            format=ReportFormat.EXCEL,
            output_path=input_data.output_path,
            check_compliance=[ComplianceFramework.CBAM]
        )

        # Generate summary
        summary = {
            'total_declarations': len(df),
            'total_emissions_tco2': df['emissions_tco2'].sum(),
            'by_country': df.groupby('country_of_origin')['emissions_tco2'].sum().to_dict(),
            'by_product': df.groupby('product_cn_code')['emissions_tco2'].sum().to_dict(),
            'average_dqi': df['data_quality_score'].mean()
        }

        return Result(
            success=True,
            data=summary,
            metadata={'report_path': input_data.output_path}
        )
```

#### Step 6: Pipeline (pipeline.py)

```python
from greenlang.sdk.base import Pipeline
from greenlang.config import ConfigManager
from greenlang.telemetry import get_logger
from agents.intake_agent import CBAMIntakeAgent, CBAMDeclarationInput
from agents.emissions_calculator_agent import CBAMEmissionsCalculatorAgent, EmissionsInput
from agents.reporting_agent import CBAMReportingAgent, ReportInput
import asyncio

class CBAMPipeline:
    def __init__(self):
        self.config = ConfigManager(config_file="config/app_config.yaml")
        self.logger = get_logger(__name__)

        # Create pipeline
        self.pipeline = Pipeline(
            name="cbam-importer",
            version="1.0.0"
        )

        # Add agents
        self.pipeline.add_agent(CBAMIntakeAgent(), name="intake")
        self.pipeline.add_agent(CBAMEmissionsCalculatorAgent(), name="calculator")
        self.pipeline.add_agent(CBAMReportingAgent(), name="reporting")

    async def run(self, file_path: str, declaration_period: str, output_path: str):
        self.logger.info("Starting CBAM pipeline", file_path=file_path)

        result = await self.pipeline.run({
            'intake': CBAMDeclarationInput(
                file_path=file_path,
                declaration_period=declaration_period
            ),
            'calculator': EmissionsInput(
                declarations=None  # Will be populated from intake result
            ),
            'reporting': ReportInput(
                declarations_with_emissions=None,  # From calculator
                output_path=output_path
            )
        })

        if result.success:
            self.logger.info(
                "Pipeline completed successfully",
                total_emissions=result.data['total_emissions_tco2'],
                report_path=result.metadata['report_path']
            )
        else:
            self.logger.error("Pipeline failed", errors=result.errors)

        return result

async def main():
    pipeline = CBAMPipeline()

    result = await pipeline.run(
        file_path="data/cbam_declarations.csv",
        declaration_period="2024-Q1",
        output_path="reports/cbam_q1_2024.xlsx"
    )

    print(f"✓ CBAM pipeline completed!")
    print(f"Total emissions: {result.data['total_emissions_tco2']:.2f} tCO2")
    print(f"Report: {result.metadata['report_path']}")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Step 7: Tests (tests/test_integration.py)

```python
import pytest
from pipeline import CBAMPipeline
from greenlang.testing import create_test_csv

@pytest.mark.asyncio
async def test_cbam_pipeline_integration():
    """Test complete CBAM pipeline."""

    # Create test data
    test_file = create_test_csv([
        {
            "shipment_id": "SHIP-001",
            "product_cn_code": "72081000",
            "country_of_origin": "CN",
            "quantity": 1000,
            "unit": "kg"
        },
        {
            "shipment_id": "SHIP-002",
            "product_cn_code": "25232900",
            "country_of_origin": "TR",
            "quantity": 5000,
            "unit": "kg"
        }
    ])

    # Run pipeline
    pipeline = CBAMPipeline()
    result = await pipeline.run(
        file_path=test_file,
        declaration_period="2024-Q1",
        output_path="test_output.xlsx"
    )

    # Assertions
    assert result.success
    assert result.data['total_declarations'] == 2
    assert result.data['total_emissions_tco2'] > 0
    assert 'CN' in result.data['by_country']
    assert 'TR' in result.data['by_country']
```

#### Step 8: Run Your Application

```bash
# Install dependencies
pip install greenlang pandas pyyaml

# Run pipeline
python pipeline.py

# Output:
# ✓ CBAM pipeline completed!
# Total emissions: 4.56 tCO2
# Report: reports/cbam_q1_2024.xlsx
```

**What you built:**
- ✅ Complete CBAM declaration processing
- ✅ Multi-format data ingestion with validation
- ✅ Emission factor lookups from authoritative sources
- ✅ Comprehensive reporting with compliance checks
- ✅ Complete provenance and audit trails
- ✅ Production-ready observability
- ✅ Tested and validated

**Infrastructure reused:**
- Agent framework (3 agents)
- Validation framework
- Factor Broker service
- Cache manager
- Database connections
- Telemetry system
- Provenance tracking
- Reporting templates

**Lines of custom code:** ~300
**Lines of infrastructure code:** ~2,000 (reused)
**Infrastructure Usage Metric (IUM):** 87%

---

## 11. Testing & Quality Assurance

### 11.1 Testing Infrastructure

**Location:** `greenlang.testing`
**Status:** ✅ Production Ready

```python
from greenlang.testing import (
    AgentTestCase,
    create_mock_llm,
    create_test_data,
    create_test_csv
)

class TestMyAgent(AgentTestCase):
    def setUp(self):
        self.agent = MyAgent()
        self.mock_llm = create_mock_llm()

    async def test_successful_execution(self):
        # Create test data
        test_data = create_test_data({
            "value": 100,
            "unit": "kwh"
        })

        # Run agent
        result = await self.agent.run(test_data)

        # Assertions
        self.assertTrue(result.success)
        self.assertEqual(result.data.value, 200)

    async def test_validation_error(self):
        # Test with invalid data
        test_data = create_test_data({
            "value": -100,  # Invalid
            "unit": "kwh"
        })

        result = await self.agent.run(test_data)

        self.assertFalse(result.success)
        self.assertIn("positive", str(result.errors))
```

### 11.2 Mock Objects

```python
from greenlang.testing.mocks import (
    MockCacheManager,
    MockDatabaseManager,
    MockChatSession
)

class TestAgentWithDependencies(AgentTestCase):
    def setUp(self):
        # Use mocks instead of real infrastructure
        self.mock_cache = MockCacheManager()
        self.mock_db = MockDatabaseManager()
        self.mock_llm = MockChatSession()

        # Inject mocks
        self.agent = MyAgent(
            cache=self.mock_cache,
            db=self.mock_db,
            llm=self.mock_llm
        )

    async def test_with_mocked_cache(self):
        # Setup mock behavior
        self.mock_cache.set_return_value("key", "mocked_value")

        result = await self.agent.run({"id": "123"})

        # Verify mock was called
        self.mock_cache.assert_called_with("key")
```

---

## 12. Production Deployment

### 12.1 Environment Configuration

```yaml
# config/production.yaml
environment: production

database:
  host: ${DB_HOST}
  port: ${DB_PORT}
  database: ${DB_NAME}
  user: ${DB_USER}
  password: ${DB_PASSWORD}
  ssl: true
  pool_size: 50
  max_overflow: 20

cache:
  redis_url: ${REDIS_URL}
  ssl: true
  connection_pool_size: 50

llm:
  provider: ${LLM_PROVIDER}
  api_key: ${LLM_API_KEY}
  rate_limit: 100  # requests per minute
  timeout: 30

telemetry:
  prometheus_port: 9090
  jaeger_endpoint: ${JAEGER_ENDPOINT}
  log_level: INFO
  enable_debug_logs: false

security:
  jwt_secret: ${JWT_SECRET}
  jwt_expiry: 3600
  enable_cors: true
  allowed_origins:
    - https://app.example.com
```

### 12.2 Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "from greenlang.telemetry import HealthChecker; \
                 import sys; \
                 sys.exit(0 if HealthChecker().check_all() else 1)"

# Run application
CMD ["python", "pipeline.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    environment:
      - DB_HOST=postgres
      - REDIS_URL=redis://redis:6379
      - LLM_API_KEY=${LLM_API_KEY}
    ports:
      - "8000:8000"
      - "9090:9090"  # Prometheus metrics
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=climate_app
      - POSTGRES_USER=greenlang
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### 12.3 Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: climate-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: climate-app
  template:
    metadata:
      labels:
        app: climate-app
    spec:
      containers:
      - name: app
        image: climate-app:latest
        ports:
        - containerPort: 8000
        - containerPort: 9090  # Metrics
        env:
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: host
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

---

# PART IV: ADVANCED TOPICS

## 13. Performance Optimization

### 13.1 Caching Strategies

**Rule:** Cache expensive operations (LLM calls, database queries, API calls)

```python
from greenlang.cache import get_cache_manager

cache = get_cache_manager()

# Pattern 1: Get or Compute
emission_factor = await cache.get_or_compute(
    key=f"factor:{fuel_type}:{region}",
    compute_fn=lambda: fetch_expensive_factor(fuel_type, region),
    ttl=86400  # 24 hours
)

# Pattern 2: Decorator-based caching
from greenlang.cache import cached

@cached(ttl=3600, key_prefix="calculation")
async def expensive_calculation(input_data):
    # This will be cached for 1 hour
    return perform_calculation(input_data)

# Pattern 3: Batch prefetch
keys = [f"factor:{ft}:{r}" for ft in fuel_types for r in regions]
factors = await cache.get_batch(keys)
```

### 13.2 Database Optimization

```python
from greenlang.db import QueryOptimizer, get_session

optimizer = QueryOptimizer()

# Pattern 1: Query caching
results = await optimizer.execute_cached(
    query="SELECT * FROM emission_factors WHERE region = :region",
    params={"region": "US"},
    ttl=3600
)

# Pattern 2: Batch loading
async with get_session() as session:
    # Load related data in single query
    stmt = (
        select(EmissionFactor)
        .options(joinedload(EmissionFactor.source))
        .filter(EmissionFactor.region.in_(regions))
    )
    results = await session.execute(stmt)
```

### 13.3 Parallel Processing

```python
from greenlang.agents.templates import CalculatorAgent
import asyncio

agent = CalculatorAgent()

# Parallel batch processing
results = await agent.batch_calculate_parallel(
    formula_name="emissions",
    inputs_list=large_dataset,  # 10,000+ items
    use_processes=True,  # Process pool for CPU-bound
    max_workers=8
)

# Or use asyncio.gather for I/O-bound
tasks = [process_item(item) for item in items]
results = await asyncio.gather(*tasks)
```

**Performance Targets:**
- Cache hit rate: >85%
- Database connection pool utilization: <80%
- LLM response caching: >90% for repeated queries
- Parallel speedup: CPU count × 0.8

---

## 14. Troubleshooting Guide

### 14.1 Common Issues

#### Issue: "Module not found: greenlang"

**Solution:**
```bash
# Verify installation
pip list | grep greenlang

# Reinstall
pip install --upgrade greenlang

# Check Python version (requires 3.9+)
python --version
```

#### Issue: "Database connection pool exhausted"

**Solution:**
```yaml
# config/app_config.yaml
database:
  pool_size: 50  # Increase from default 20
  max_overflow: 20  # Increase overflow
  pool_recycle: 3600  # Recycle connections every hour
```

#### Issue: "Redis connection timeout"

**Solution:**
```python
from greenlang.cache import initialize_cache_manager

initialize_cache_manager(
    enable_l1=True,  # Enable in-memory fallback
    enable_l2=True,
    redis_timeout=5,  # Increase timeout
    redis_retry_on_timeout=True
)
```

#### Issue: "LLM API rate limit exceeded"

**Solution:**
```python
from greenlang.intelligence import ChatSession

session = ChatSession(
    provider="openai",
    rate_limit=50,  # Reduce from default 100
    retry_on_rate_limit=True,
    max_retries=5
)
```

#### Issue: "Out of memory"

**Solution:**
```python
# Use streaming for large files
from greenlang.agents.templates import IntakeAgent

agent = IntakeAgent()

async for chunk in agent.ingest_streaming(
    file_path="large_file.csv",
    chunk_size=1000  # Process 1000 rows at a time
):
    process_chunk(chunk.data)
    del chunk  # Free memory
```

### 14.2 Debug Mode

```python
import logging
from greenlang.telemetry import configure_logging

# Enable debug logging
configure_logging(
    level=logging.DEBUG,
    format="detailed",  # Include file, line number
    output="file",  # Write to file instead of console
    file_path="debug.log"
)
```

### 14.3 Performance Profiling

```python
from greenlang.telemetry import PerformanceMonitor

monitor = PerformanceMonitor()

with monitor.profile("my_operation"):
    # Code to profile
    result = expensive_operation()

# Get report
report = monitor.get_report()
print(f"Duration: {report.duration_ms}ms")
print(f"Memory: {report.memory_mb}MB")
print(f"CPU: {report.cpu_percent}%")
```

---

## 15. Best Practices & Anti-Patterns

### 15.1 DO's

✅ **Always inherit from Agent base class**
```python
from greenlang.sdk.base import Agent

class MyAgent(Agent):  # ✓ Correct
    pass
```

✅ **Use infrastructure for common operations**
```python
# ✓ Correct
from greenlang.cache import get_cache_manager
cache = get_cache_manager()

# ✗ Wrong
cache = {}  # Don't build custom cache
```

✅ **Track provenance for regulatory compliance**
```python
from greenlang.provenance import ProvenanceTracker

tracker = ProvenanceTracker("operation-name")
with tracker.track_operation("calculation"):
    result = calculate()
tracker.save_record("provenance.json")
```

✅ **Use type hints and Pydantic models**
```python
from pydantic import BaseModel

class Input(BaseModel):
    value: float
    unit: str

def process(data: Input) -> Output:
    pass
```

✅ **Use structured logging**
```python
from greenlang.telemetry import get_logger

logger = get_logger(__name__)
logger.info("Event occurred", key="value", metric=123)
```

### 15.2 DON'Ts

❌ **Never use LLM for numeric calculations**
```python
# ✗ WRONG
response = await llm.complete("What is 2.5 * 3.7?")
result = float(response.content)  # Can hallucinate!

# ✓ CORRECT
result = 2.5 * 3.7  # Deterministic
```

❌ **Never hardcode secrets**
```python
# ✗ WRONG
API_KEY = "sk-1234567890"

# ✓ CORRECT
from greenlang.config import ConfigManager
config = ConfigManager()
API_KEY = config.get_secret("api_key")
```

❌ **Never build custom when infrastructure exists**
```python
# ✗ WRONG
def my_cache_get(key):
    return my_dict.get(key)

# ✓ CORRECT
from greenlang.cache import get_cache_manager
cache = get_cache_manager()
value = await cache.get(key)
```

❌ **Never skip validation**
```python
# ✗ WRONG
def process(data):
    return data['value'] * 2  # No validation!

# ✓ CORRECT
from pydantic import BaseModel

class Input(BaseModel):
    value: float

def process(data: Input) -> float:
    return data.value * 2
```

❌ **Never ignore errors**
```python
# ✗ WRONG
try:
    result = risky_operation()
except:
    pass  # Silent failure!

# ✓ CORRECT
try:
    result = risky_operation()
except Exception as e:
    logger.error("Operation failed", error=str(e))
    raise
```

---

# PART V: REFERENCE

## 16. Quick Reference Tables

### 16.1 Infrastructure Component Summary

| Component | Import Path | Use Case | Performance |
|-----------|-------------|----------|-------------|
| Agent | `greenlang.sdk.base.Agent` | Base class for all agents | N/A |
| Pipeline | `greenlang.sdk.base.Pipeline` | Orchestrate multiple agents | N/A |
| ChatSession | `greenlang.intelligence.ChatSession` | LLM operations | 500-5000ms |
| FactorBroker | `greenlang.services.FactorBroker` | Emission factors | <50ms P95 |
| CacheManager | `greenlang.cache.CacheManager` | Multi-tier caching | <1ms (L1) |
| ValidationFramework | `greenlang.validation.ValidationFramework` | Data validation | <1ms per rule |
| ProvenanceTracker | `greenlang.provenance.ProvenanceTracker` | Audit trails | <5ms |
| EntityResolver | `greenlang.services.EntityResolver` | Entity matching | <100ms |

### 16.2 When to Use What

| Task | Use This | Don't Use |
|------|----------|-----------|
| Data ingestion | `IntakeAgent` | Custom pandas code |
| Calculations | `CalculatorAgent` | LLM for math |
| Reporting | `ReportingAgent` | Custom Excel generation |
| LLM operations | `ChatSession` | Direct OpenAI API |
| Caching | `CacheManager` | Dictionary caching |
| Validation | `ValidationFramework` | Manual if/else checks |
| Database | `get_session()` | Direct SQLAlchemy |
| Logging | `get_logger()` | Python logging |
| Metrics | `get_metrics_collector()` | Custom metrics |

### 16.3 Performance Benchmarks

| Operation | Target | Acceptable | Poor |
|-----------|--------|------------|------|
| Cache hit (L1) | <1ms | <5ms | >10ms |
| Cache hit (L2) | <5ms | <10ms | >20ms |
| Database query | <10ms | <50ms | >100ms |
| Factor lookup | <50ms | <100ms | >200ms |
| LLM completion | <2s | <5s | >10s |
| Validation | <1ms | <5ms | >10ms |
| File ingestion (1K rows) | <100ms | <500ms | >1s |

---

## 17. Import Cheat Sheet

### 17.1 Core SDK

```python
# Agent framework
from greenlang.sdk.base import Agent, Pipeline, Result, Metadata, Status

# Context management
from greenlang.sdk.context import Context, Artifact

# Builders
from greenlang.sdk.builder import AgentBuilder, WorkflowBuilder
```

### 17.2 Intelligence

```python
# LLM operations
from greenlang.intelligence import (
    ChatSession,
    ChatMessage,
    Role,
    create_provider
)

# RAG
from greenlang.intelligence.rag import RAGEngine, Chunker, VectorStore

# Optimization
from greenlang.intelligence import (
    SemanticCache,
    PromptCompressor,
    FallbackManager
)
```

### 17.3 Services

```python
# Factor Broker
from greenlang.services import FactorBroker, FactorRequest

# Entity MDM
from greenlang.services import EntityResolver

# Methodologies
from greenlang.services.methodologies import (
    PedigreeMatrixEvaluator,
    MonteCarloSimulator,
    PedigreeScore
)

# PCF Exchange
from greenlang.services import PCFExchangeService
```

### 17.4 Data Infrastructure

```python
# Validation
from greenlang.validation import (
    ValidationFramework,
    SchemaValidator,
    RulesEngine
)

# Cache
from greenlang.cache import (
    CacheManager,
    get_cache_manager,
    initialize_cache_manager
)

# Database
from greenlang.db import (
    get_engine,
    get_session,
    init_db,
    Base
)

# Provenance
from greenlang.provenance import (
    ProvenanceTracker,
    track_provenance
)
```

### 17.5 Observability

```python
# Telemetry
from greenlang.telemetry import (
    get_logger,
    get_metrics_collector,
    TracingManager
)

# Monitoring
from greenlang.monitoring import (
    DashboardOrchestrator,
    AlertEngine
)
```

### 17.6 Agent Templates

```python
from greenlang.agents.templates import (
    IntakeAgent,
    CalculatorAgent,
    ReportingAgent,
    DataFormat,
    ReportFormat,
    ComplianceFramework
)
```

---

## 18. Decision Trees

### 18.1 Should I Use Infrastructure or Build Custom?

```
Do you need this feature?
│
├─> Search infrastructure catalog
│   │
│   ├─> Found exact match?
│   │   └─> ✅ Use infrastructure
│   │
│   └─> Found 80%+ match?
│       ├─> Workaround acceptable?
│       │   └─> ✅ Use infrastructure with workaround
│       │
│       └─> Enhancement needed?
│           ├─> ETA < 2 weeks?
│           │   └─> ✅ Wait for enhancement
│           │
│           └─> ETA > 2 weeks?
│               └─> ⚠️ Build custom (with ADR)
│
└─> No infrastructure found?
    ├─> Generic functionality?
    │   └─> ✅ Request new infrastructure
    │
    └─> App-specific business logic?
        └─> ✅ Build custom (allowed)
```

### 18.2 Which LLM Model Should I Use?

```
What's your use case?
│
├─> Simple classification/extraction?
│   └─> Use GPT-3.5 Turbo (60x cheaper)
│
├─> Complex reasoning/analysis?
│   └─> Use GPT-4 or Claude Opus
│
├─> Long documents (>8K tokens)?
│   └─> Use GPT-4 Turbo (128K) or Claude (200K)
│
├─> Cost-sensitive?
│   └─> Use GPT-3.5 Turbo + semantic caching
│
└─> Regulatory compliance?
    └─> Use temperature=0.0 + tool calling + provenance
```

### 18.3 Which Caching Strategy?

```
What are you caching?
│
├─> Small, frequently accessed data (<100MB)?
│   └─> L1 (in-memory)
│
├─> Shared across instances?
│   └─> L2 (Redis)
│
├─> Large, persistent data?
│   └─> L3 (disk)
│
├─> LLM responses?
│   ├─> Exact prompts repeat?
│   │   └─> L2 (Redis) with SHA-256 keys
│   │
│   └─> Similar prompts?
│       └─> SemanticCache (L3)
│
└─> Database queries?
    └─> QueryOptimizer (L2)
```

---

## APPENDIX A: Application Templates

### Template A1: CBAM Importer
See Step-by-Step Guide in Section 10

### Template A2: CSRD Materiality Assessment

```python
# agents/materiality_agent.py
from greenlang.sdk.base import Agent, Result
from greenlang.intelligence import ChatSession, RAGEngine

class MaterialityAgent(Agent):
    def __init__(self):
        super().__init__(name="csrd-materiality", version="1.0.0")

        self.llm = ChatSession(provider="anthropic", model="claude-3-opus")
        self.rag = RAGEngine(vector_store="weaviate")

    async def execute(self, input_data: dict) -> Result:
        # Load ESRS knowledge base
        esrs_context = await self.rag.query(
            question=f"What are the requirements for {input_data['topic']}?",
            top_k=5
        )

        # Assess materiality
        assessment = await self.llm.complete(
            prompt=f"""
            Assess materiality for:
            Topic: {input_data['topic']}
            Context: {esrs_context.answer}

            Company data:
            - Impact: {input_data['impact_data']}
            - Financial: {input_data['financial_data']}
            - Stakeholders: {input_data['stakeholder_input']}

            Determine if material according to ESRS 1 AR 16.
            """,
            system="You are an ESRS expert with 10 years experience."
        )

        return Result(
            success=True,
            data={
                'is_material': 'material' in assessment.content.lower(),
                'assessment': assessment.content,
                'sources': esrs_context.sources
            }
        )
```

### Template A3: Scope 3 Category Calculator

```python
# agents/scope3_category_agent.py
from greenlang.sdk.base import Agent, Result
from greenlang.services import FactorBroker, FactorRequest

class Scope3CategoryAgent(Agent):
    def __init__(self, category_number: int, category_name: str):
        super().__init__(
            name=f"scope3-cat{category_number}",
            version="1.0.0",
            description=f"Calculate Scope 3 Category {category_number}: {category_name}"
        )
        self.category = category_number
        self.category_name = category_name
        self.factor_broker = FactorBroker()

    async def execute(self, input_data: dict) -> Result:
        results = []

        for item in input_data['activities']:
            # Get emission factor
            factor = await self.factor_broker.resolve(
                FactorRequest(
                    activity_type=item['activity_type'],
                    region=item['region'],
                    scope3_category=self.category
                )
            )

            # Calculate emissions
            emissions = item['quantity'] * factor.kgco2e_per_unit / 1000

            results.append({
                **item,
                'emissions_tco2': emissions,
                'category': self.category,
                'category_name': self.category_name
            })

        return Result(
            success=True,
            data={
                'category': self.category,
                'activities': results,
                'total_emissions': sum(r['emissions_tco2'] for r in results)
            }
        )

# Usage: Create 15 agents for all Scope 3 categories
category1_agent = Scope3CategoryAgent(1, "Purchased Goods and Services")
category2_agent = Scope3CategoryAgent(2, "Capital Goods")
# ... etc for all 15 categories
```

---

## APPENDIX B: Common Recipes

### Recipe B1: Batch Processing with Progress

```python
from greenlang.sdk.base import Agent
from tqdm import tqdm

class BatchProcessor(Agent):
    async def execute(self, input_data: dict) -> Result:
        items = input_data['items']
        results = []

        with tqdm(total=len(items), desc="Processing") as pbar:
            for item in items:
                result = await self.process_item(item)
                results.append(result)
                pbar.update(1)

        return Result(success=True, data=results)
```

### Recipe B2: Error Recovery with Retry

```python
from greenlang.sdk.base import Agent
from tenacity import retry, stop_after_attempt, wait_exponential

class ResilientAgent(Agent):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def execute(self, input_data: dict) -> Result:
        # This will auto-retry on failure
        result = await self.risky_operation(input_data)
        return Result(success=True, data=result)
```

### Recipe B3: Multi-Source Data Aggregation

```python
from greenlang.sdk.base import Pipeline
import asyncio

async def aggregate_from_multiple_sources():
    # Fetch from multiple sources in parallel
    sources = [
        fetch_from_sap(),
        fetch_from_oracle(),
        fetch_from_workday()
    ]

    results = await asyncio.gather(*sources)

    # Merge results
    aggregated = merge_datasets(results)

    return aggregated
```

---

## APPENDIX C: Migration Examples

### From Custom Code to Infrastructure

**Before (Custom):**
```python
import openai

def get_llm_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

**After (Infrastructure):**
```python
from greenlang.intelligence import ChatSession

session = ChatSession(provider="openai", model="gpt-4o")

async def get_llm_response(prompt):
    response = await session.complete(prompt)
    return response.content
    # Bonus: Automatic cost tracking, retry, caching!
```

---

## APPENDIX D: Glossary

| Term | Definition |
|------|------------|
| **Agent** | Self-contained processing unit with inputs/outputs |
| **Pipeline** | Orchestrated sequence of agents |
| **IUM** | Infrastructure Usage Metric (% of code using infrastructure) |
| **Provenance** | Complete audit trail from input to output |
| **DQI** | Data Quality Indicator (1-5 scale) |
| **RAG** | Retrieval-Augmented Generation (LLM + knowledge base) |
| **CBAM** | Carbon Border Adjustment Mechanism (EU regulation) |
| **CSRD** | Corporate Sustainability Reporting Directive (EU) |
| **GHG Protocol** | Greenhouse Gas Protocol (accounting standard) |
| **Scope 1/2/3** | Direct, indirect energy, indirect value chain emissions |

---

## APPENDIX E: Support & Resources

### Documentation
- **Full Infrastructure Guide**: GL-INFRASTRUCTURE.md
- **Component Catalog**: GREENLANG_INFRASTRUCTURE_CATALOG.md
- **FAQ**: INFRASTRUCTURE_FAQ.md
- **API Reference**: docs/API_REFERENCE.md

### Examples
- **Quick Start Examples**: examples/quick-start/
- **Integration Examples**: examples/integrations/
- **Application Templates**: templates/

### Community
- **GitHub**: https://github.com/greenlang/platform
- **Issues**: Report bugs and request features
- **Discussions**: Ask questions, share patterns

### Enterprise Support
- **Email**: enterprise@greenlang.com
- **Slack**: #greenlang-support (for licensed customers)

---

## Document Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2025-11-09 | Complete rewrite as comprehensive reference |
| 1.0.0 | 2025-01-26 | Initial infrastructure guide |

---

**Last Updated:** November 9, 2025
**Maintained By:** GreenLang Infrastructure Team
**License:** MIT

---

## Quick Start Reminder

**Building your first app?** Jump to [30-Minute Quick Start](#2-30-minute-quick-start)

**Looking for a specific component?** Jump to [Infrastructure Catalog](#part-ii-infrastructure-catalog)

**Ready to deploy?** Jump to [Production Deployment](#12-production-deployment)

**Having issues?** Jump to [Troubleshooting Guide](#14-troubleshooting-guide)

**Need examples?** See [Appendix A: Application Templates](#appendix-a-application-templates)

---

🎉 **You're ready to build world-class climate applications with GreenLang!**
