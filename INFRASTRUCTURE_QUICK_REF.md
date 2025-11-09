# GreenLang Infrastructure Quick Reference

**One-Page Cheat Sheet for Common Tasks**

Version: 1.0.0 | Last Updated: November 9, 2025

---

## Core Principle

**Always use GreenLang infrastructure. Never build custom when infrastructure exists.**

Before writing ANY code, check: [GREENLANG_INFRASTRUCTURE_CATALOG.md](GREENLANG_INFRASTRUCTURE_CATALOG.md)

---

## Common Tasks → Infrastructure

### Need LLM / AI Capabilities?

→ **`greenlang.intelligence.ChatSession`**

```python
from greenlang.intelligence import ChatSession
session = ChatSession(provider="openai", temperature=0.0)
response = session.complete("Explain GHG Protocol Scope 3")
```

**Use for:** Narratives, categorization, Q&A, insights
**Never for:** Calculations, compliance decisions

---

### Need to Build an Agent?

→ **`greenlang.sdk.base.Agent`**

```python
from greenlang.sdk.base import Agent

class MyAgent(Agent):
    def execute(self, input_data):
        # Your logic here
        return result

agent = MyAgent()
result = agent.run(input_data)
```

**Automatic:** Error handling, provenance, retry, telemetry

---

### Need to Cache Data?

→ **`greenlang.cache.CacheManager`**

```python
from greenlang.cache import CacheManager
cache = CacheManager()
cache.set("key", value, ttl=3600)
value = cache.get("key")
```

**Use for:** LLM responses, DB queries, emission factors, sessions

---

### Need to Validate Data?

→ **`greenlang.validation.ValidationFramework`**

```python
from greenlang.validation import ValidationFramework, ValidationRule
validator = ValidationFramework(rules=[
    ValidationRule(field="value", rule_type="positive")
])
result = validator.validate(data)
```

**50+ built-in rules:** positive, enum, range, date_range, email, etc.

---

### Need Authentication?

→ **`greenlang.auth.AuthManager`**

```python
from greenlang.auth import AuthManager
auth = AuthManager()
token = auth.authenticate(username, password)
user = auth.verify_token(token)
```

**Features:** JWT tokens, RBAC, API keys, audit logs

---

### Need Database Access?

→ **`greenlang.db.DatabaseManager`**

```python
from greenlang.db import DatabaseManager
db = DatabaseManager(provider="postgresql")
results = db.query("SELECT * FROM emissions WHERE id = %s", [123])
```

**Features:** Connection pooling, retry, transactions, migration

---

### Need Configuration?

→ **`greenlang.config.ConfigManager`**

```python
from greenlang.config import ConfigManager
config = ConfigManager(environment="production")
db_host = config.get("database.host")
api_key = config.get_secret("openai.api_key")
```

**Never hardcode:** Secrets, endpoints, settings

---

### Need to Build a Pipeline?

→ **`greenlang.pipeline.PipelineOrchestrator`**

```python
from greenlang.pipeline import PipelineOrchestrator, PipelineStep

pipeline = PipelineOrchestrator(name="my-pipeline")
pipeline.add_step(PipelineStep(name="step1", agent=Agent1()))
pipeline.add_step(PipelineStep(name="step2", agent=Agent2(), dependencies=["step1"]))
result = pipeline.run(input_data)
```

**Features:** Dependency management, parallel execution, checkpointing

---

### Need to Generate Reports?

→ **`greenlang.reporting.ReportGenerator`**

```python
from greenlang.reporting import ReportGenerator
generator = ReportGenerator()
generator.generate_pdf(template="report.html", data=data, output="report.pdf")
generator.generate_excel(sheets={"Sheet1": df}, output="report.xlsx")
```

**Formats:** PDF, Excel, XBRL, Markdown

---

### Need Monitoring?

→ **`greenlang.monitoring.TelemetryManager`**

```python
from greenlang.monitoring import TelemetryManager
telemetry = TelemetryManager(service_name="my-app")
telemetry.record_counter("emissions_calculated")
with telemetry.trace_span("calculate"):
    result = calculate()
```

**Automatic:** Prometheus metrics, OpenTelemetry traces, structured logs

---

### Need Time Series Forecasting?

→ **`greenlang.agents.ForecastAgentSARIMA`**

```python
from greenlang.agents import ForecastAgentSARIMA
agent = ForecastAgentSARIMA(auto_tune=True)
forecast = agent.run({"historical_data": df, "forecast_periods": 12})
```

**Use for:** Emissions trends, energy forecasts, budget predictions

---

### Need Anomaly Detection?

→ **`greenlang.agents.AnomalyAgentIForest`**

```python
from greenlang.agents import AnomalyAgentIForest
agent = AnomalyAgentIForest()
result = agent.run({"data": df, "features": ["emissions", "cost"]})
anomalies = result.anomalies
```

**Use for:** Data quality, fraud detection, outliers

---

### Need Emission Factors?

→ **`greenlang.data.EmissionFactorDatabase`**

```python
from greenlang.data import EmissionFactorDatabase
efdb = EmissionFactorDatabase()
factor = efdb.get_factor(activity="electricity", region="UK", year=2024)
```

**100,000+ factors:** DEFRA, EPA, IEA, IPCC, Ecoinvent

---

### Need ERP Integration?

→ **SAP:** `vcci_scope3.connectors.sap.SAPConnector`
→ **Oracle:** `vcci_scope3.connectors.oracle.OracleConnector`
→ **Workday:** `vcci_scope3.connectors.workday.WorkdayConnector`

```python
from vcci_scope3.connectors.sap import SAPConnector
sap = SAPConnector(base_url=url, username=user, password=pwd)
data = sap.extract_procurement_data(fiscal_year=2024)
```

---

### Need RAG (Retrieval-Augmented Generation)?

→ **`greenlang.intelligence.RAGManager`**

```python
from greenlang.intelligence import RAGManager
rag = RAGManager(vector_db="weaviate")
rag.index_documents(documents)
response = rag.query("What are Scope 3 categories?", top_k=3)
```

**Use for:** Knowledge base Q&A, citation-backed answers

---

### Need Semantic Search?

→ **`greenlang.intelligence.EmbeddingService`**

```python
from greenlang.intelligence import EmbeddingService
embedder = EmbeddingService(provider="openai")
embedding = embedder.embed("text to embed")
similarities = embedder.cosine_similarity(query_emb, doc_embs)
```

**Use for:** Entity resolution, duplicate detection, clustering

---

## Decision Tree

```
Need to write code?
├─ Is this LLM/AI related?
│  ├─ Yes → ChatSession, RAGManager, EmbeddingService
│  └─ No → Continue
│
├─ Is this an agent?
│  ├─ Yes → greenlang.sdk.base.Agent
│  └─ No → Continue
│
├─ Is this data processing?
│  ├─ Validation → ValidationFramework
│  ├─ Transformation → DataTransformer
│  ├─ Storage → DatabaseManager
│  └─ Caching → CacheManager
│
├─ Is this infrastructure?
│  ├─ Auth → AuthManager
│  ├─ Config → ConfigManager
│  ├─ Monitoring → TelemetryManager
│  └─ API → FastAPI Integration
│
├─ Is this reporting?
│  ├─ Yes → ReportGenerator
│  └─ No → Continue
│
├─ Is this ML?
│  ├─ Forecasting → ForecastAgentSARIMA
│  ├─ Anomaly Detection → AnomalyAgentIForest
│  └─ Custom → Request new infrastructure
│
└─ Still not sure?
   └─ Read GREENLANG_INFRASTRUCTURE_CATALOG.md
```

---

## Anti-Patterns (DON'T DO THIS!)

### ❌ Custom LLM Code

```python
# WRONG
import openai
response = openai.ChatCompletion.create(...)
```

### ✅ Use ChatSession

```python
# CORRECT
from greenlang.intelligence import ChatSession
session = ChatSession(provider="openai")
response = session.complete("...")
```

---

### ❌ LLM for Calculations

```python
# WRONG
response = llm.complete("What is 100 * 0.18?")
result = parse_number(response)  # Hallucination risk!
```

### ✅ Python Arithmetic

```python
# CORRECT
result = 100 * 0.18  # Deterministic
```

---

### ❌ Custom Agent Class

```python
# WRONG
class MyAgent:
    def __init__(self):
        self.name = "my-agent"
    def run(self, data):
        # Custom error handling, logging...
```

### ✅ Inherit from Agent

```python
# CORRECT
from greenlang.sdk.base import Agent
class MyAgent(Agent):
    def execute(self, input_data):
        # Automatic error handling, provenance, telemetry
```

---

### ❌ Hardcoded Secrets

```python
# WRONG
API_KEY = "sk-abc123..."
DB_PASSWORD = "password123"
```

### ✅ Environment Variables

```python
# CORRECT
from greenlang.config import ConfigManager
config = ConfigManager()
api_key = config.get_secret("openai.api_key")
```

---

### ❌ Custom Validation

```python
# WRONG
def validate(data):
    errors = []
    if data["value"] < 0:
        errors.append("Negative value")
    # ... 50 more checks
```

### ✅ Validation Framework

```python
# CORRECT
from greenlang.validation import ValidationFramework
validator = ValidationFramework(rules=rules_from_yaml)
result = validator.validate(data)
```

---

## When Custom Code is Allowed

1. **Business Logic** - Domain-specific calculations unique to your app
2. **UI/UX** - User interface code
3. **Integration Logic** - Glue code between infrastructure components
4. **After ADR Approval** - Architecture Decision Record required

**Process for Custom Code:**
1. Search infrastructure catalog
2. Request feature enhancement if infrastructure is close
3. Write ADR justifying custom code
4. Get approval from architecture team
5. Document in app README

---

## Quick Stats

- **Infrastructure Components:** 100+
- **LOC Available:** 50,000+
- **Target IUM:** 80%
- **Typical Code Reduction:** 70-95%
- **Typical Time Savings:** 60-80%

---

## Resources

- **Full Catalog:** [GREENLANG_INFRASTRUCTURE_CATALOG.md](GREENLANG_INFRASTRUCTURE_CATALOG.md) (5000+ lines)
- **Onboarding:** [DEVELOPER_ONBOARDING.md](DEVELOPER_ONBOARDING.md)
- **Tutorials:** [INFRASTRUCTURE_TUTORIAL.md](INFRASTRUCTURE_TUTORIAL.md)
- **FAQ:** [INFRASTRUCTURE_FAQ.md](INFRASTRUCTURE_FAQ.md)
- **Examples:** `examples/` directory

---

## Getting Help

- **Discord:** #infrastructure channel
- **GitHub:** Issues tagged `infrastructure`
- **Email:** infrastructure@greenlang.io
- **Office Hours:** Tuesdays 2-3pm PT

---

**Remember: Search First, Build Last!**

Print this page and keep it at your desk. 90% of what you need is already built.
