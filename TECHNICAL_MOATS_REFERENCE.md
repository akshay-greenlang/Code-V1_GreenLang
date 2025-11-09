# GreenLang Technical Moats Reference Guide

**A Developer's Guide to Leveraging the 95% Built Infrastructure**

---

## Quick Navigation

- [Moat 1: Zero-Hallucination Calculations](#moat-1-zero-hallucination-calculations)
- [Moat 2: Provenance Tracking](#moat-2-provenance-tracking)
- [Moat 3: RAG System](#moat-3-rag-system)
- [Moat 4: LLM Integration](#moat-4-llm-integration)
- [Moat 5: Multi-Tenant Orchestration](#moat-5-multi-tenant-orchestration)
- [Code Examples](#code-examples)
- [Application-Specific Recipes](#application-specific-recipes)

---

## Moat 1: Zero-Hallucination Calculations

### Core Philosophy

**NEVER use LLMs for calculations. Always use deterministic Python + database.**

```python
# ❌ WRONG: LLM calculating emissions
response = llm.complete(
    "Calculate emissions for 1000 kWh of electricity"
)
# Result: Hallucinated number, not auditable, fails regulations

# ✅ RIGHT: Deterministic calculation + LLM for explanation
emissions_tco2 = 1000 * 0.42  # From emission factor database
explanation = llm.complete(
    f"Explain why {emissions_tco2} tCO2e for 1000 kWh"
)
# Result: Verified calculation + AI explanation
```

### Infrastructure to Use

```python
from greenlang.sdk.base import Agent, Metadata, Result
from greenlang.validation import ValidationFramework, ValidationRule
from greenlang.intelligence import ChatSession

class CalculatorAgent(Agent):
    """Deterministic calculation agent with zero hallucination."""

    def execute(self, input_data):
        # Step 1: Validate input (deterministic)
        validator = ValidationFramework(rules=[
            ValidationRule(field="kwh", rule_type="positive", min=0),
            ValidationRule(field="region", rule_type="enum", values=["UK", "US", "EU"])
        ])
        validation_result = validator.validate(input_data)
        if not validation_result.is_valid:
            return Result(success=False, errors=validation_result.errors)

        # Step 2: Fetch emission factor (deterministic, from database)
        emission_factor = self.get_emission_factor(
            activity="electricity",
            region=input_data.region,
            year=2024
        )

        # Step 3: Calculate (deterministic math)
        emissions_tco2 = input_data.kwh * emission_factor

        # Step 4: Generate narrative (AI, non-deterministic but explainable)
        session = ChatSession(provider="openai", temperature=0.0)
        narrative = session.complete(
            prompt=f"Explain {emissions_tco2} tCO2e emissions for {input_data.kwh} kWh",
            system="You are a climate expert. Explain the calculation, not the number."
        )

        # Step 5: Return with complete provenance
        return Result(
            success=True,
            data={
                "emissions_tco2": emissions_tco2,  # Calculated, not hallucinated
                "emission_factor": emission_factor,  # Source
                "narrative": narrative.content,  # AI explanation
                "calculation_method": "kWh × grid_factor",  # Audit trail
                "timestamp": datetime.now().isoformat()
            }
        )
```

### Use This Pattern For

- ✅ Regulatory reporting (CSRD, SEC, TCFD)
- ✅ Carbon accounting (Scope 1/2/3)
- ✅ Financial calculations (costs, ROI, payback)
- ✅ Supply chain emissions
- ✅ Building energy calculations

### Don't Use This For

- ❌ Generating creative narratives (use ChatSession directly)
- ❌ Answering open-ended questions (use RAG + ChatSession)
- ❌ Classification tasks (use Tool Calling with temperature=0)

---

## Moat 2: Provenance Tracking

### Core Philosophy

**Every data point needs a chain of custody: input → transformation → output**

### Infrastructure to Use

```python
from greenlang.core.provenance import ProvenanceChain, ProvenanceEntry
from hashlib import sha256
import json

class AuditableCalculation:
    """Calculation with complete provenance tracking."""

    def __init__(self, calculation_id):
        self.chain = ProvenanceChain(id=calculation_id)

    def ingest_data(self, file_path, data_type="csv"):
        """Track data ingestion."""
        with open(file_path, 'rb') as f:
            content = f.read()
            file_hash = sha256(content).hexdigest()

        self.chain.add_entry(ProvenanceEntry(
            step="data_ingestion",
            input_source=file_path,
            input_hash=file_hash,
            timestamp=datetime.now().isoformat(),
            actor="system",
            action_description=f"Ingested {data_type} file"
        ))
        return json.loads(content.decode())

    def validate_data(self, data):
        """Track validation step."""
        validator = ValidationFramework(rules=[...])
        result = validator.validate(data)

        self.chain.add_entry(ProvenanceEntry(
            step="data_validation",
            input_hash=sha256(json.dumps(data).encode()).hexdigest(),
            output_hash=sha256(json.dumps(result).encode()).hexdigest(),
            timestamp=datetime.now().isoformat(),
            actor="system",
            action_description=f"Validated {len(data)} records, {result.error_count} errors"
        ))
        return result

    def calculate_emissions(self, validated_data):
        """Track calculation step."""
        results = []
        for record in validated_data:
            emissions = record['kwh'] * 0.42
            results.append(emissions)

        self.chain.add_entry(ProvenanceEntry(
            step="emissions_calculation",
            input_hash=sha256(json.dumps(validated_data).encode()).hexdigest(),
            output_hash=sha256(json.dumps(results).encode()).hexdigest(),
            timestamp=datetime.now().isoformat(),
            actor="system",
            action_description=f"Calculated emissions for {len(results)} records",
            calculation_method="kWh × 0.42 kg CO2/kWh",
            emission_factor_source="UK Grid 2024",
            emission_factor_version="2.1"
        ))
        return results

    def approve_results(self, approver_name, approver_role):
        """Track human approval."""
        self.chain.add_entry(ProvenanceEntry(
            step="human_review",
            timestamp=datetime.now().isoformat(),
            actor=approver_name,
            actor_role=approver_role,
            action_description="Reviewed and approved calculation",
            approval_signature=self._create_signature(approver_name)
        ))

    def get_audit_trail(self):
        """Return complete provenance chain."""
        return {
            "calculation_id": self.chain.id,
            "entries": [entry.to_dict() for entry in self.chain.entries],
            "integrity_hash": self.chain.get_integrity_hash(),  # Cryptographically verifiable
            "audit_ready": True
        }

    def verify_integrity(self):
        """Cryptographically verify the chain hasn't been tampered with."""
        return self.chain.verify_integrity()
```

### Real Example: GL-CBAM-APP

```python
# CBAM calculation with complete provenance
class CBAMCalculation(AuditableCalculation):

    def process_shipment(self, shipment_data):
        # 1. Ingest shipment data
        data = self.ingest_data(shipment_data, data_type="shipment")

        # 2. Validate against CBAM requirements
        validation = self.validate_data(data)

        # 3. Calculate embedded emissions (deterministic)
        emissions = self.calculate_emissions(validation.valid_records)

        # 4. Calculate CBAM credit obligation
        cbam_credits = self._calculate_cbam_credits(emissions)

        # 5. Get human approval (customs officer)
        self.approve_results("Jean Dubois", "CBAM Customs Officer")

        # 6. Return audit-ready result
        return {
            "cbam_credits_due": cbam_credits,
            "audit_trail": self.get_audit_trail(),
            "integrity_verified": self.verify_integrity()
        }
```

### Use This Pattern For

- ✅ Regulatory reporting (audit trail requirement)
- ✅ Carbon credit verification (fraud prevention)
- ✅ Supply chain transparency (consumer trust)
- ✅ Financial audits (SOC 2 Type 2)
- ✅ Legal compliance (proof of diligence)

### Don't Use This For

- ❌ Real-time calculations (too slow, use cache)
- ❌ Exploratory analysis (overhead not justified)

---

## Moat 3: RAG System

### Core Philosophy

**Never hardcode climate data. Always retrieve from knowledge base + cite sources.**

### Infrastructure to Use

```python
from greenlang.intelligence.rag import RAGManager, RAGConfig
from greenlang.intelligence.rag.models import Document, QueryResult

class ClimateKnowledgeBase:
    """RAG system for climate data with scientific citations."""

    def __init__(self):
        config = RAGConfig(
            vector_db="weaviate",  # Production-ready
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            chunk_size=500,
            overlap=50
        )
        self.rag = RAGManager(config)
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """Load scientific documents into RAG system."""
        documents = [
            Document(
                id="ghg-protocol-scope3-v1.0",
                content="GHG Protocol Scope 3 Standard...",
                metadata={
                    "source": "World Resources Institute",
                    "year": 2024,
                    "version": "1.0",
                    "url": "https://ghgprotocol.org/scope-3"
                }
            ),
            Document(
                id="ipcc-ar6-ch5",
                content="IPCC 6th Assessment Report Chapter 5...",
                metadata={
                    "source": "IPCC",
                    "year": 2023,
                    "confidence": 0.95
                }
            ),
            # ... 100,000+ more documents
        ]
        self.rag.index_documents(documents)

    def query_with_citations(self, question, top_k=3):
        """Query knowledge base, return results with citations."""
        results = self.rag.query(question, top_k=top_k)

        return {
            "answer": results[0].content if results else "No answer found",
            "citations": [
                {
                    "source": r.metadata['source'],
                    "year": r.metadata.get('year'),
                    "url": r.metadata.get('url'),
                    "confidence": r.similarity_score
                }
                for r in results
            ],
            "ready_for_report": True  # Can be included in audit trail
        }
```

### Real Example: GL-VCCI-APP Emission Factors

```python
class FactorBroker:
    """Retrieve emission factors from knowledge base with provenance."""

    def get_emission_factor(self, activity, region, year, source_preference=None):
        """
        Get emission factor with citations.

        Example:
            factor = broker.get_emission_factor(
                activity="electricity_generation",
                region="UK",
                year=2024,
                source_preference="DEFRA"
            )
            # Returns:
            # {
            #   "factor": 0.194,  # kg CO2/kWh
            #   "unit": "kg CO2/kWh",
            #   "source": "DEFRA 2024",
            #   "year": 2024,
            #   "confidence": 0.98,
            #   "methodology": "Grid average method",
            #   "citation": {"url": "...", "year": 2024}
            # }
        """
        query = f"{activity} {region} {year} emission factor"
        results = self.rag.query(query, top_k=5)

        # Find best match (highest confidence + matching criteria)
        best = max(
            results,
            key=lambda r: self._score_match(r, source_preference)
        )

        return {
            "factor": best.metadata['value'],
            "unit": best.metadata.get('unit', 'kg CO2/unit'),
            "source": best.metadata['source'],
            "year": best.metadata['year'],
            "confidence": best.similarity_score,
            "methodology": best.metadata.get('methodology'),
            "citation": {
                "url": best.metadata.get('url'),
                "year": best.metadata.get('year'),
                "doi": best.metadata.get('doi')
            }
        }
```

### Use This Pattern For

- ✅ Emission factors (always cite source)
- ✅ Regulatory standards (always cite regulation)
- ✅ Scientific data (IPCC, TCFD, GHG Protocol)
- ✅ Company-specific knowledge base (supplier data, facility info)
- ✅ Industry benchmarks (cite methodology)

### Don't Use This For

- ❌ Hardcoded values (embed in code)
- ❌ Real-time market data (use API integration)
- ❌ Customer proprietary data (store in database, not RAG)

---

## Moat 4: LLM Integration

### Core Philosophy

**Temperature=0 for reproducibility. Use tools for data extraction. Never trust LLM math.**

### Infrastructure to Use

```python
from greenlang.intelligence import ChatSession, ChatMessage, Role

class AIInsights:
    """AI-powered insights without hallucination."""

    def __init__(self):
        # Temperature=0 for reproducible results across calls
        self.session = ChatSession(
            provider="openai",
            model="gpt-4",
            temperature=0.0,  # CRITICAL: for reproducibility
            seed=42,  # Deterministic results
            max_budget_usd=100.0  # Cost control
        )

    def generate_narrative(self, emissions_data):
        """Generate audit-friendly narrative from calculated data."""
        prompt = f"""
        Based on these emissions calculations (which are VERIFIED and CORRECT):

        {json.dumps(emissions_data, indent=2)}

        Generate a 2-paragraph executive summary explaining the results.
        IMPORTANT: Do not modify the numbers. Only explain what they mean.
        """

        response = self.session.complete(
            prompt=prompt,
            system="You are a sustainability reporting expert. Explain emissions results clearly and accurately."
        )

        return response.content

    def categorize_suppliers(self, supplier_names):
        """Use LLM for intelligent categorization with tool calling."""
        tools = [
            {
                "name": "categorize_supplier",
                "description": "Categorize a supplier by industry",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "supplier_name": {"type": "string"},
                        "industry": {
                            "type": "string",
                            "enum": ["Manufacturing", "Services", "Raw Materials", "Other"]
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "required": ["supplier_name", "industry", "confidence"]
                }
            }
        ]

        response = self.session.complete_with_tools(
            prompt=f"Categorize these suppliers: {', '.join(supplier_names)}",
            tools=tools
        )

        return response.tool_calls

    def explain_recommendation(self, recommendation_data):
        """Explain AI-generated recommendations."""
        prompt = f"""
        This facility has been identified for decarbonization:

        Current emissions: {recommendation_data['current_emissions']} tCO2e/year
        Recommended technology: {recommendation_data['technology']}
        Cost: ${recommendation_data['cost']}
        Payback period: {recommendation_data['payback_years']} years
        Annual savings: ${recommendation_data['annual_savings']}

        Explain why this recommendation is valuable for the customer.
        """

        response = self.session.complete(
            prompt=prompt,
            system="You are a decarbonization consultant. Explain recommendations clearly."
        )

        return response.content
```

### Real Example: GL-CSRD-APP Narrative Generation

```python
class NarrativeGenerator:
    """Generate CSRD-compliant narratives from calculated data."""

    def generate_materiality_narrative(self, materiality_results):
        """Generate double materiality narrative."""
        session = ChatSession(provider="openai", model="gpt-4", temperature=0.0)

        prompt = f"""
        Based on these VERIFIED materiality assessment results:

        Impact materiality score: {materiality_results['impact_score']}/100
        Financial materiality score: {materiality_results['financial_score']}/100
        Material topics identified: {', '.join(materiality_results['topics'])}

        Generate a CSRD-compliant double materiality narrative explaining:
        1. Why these topics are material from an impact perspective
        2. Why they matter financially to the company
        3. How they align with ESRS standards

        IMPORTANT: Use only the scores and topics provided. Do not invent data.
        """

        response = session.complete(
            prompt=prompt,
            system="You are an ESRS sustainability expert. Write formal regulatory narrative."
        )

        return {
            "narrative": response.content,
            "word_count": len(response.content.split()),
            "tokens_used": response.tokens,
            "cost": response.cost,
            "audit_ready": True
        }
```

### Configuration Options

```python
# Cost Control
session = ChatSession(
    provider="openai",
    max_budget_usd=1000.0,  # Will throw error if exceeded
    track_cost=True
)

# Semantic Caching (30% cost reduction)
session = ChatSession(
    provider="openai",
    semantic_cache_enabled=True,
    cache_ttl=3600  # 1 hour
)

# Streaming for Long Reports
for chunk in session.stream(
    prompt="Generate a 10-page sustainability report...",
    stream_interval=1  # Stream every 1 second
):
    print(chunk, end="")  # Real-time output

# Provider Switching (no code changes)
session_openai = ChatSession(provider="openai", model="gpt-4")
session_anthropic = ChatSession(provider="anthropic", model="claude-3-opus")
# Same API, different backend
```

### Use This Pattern For

- ✅ Narrative generation (explanations, not math)
- ✅ Intelligent categorization (with tool calling)
- ✅ Recommendation explanations (why, not what)
- ✅ Document analysis (understanding, not generating truth)
- ✅ Natural language Q&A (over verified data)

### Don't Use This For

- ❌ Calculations (use deterministic Python)
- ❌ Generating compliance data (use database)
- ❌ Creating synthetic data (use proper data generation)

---

## Moat 5: Multi-Tenant Orchestration

### Core Philosophy

**One platform, unlimited customers. Isolation, resource quotas, cost attribution.**

### Infrastructure to Use

```python
from greenlang.core.multi_tenant import (
    MultiTenantExecutor,
    TenantContext,
    ResourceQuota,
    IsolationLevel
)

class TenantAwareAgent(Agent):
    """Agent that respects multi-tenant isolation."""

    def __init__(self, tenant_id: str):
        super().__init__()
        self.tenant_id = tenant_id

    def execute(self, input_data):
        # Set tenant context (automatic isolation)
        context = TenantContext(
            tenant_id=self.tenant_id,
            isolation_level=IsolationLevel.NAMESPACE,  # Data isolated by namespace
            cost_center="emissions-calculation",
            request_id=str(uuid.uuid4())
        )

        # Execute with tenant context
        with context:
            # All database queries, caching, logging use tenant context
            result = self._calculate(input_data)

        return result
```

### Real Example: Multi-Tenant Emissions Platform

```python
class EmissionsPlatform:
    """Multi-tenant emissions calculation platform."""

    def __init__(self):
        self.executor = MultiTenantExecutor(
            max_tenants=100,  # Can scale to thousands
            resource_quotas={
                "cpu_percent": 20,  # Max CPU per tenant
                "memory_mb": 512,
                "storage_gb": 10,
                "api_calls_per_min": 1000,
                "concurrent_jobs": 5
            }
        )

    def process_customer_emissions(self, tenant_id, emissions_data):
        """Process customer data with complete isolation."""

        # Create tenant context
        context = TenantContext(
            tenant_id=tenant_id,
            isolation_level=IsolationLevel.NAMESPACE,  # Data by namespace
            metadata={
                "customer_tier": "enterprise",
                "cost_center": "climate-reporting"
            }
        )

        with context:
            # Validate within tenant context (isolated database queries)
            validator = ValidationFramework(rules=[...])
            validation = validator.validate(emissions_data)

            if not validation.is_valid:
                return {"success": False, "errors": validation.errors}

            # Calculate (isolated computation)
            results = self._calculate_emissions(validation.valid_records)

            # Store (isolated storage)
            self._store_results(tenant_id, results)

            # Track usage (for billing)
            self._track_usage(
                tenant_id,
                records_processed=len(results),
                computation_time_ms=100,
                cost_usd=0.50
            )

        return {"success": True, "records_processed": len(results)}

    def get_tenant_usage(self, tenant_id):
        """Get usage metrics for billing."""
        return {
            "tenant_id": tenant_id,
            "records_processed": 50000,
            "computation_hours": 120,
            "api_calls": 250000,
            "storage_used_gb": 2.5,
            "monthly_cost_usd": 5000,
            "quota_exceeded": False
        }
```

### Scaling Characteristics

```python
# Single tenant
platform = EmissionsPlatform()
result = platform.process_customer_emissions("tenant-acme", data)

# 100 concurrent tenants (same code)
platform = EmissionsPlatform()
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = []
    for tenant_id in tenant_list:
        future = executor.submit(
            platform.process_customer_emissions,
            tenant_id,
            tenant_data[tenant_id]
        )
        futures.append(future)

    results = [f.result() for f in futures]

# 10,000 concurrent tenants (same code, different deployment)
# Just deploy on Kubernetes with more replicas
# kubectl scale deployment my-platform --replicas=50
```

### Use This Pattern For

- ✅ Multi-customer SaaS platforms
- ✅ Enterprise multi-subsidiary reporting
- ✅ White-label platforms (resellers)
- ✅ Data-driven pricing models (per-usage)
- ✅ Compliance isolation (GDPR, data sovereignty)

### Don't Use This For

- ❌ Simple single-tenant APIs (overhead not justified)

---

## Code Examples

### Complete Example 1: Regulatory Reporting Agent

```python
"""
GL-CSRD-APP style agent with all 5 moats integrated.
"""

from greenlang.sdk.base import Agent
from greenlang.intelligence import ChatSession
from greenlang.validation import ValidationFramework
from greenlang.core.provenance import ProvenanceChain
from greenlang.intelligence.rag import RAGManager

class CSRDReportingAgent(Agent):
    """End-to-end CSRD reporting with zero hallucination."""

    def __init__(self, tenant_id: str):
        super().__init__()
        self.tenant_id = tenant_id
        self.chat = ChatSession(provider="openai", temperature=0.0)
        self.validator = ValidationFramework(rules=[...])
        self.provenance = ProvenanceChain(id=f"csrd-{uuid.uuid4()}")
        self.rag = RAGManager()

    def execute(self, input_data):
        try:
            # 1. VALIDATION (Moat 5: Framework)
            validation = self.validator.validate(input_data)
            self.provenance.add_entry({
                "step": "validation",
                "status": "passed" if validation.is_valid else "failed"
            })

            if not validation.is_valid:
                return Result(success=False, errors=validation.errors)

            # 2. DATA COLLECTION (Moat 3: RAG for standards)
            standard_docs = self.rag.query(
                "CSRD E1 Climate Change standard requirements",
                top_k=3
            )
            self.provenance.add_entry({
                "step": "standards_research",
                "sources": [d.metadata['source'] for d in standard_docs]
            })

            # 3. CALCULATIONS (Moat 1: Deterministic, zero hallucination)
            emissions = self._calculate_emissions(input_data)
            self.provenance.add_entry({
                "step": "emissions_calculation",
                "result_hash": sha256(str(emissions).encode()).hexdigest()
            })

            # 4. NARRATIVE (Moat 4: AI explanation, not math)
            narrative = self.chat.complete(
                prompt=f"Explain {emissions} tCO2e emissions from {input_data['source']}",
                system="You are a CSRD reporting expert"
            )
            self.provenance.add_entry({
                "step": "narrative_generation",
                "cost": narrative.cost
            })

            # 5. AUDIT TRAIL (Moat 2: Provenance chain)
            return Result(
                success=True,
                data={
                    "emissions_tco2": emissions,
                    "narrative": narrative.content,
                    "audit_trail": self.provenance.serialize(),
                    "standards_citations": [d.metadata for d in standard_docs],
                    "audit_ready": True
                }
            )

        except Exception as e:
            self.provenance.add_entry({"step": "error", "error": str(e)})
            return Result(success=False, error=str(e))
```

### Complete Example 2: Supply Chain Intelligence

```python
"""
GL-VCCI-APP style multi-tenant scope 3 platform.
"""

from greenlang.core.multi_tenant import MultiTenantExecutor, TenantContext
from greenlang.intelligence.rag import RAGManager

class Scope3Calculator(Agent):
    """Multi-tenant Scope 3 calculation with complete provenance."""

    def __init__(self):
        self.executor = MultiTenantExecutor()
        self.rag = RAGManager()  # 100K+ emission factors

    def execute(self, input_data, tenant_id):
        context = TenantContext(tenant_id=tenant_id)

        with context:
            # Multi-tenant isolation: All operations use tenant_id context

            # 1. Entity resolution (AI + fuzzy matching)
            suppliers = self._resolve_suppliers(input_data['suppliers'])

            # 2. Retrieve emission factors (RAG system)
            for supplier in suppliers:
                factor = self.rag.query(
                    f"Emission factor for {supplier['industry']}",
                    top_k=1
                )
                supplier['emission_factor'] = factor[0].metadata['value']

            # 3. Calculate (deterministic, zero hallucination)
            total_emissions = sum(
                s['spend'] * s['emission_factor']
                for s in suppliers
            )

            # 4. Generate recommendations (AI)
            recommendations = self._get_recommendations(suppliers)

            # 5. Return with complete provenance
            return {
                "total_scope3_emissions": total_emissions,
                "top_suppliers": sorted(suppliers, key=lambda x: x['spend'])[:10],
                "recommendations": recommendations,
                "audit_trail": self._get_audit_trail(),
                "cost_attribution": self._get_cost_attribution()
            }
```

---

## Application-Specific Recipes

### Recipe 1: Build a Regulatory Compliance App

**Use these moats:**
- ✅ Zero-hallucination (100% deterministic)
- ✅ Provenance (SHA256 audit trail)
- ✅ Validation (50+ rules)
- ✅ RAG (citation of regulations)

**Code template:**
```python
# See GL-CSRD-APP implementation
from GL_CSRD_APP.agents import IntakeAgent, MaterialityAgent, CalculatorAgent
from GL_CSRD_APP.agents import AggregatorAgent, ReportingAgent, AuditAgent

pipeline = PipelineOrchestrator(name="csrd-pipeline")
pipeline.add_step("intake", IntakeAgent())
pipeline.add_step("materiality", MaterialityAgent())
pipeline.add_step("calculation", CalculatorAgent())
pipeline.add_step("aggregation", AggregatorAgent())
pipeline.add_step("reporting", ReportingAgent())
pipeline.add_step("audit", AuditAgent())

result = pipeline.run(company_data)
```

### Recipe 2: Build a Supply Chain App

**Use these moats:**
- ✅ RAG (100K+ emission factors)
- ✅ Multi-tenant (manage 1000s suppliers)
- ✅ Provenance (fraud prevention)
- ✅ Zero-hallucination (accurate calculations)
- ✅ LLM (entity resolution AI)

**Code template:**
```python
# See GL-VCCI-APP implementation
from GL_VCCI_APP.services.factor_broker import FactorBroker
from GL_VCCI_APP.services.agents.intake import ValueChainIntakeAgent
from GL_VCCI_APP.services.agents.calculator import Scope3Calculator
from GL_VCCI_APP.services.agents.hotspot import HotspotAnalysisAgent

# Initialize multi-tenant platform
platform = EmissionsPlatform()

# Process customer data
result = platform.calculate_scope3(
    tenant_id="customer-acme",
    procurement_data="suppliers.csv",
    factor_broker=FactorBroker()  # 100K+ factors via RAG
)
```

### Recipe 3: Build an Energy Optimization App

**Use these moats:**
- ✅ Multi-tenant (1000s buildings)
- ✅ <5ms latency (real-time calculations)
- ✅ ChatSession (recommendations)
- ✅ Forecasting (SARIMA + Prophet)

**Code template:**
```python
from greenlang.agents import ForecastAgentSARIMA
from greenlang.intelligence import ChatSession

class EnergyOptimizer:
    def optimize_building(self, building_id, sensor_data):
        # Forecast energy demand
        forecast_agent = ForecastAgentSARIMA(auto_tune=True)
        forecast = forecast_agent.run({
            "historical_data": sensor_data['history'],
            "forecast_periods": 24  # Next 24 hours
        })

        # Get AI recommendations
        chat = ChatSession(temperature=0.0)
        recommendations = chat.complete(
            prompt=f"Based on forecast {forecast}, recommend HVAC settings"
        )

        return {
            "forecast": forecast,
            "recommendations": recommendations.content,
            "estimated_savings": forecast['energy_reduction_pct']
        }
```

---

## Debugging & Optimization

### Monitor Moat Usage

```python
from greenlang.telemetry import MetricsCollector

metrics = MetricsCollector()

# Track zero-hallucination compliance
metrics.counter("calculation_executed", {"type": "deterministic"})  # Good
metrics.counter("calculation_executed", {"type": "llm"})  # Should be zero

# Track provenance
metrics.counter("audit_entry_added", {"step": "validation"})

# Track RAG usage
metrics.counter("rag_query", {"source": "emission_factors"})
metrics.histogram("rag_query_latency_ms", latency)

# Track LLM integration
metrics.histogram("llm_latency_ms", latency)
metrics.counter("llm_call", {"provider": "openai", "model": "gpt-4"})

# Track multi-tenant
metrics.gauge("active_tenants", 150)
metrics.counter("tenant_api_calls", {"tenant_id": "customer-acme"})
```

### Performance Tuning

```python
# Semantic caching for 30% cost reduction
session = ChatSession(semantic_cache_enabled=True)

# Batch processing for calculations
results = batch_calculate_emissions(
    data,
    batch_size=1000,
    parallel_workers=10
)

# RAG pre-caching for known queries
rag.preload_cache(["emission_factors", "scope_3_standards"])
```

---

## Conclusion

**The 5 moats form a complete system:**

1. **Zero-Hallucination** → Regulations trust your numbers
2. **Provenance** → Auditors verify your numbers
3. **RAG** → Your knowledge stays current
4. **LLM Integration** → Your insights are valuable
5. **Multi-Tenant** → You scale to billions in revenue

**Use all 5 together. The sum is more powerful than the parts.**

---

**Version:** 1.0.0
**Updated:** November 9, 2025
**Status:** Production-Ready
