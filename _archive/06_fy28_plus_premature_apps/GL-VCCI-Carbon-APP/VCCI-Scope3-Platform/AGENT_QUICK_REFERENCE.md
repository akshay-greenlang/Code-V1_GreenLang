# Agent Quick Reference Guide
**GreenLang-First Architecture - Agent Usage**

---

## All Agents Are Now SDK-Compliant

All 5 agents in the VCCI Scope 3 Platform now inherit from `greenlang.sdk.base.Agent` and support both legacy and modern interfaces.

---

## 1. ValueChainIntakeAgent

**Purpose:** Multi-format data ingestion for Scope 3 value chain data

**Import:**
```python
from services.agents.intake.agent import ValueChainIntakeAgent
```

**Legacy Usage (Still Works):**
```python
agent = ValueChainIntakeAgent(tenant_id="acme-corp")

# Ingest file
result = agent.ingest_file(
    file_path=Path("suppliers.csv"),
    format="csv",
    entity_type="supplier"
)

# Process batch
result = agent.process_batch(records, batch_id="BATCH-001")
```

**Modern SDK Usage:**
```python
agent = ValueChainIntakeAgent(tenant_id="acme-corp")

# Use standardized process() method
result = agent.process(ingestion_records)  # Returns IngestionResult

# Use run() with automatic error handling
sdk_result = agent.run(ingestion_records)  # Returns Result[IngestionResult]
if sdk_result.success:
    print(f"Processed {sdk_result.data.statistics.total_records} records")
else:
    print(f"Error: {sdk_result.error}")
```

**Metadata:**
```python
agent.metadata.id         # "intake_agent_{tenant_id}"
agent.metadata.version    # "2.0.0"
agent.metadata.tags       # ["scope3", "ingestion", "entity-resolution"]
```

---

## 2. Scope3CalculatorAgent

**Purpose:** Scope 3 emissions calculator for all 15 categories

**Import:**
```python
from services.agents.calculator.agent import Scope3CalculatorAgent
```

**Legacy Usage (Still Works):**
```python
agent = Scope3CalculatorAgent(factor_broker=broker, industry_mapper=mapper)

# Calculate by category
result = await agent.calculate_category_1(data)
result = await agent.calculate_category_4(data)

# Calculate batch
batch_result = await agent.calculate_batch(records, category=1)
```

**Modern SDK Usage:**
```python
agent = Scope3CalculatorAgent(factor_broker=broker)

# Use standardized process() method
input_data = {
    "category": 1,
    "data": category1_data
}
result = await agent.process(input_data)  # Returns CalculationResult

# Use run() with automatic error handling
sdk_result = await agent.run(input_data)  # Returns Result[CalculationResult]
if sdk_result.success:
    print(f"Emissions: {sdk_result.data.emissions_kgco2e} kgCO2e")
```

**Metadata:**
```python
agent.metadata.id         # "scope3_calculator_agent"
agent.metadata.version    # "2.0.0"
agent.metadata.tags       # ["scope3", "emissions", "calculator", "ghg-protocol"]
```

**Telemetry:**
```python
# All calculations are automatically tracked
agent.metrics.get_metrics()  # View performance stats
# Metrics include: emissions.category_1, emissions.category_4, etc.
```

---

## 3. HotspotAnalysisAgent

**Purpose:** Emissions hotspot analysis and scenario modeling

**Import:**
```python
from services.agents.hotspot.agent import HotspotAnalysisAgent
```

**Legacy Usage (Still Works):**
```python
agent = HotspotAnalysisAgent(config=config)

# Pareto analysis
pareto = agent.analyze_pareto(emissions_data, dimension="supplier_name")

# Hotspot detection
hotspots = agent.identify_hotspots(emissions_data)

# Comprehensive analysis
results = agent.analyze_comprehensive(emissions_data)
```

**Modern SDK Usage:**
```python
agent = HotspotAnalysisAgent()

# Use standardized process() method
results = agent.process(emissions_data)  # Returns comprehensive analysis dict

# Use run() with automatic error handling
sdk_result = agent.run(emissions_data)  # Returns Result[Dict]
if sdk_result.success:
    print(f"Found {sdk_result.data['summary']['n_hotspots']} hotspots")
```

**Metadata:**
```python
agent.metadata.id         # "hotspot_analysis_agent"
agent.metadata.version    # "2.0.0"
agent.metadata.tags       # ["hotspot", "analysis", "pareto", "abatement"]
```

**Telemetry:**
```python
# Hotspot counts tracked automatically
agent.metrics.get_metrics()
# Metrics include: hotspots.total
```

---

## 4. Scope3ReportingAgent

**Purpose:** Multi-standard sustainability reporting (ESRS, CDP, IFRS, ISO 14083)

**Import:**
```python
from services.agents.reporting.agent import Scope3ReportingAgent
```

**Legacy Usage (Still Works):**
```python
agent = Scope3ReportingAgent(config=config)

# Generate ESRS E1 report
result = agent.generate_esrs_e1_report(
    emissions_data=emissions,
    company_info=company,
    export_format="pdf"
)

# Generate CDP questionnaire
result = agent.generate_cdp_report(emissions_data, company_info)

# Generate IFRS S2 report
result = agent.generate_ifrs_s2_report(emissions_data, company_info)

# Generate ISO 14083 certificate
result = agent.generate_iso_14083_certificate(transport_data)
```

**Modern SDK Usage:**
```python
agent = Scope3ReportingAgent()

# Use standardized process() method
input_data = {
    "standard": "ESRS_E1",
    "emissions_data": emissions,
    "company_info": company,
    "export_format": "pdf",
    "output_path": "report.pdf"
}
result = agent.process(input_data)  # Returns ReportResult

# Use run() with automatic error handling
sdk_result = agent.run(input_data)  # Returns Result[ReportResult]
if sdk_result.success:
    print(f"Report saved to: {sdk_result.data.file_path}")
```

**Supported Standards:**
- `ESRS_E1` - EU CSRD reporting
- `CDP` - CDP Climate Change questionnaire
- `IFRS_S2` - IFRS S2 climate disclosures
- `ISO_14083` - ISO 14083 transport conformance

**Metadata:**
```python
agent.metadata.id         # "scope3_reporting_agent"
agent.metadata.version    # "2.0.0"
agent.metadata.tags       # ["reporting", "esrs", "cdp", "ifrs", "iso14083"]
```

**Telemetry:**
```python
# Report generation tracked by standard
agent.metrics.get_metrics()
# Metrics include: reports.ESRS_E1, reports.CDP, reports.IFRS_S2, reports.ISO_14083
```

---

## 5. SupplierEngagementAgent

**Purpose:** Consent-aware supplier engagement and data collection

**Import:**
```python
from services.agents.engagement.agent import SupplierEngagementAgent
```

**Legacy Usage (Still Works):**
```python
agent = SupplierEngagementAgent(config=config)

# Create campaign
campaign = agent.create_campaign(
    name="Q1 Data Collection",
    target_suppliers=["SUP-001", "SUP-002"]
)

# Send email
result = agent.send_email(
    supplier_id="SUP-001",
    template=template,
    personalization_data=data
)

# Get analytics
analytics = agent.get_campaign_analytics(campaign_id)
```

**Modern SDK Usage:**
```python
agent = SupplierEngagementAgent()

# Use standardized process() method for operations
input_data = {
    "operation": "create_campaign",
    "params": {
        "name": "Q1 Data Collection",
        "target_suppliers": ["SUP-001", "SUP-002"]
    }
}
result = agent.process(input_data)  # Returns operation result dict

# Use run() with automatic error handling
sdk_result = agent.run(input_data)  # Returns Result[Dict]
if sdk_result.success:
    print(f"Campaign created: {sdk_result.data['campaign_id']}")
```

**Supported Operations:**
- `create_campaign` - Create engagement campaign
- `send_email` - Send email to supplier
- `validate_upload` - Validate supplier data upload
- `get_analytics` - Get campaign analytics

**Metadata:**
```python
agent.metadata.id         # "supplier_engagement_agent"
agent.metadata.version    # "2.0.0"
agent.metadata.tags       # ["engagement", "supplier", "consent", "gdpr", "campaigns"]
```

**Telemetry:**
```python
# Operations tracked automatically
agent.metrics.get_metrics()
# Metrics include: engagement.create_campaign, engagement.send_email, etc.
```

---

## Common Patterns

### 1. Error Handling with run()

All agents support the `run()` method which provides automatic error handling:

```python
# Instead of try/catch
try:
    result = await agent.process(data)
except Exception as e:
    # Handle error
    ...

# Use run() for cleaner code
sdk_result = await agent.run(data)
if sdk_result.success:
    # Success path
    result = sdk_result.data
else:
    # Error path
    print(f"Error: {sdk_result.error}")
```

### 2. Input Validation

All agents implement `validate()`:

```python
agent = Scope3CalculatorAgent(...)

input_data = {"category": 1, "data": {...}}

# Validate before processing
if agent.validate(input_data):
    result = await agent.process(input_data)
else:
    print("Invalid input")
```

### 3. Accessing Metadata

All agents have rich metadata:

```python
# Get agent info
print(agent.metadata.id)
print(agent.metadata.name)
print(agent.metadata.version)
print(agent.metadata.description)
print(agent.metadata.tags)

# Get full metadata dict
metadata_dict = agent.metadata.to_dict()
```

### 4. Telemetry and Metrics

All agents automatically collect telemetry:

```python
# Metrics are collected automatically during execution
result = await agent.process(data)

# Retrieve metrics
metrics = agent.metrics.get_metrics()
print(metrics)

# Example output:
# {
#   "emissions.category_1": 1234.56,
#   "calculator_process.duration_ms": 150.2,
#   "calculator_process.count": 1
# }
```

### 5. Caching

All agents have caching infrastructure:

```python
# Cache is integrated but not enabled by default
agent.cache_manager  # Available if configured

# To enable caching (future):
# agent.cache_manager.enable()
```

---

## Pipeline Composition (Future)

Now that all agents are SDK-compliant, they can be composed into pipelines:

```python
from greenlang.sdk.pipeline import SequentialPipeline

# Create pipeline: Intake -> Calculator -> Hotspot -> Reporting
pipeline = SequentialPipeline()
pipeline.add_agent(intake_agent)
pipeline.add_agent(calc_agent)
pipeline.add_agent(hotspot_agent)
pipeline.add_agent(report_agent)

# Execute entire pipeline
result = await pipeline.execute(raw_data)
```

---

## Migration Checklist

If you're migrating existing code:

- [ ] **No changes required** - All existing method calls still work
- [ ] Consider using `run()` for better error handling
- [ ] Consider using `process()` for standardized interface
- [ ] Access `agent.metadata` for agent information
- [ ] Access `agent.metrics` for performance insights
- [ ] Enable caching if needed for performance

---

## Performance Tips

1. **Use async/await consistently**
   ```python
   # Good
   result = await agent.process(data)

   # Avoid blocking
   result = agent.process(data)  # May cause issues
   ```

2. **Batch operations when possible**
   ```python
   # Good - batch processing
   results = await calc_agent.calculate_batch(records, category=1)

   # Avoid - individual calls
   for record in records:
       result = await calc_agent.calculate_category_1(record)  # Slow!
   ```

3. **Monitor metrics**
   ```python
   # Check performance periodically
   stats = agent.get_performance_stats()  # Calculator agent
   metrics = agent.metrics.get_metrics()  # All agents
   ```

---

## Support

For questions or issues:
- **Documentation:** See `AGENT_COMPLIANCE_REPORT.md`
- **Examples:** See `tests/agents/` directory
- **Architecture:** See `GREENLANG_FIRST_ARCHITECTURE_POLICY.md`

---

**Last Updated:** 2025-11-09
**Version:** All agents v2.0.0
**Status:** Production Ready
