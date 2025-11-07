# Anomaly Investigation Agent - Architecture Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Anomaly Investigation Agent                         │
│                     (InsightAgent Pattern)                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │          DETERMINISTIC LAYER (calculate method)             │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │                                                              │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │     Isolation Forest Anomaly Agent                   │  │   │
│  │  │     (anomaly_agent_iforest.py)                       │  │   │
│  │  ├──────────────────────────────────────────────────────┤  │   │
│  │  │  • Fit Isolation Forest model                        │  │   │
│  │  │  • Detect anomalies                                  │  │   │
│  │  │  • Calculate anomaly scores                          │  │   │
│  │  │  • Rank anomalies by severity                        │  │   │
│  │  │  • Analyze patterns                                  │  │   │
│  │  │  • Generate alerts                                   │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                           ↓                                 │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │              Detection Results                       │  │   │
│  │  ├──────────────────────────────────────────────────────┤  │   │
│  │  │  • Anomalies: [bool]                                 │  │   │
│  │  │  • Scores: [float]                                   │  │   │
│  │  │  • Severity distribution: {str: int}                 │  │   │
│  │  │  • Top anomalies: [dict]                             │  │   │
│  │  │  • Patterns: dict                                    │  │   │
│  │  │  • Audit trail: [str]                                │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  │  Characteristics:                                            │   │
│  │  ✓ 100% reproducible                                        │   │
│  │  ✓ No AI/LLM usage                                          │   │
│  │  ✓ Full audit trail                                         │   │
│  │  ✓ Temperature: N/A (deterministic)                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│                                ↓                                      │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │           AI-POWERED LAYER (explain method)                 │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │                                                              │   │
│  │  Step 1: RAG Retrieval                                      │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │           RAG Engine (4 Collections)                 │  │   │
│  │  ├──────────────────────────────────────────────────────┤  │   │
│  │  │  1. anomaly_patterns (historical cases)              │  │   │
│  │  │  2. root_cause_database (failure modes)              │  │   │
│  │  │  3. sensor_specifications (sensor behavior)          │  │   │
│  │  │  4. maintenance_procedures (correlation)             │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                           ↓                                 │   │
│  │                                                              │   │
│  │  Step 2: Tool-Based Evidence Gathering                      │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │              Investigation Tools                     │  │   │
│  │  ├──────────────────────────────────────────────────────┤  │   │
│  │  │  Tool 1: maintenance_log_tool                        │  │   │
│  │  │    • Query maintenance events                        │  │   │
│  │  │    • Correlate with anomaly times                    │  │   │
│  │  │    • Return: events, correlation, confidence         │  │   │
│  │  │                                                       │  │   │
│  │  │  Tool 2: sensor_diagnostic_tool                      │  │   │
│  │  │    • Check sensor health/calibration                 │  │   │
│  │  │    • Detect drift patterns                           │  │   │
│  │  │    • Return: status, recommendations                 │  │   │
│  │  │                                                       │  │   │
│  │  │  Tool 3: weather_data_tool                           │  │   │
│  │  │    • Fetch weather conditions                        │  │   │
│  │  │    • Identify extreme events                         │  │   │
│  │  │    • Return: conditions, correlation                 │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                           ↓                                 │   │
│  │                                                              │   │
│  │  Step 3: AI Analysis                                        │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │              ChatSession + LLM                       │  │   │
│  │  ├──────────────────────────────────────────────────────┤  │   │
│  │  │  • Temperature: 0.6 (analytical consistency)         │  │   │
│  │  │  • System prompt: Investigation expert               │  │   │
│  │  │  • User prompt: Detection results + RAG + tools      │  │   │
│  │  │  • Process: Root cause analysis                      │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                           ↓                                 │   │
│  │                                                              │   │
│  │  Step 4: Report Generation                                  │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │         Investigation Report (Markdown)              │  │   │
│  │  ├──────────────────────────────────────────────────────┤  │   │
│  │  │  • Executive summary                                 │  │   │
│  │  │  • Root cause analysis                               │  │   │
│  │  │  • Evidence from tools                               │  │   │
│  │  │  • Pattern insights                                  │  │   │
│  │  │  • Confidence-scored recommendations                 │  │   │
│  │  │  • Audit trail reference                             │  │   │
│  │  │  • Investigation metadata                            │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  │  Characteristics:                                            │   │
│  │  ✓ Evidence-based analysis                                  │   │
│  │  ✓ Multi-source investigation                               │   │
│  │  ✓ Confidence scoring                                       │   │
│  │  ✓ Temperature: 0.6                                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Input Data (DataFrame)
         ↓
┌─────────────────────┐
│  calculate()        │  ← Deterministic Layer
├─────────────────────┤
│  • Validate inputs  │
│  • Call detector    │
│  • Extract metrics  │
│  • Audit trail      │
└─────────────────────┘
         ↓
Detection Result (Dict)
         ↓
┌─────────────────────┐
│  explain()          │  ← AI-Powered Layer
├─────────────────────┤
│  1. Build RAG query │
│  2. Retrieve from 4 │
│     collections     │
│  3. Call 3 tools    │
│     for evidence    │
│  4. AI analysis     │
│  5. Format report   │
└─────────────────────┘
         ↓
Investigation Report (Markdown)
```

## Component Interactions

```
┌─────────────────────────────────────────────────────────────┐
│                    External Dependencies                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │  ChatSession   │  │   RAGEngine    │  │  Detector    │  │
│  │  (LLM API)     │  │  (Vector DB)   │  │  (IForest)   │  │
│  └────────┬───────┘  └────────┬───────┘  └──────┬───────┘  │
│           │                    │                  │           │
│           │                    │                  │           │
└───────────┼────────────────────┼──────────────────┼───────────┘
            │                    │                  │
            ↓                    ↓                  ↓
┌─────────────────────────────────────────────────────────────┐
│         Anomaly Investigation Agent (InsightAgent)          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  __init__()                                         │   │
│  │  • Initialize detector                              │   │
│  │  • Set budgets                                      │   │
│  │  • Setup tracking                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  calculate(inputs) → Dict                           │   │
│  │  • Validate inputs                                  │   │
│  │  • detector.process(inputs)                         │   │
│  │  • Extract results                                  │   │
│  │  • Capture audit trail                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  explain(result, context, session, rag) → str       │   │
│  │  • Build RAG query                                  │   │
│  │  • rag_engine.query(collections=[...])              │   │
│  │  • session.chat(tools=[...])                        │   │
│  │  • Process tool calls                               │   │
│  │  • Format report                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tool Implementations (Private Methods)             │   │
│  │  • _query_maintenance_logs()                        │   │
│  │  • _run_sensor_diagnostics()                        │   │
│  │  • _fetch_weather_data()                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Helper Methods                                     │   │
│  │  • _build_rag_query()                               │   │
│  │  • _build_investigation_prompt()                    │   │
│  │  • _get_system_prompt()                             │   │
│  │  • _get_investigation_tools()                       │   │
│  │  • _process_tool_calls()                            │   │
│  │  • _format_investigation_report()                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Pattern Compliance Matrix

```
┌──────────────────────────────────────────────────────────────┐
│              InsightAgent Pattern Requirements               │
├────────────────────┬─────────────────────────────────────────┤
│ Requirement        │ Implementation                          │
├────────────────────┼─────────────────────────────────────────┤
│ Base Class         │ class AnomalyInvestigationAgent(        │
│                    │     InsightAgent)                       │
├────────────────────┼─────────────────────────────────────────┤
│ Category           │ category = AgentCategory.INSIGHT        │
├────────────────────┼─────────────────────────────────────────┤
│ Metadata           │ metadata = AgentMetadata(...)           │
├────────────────────┼─────────────────────────────────────────┤
│ calculate()        │ def calculate(inputs) -> Dict:          │
│                    │   # Deterministic Isolation Forest     │
│                    │   return detection_results              │
├────────────────────┼─────────────────────────────────────────┤
│ explain()          │ async def explain(result, ...) -> str:  │
│                    │   # AI investigation with RAG + tools  │
│                    │   return investigation_report           │
├────────────────────┼─────────────────────────────────────────┤
│ Deterministic      │ ✓ Uses IsolationForest (seeded)        │
│ Calculations       │ ✓ Same inputs → same outputs            │
├────────────────────┼─────────────────────────────────────────┤
│ AI Enhancement     │ ✓ ChatSession for reasoning             │
│                    │ ✓ RAG for knowledge                     │
│                    │ ✓ Tools for evidence                    │
├────────────────────┼─────────────────────────────────────────┤
│ Temperature        │ temperature=0.6 (analytical)            │
├────────────────────┼─────────────────────────────────────────┤
│ Audit Trail        │ ✓ _capture_calculation_audit()          │
│                    │ ✓ Full calculation trace                │
├────────────────────┼─────────────────────────────────────────┤
│ Documentation      │ ✓ Comprehensive docstrings              │
│                    │ ✓ Type hints                            │
│                    │ ✓ Examples                              │
└────────────────────┴─────────────────────────────────────────┘
```

## Tool Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Investigation Tools                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Tool 1: maintenance_log_tool                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Purpose: Maintenance correlation                     │  │
│  │  ───────────────────────────────────────────────────  │  │
│  │  Input:                                               │  │
│  │    • start_time, end_time                             │  │
│  │    • system_id                                        │  │
│  │    • event_types                                      │  │
│  │  ───────────────────────────────────────────────────  │  │
│  │  Process:                                             │  │
│  │    1. Query maintenance database                      │  │
│  │    2. Filter by time window                           │  │
│  │    3. Correlate with anomaly times                    │  │
│  │    4. Calculate correlation strength                  │  │
│  │  ───────────────────────────────────────────────────  │  │
│  │  Output:                                              │  │
│  │    • events: List[Dict]                               │  │
│  │    • correlation: str                                 │  │
│  │    • confidence: float                                │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  Tool 2: sensor_diagnostic_tool                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Purpose: Sensor health analysis                      │  │
│  │  ───────────────────────────────────────────────────  │  │
│  │  Input:                                               │  │
│  │    • sensor_ids                                       │  │
│  │    • diagnostic_type                                  │  │
│  │    • time_window                                      │  │
│  │  ───────────────────────────────────────────────────  │  │
│  │  Process:                                             │  │
│  │    1. Check calibration dates                         │  │
│  │    2. Analyze drift patterns                          │  │
│  │    3. Detect failures                                 │  │
│  │    4. Generate recommendations                        │  │
│  │  ───────────────────────────────────────────────────  │  │
│  │  Output:                                              │  │
│  │    • diagnostics: List[Dict]                          │  │
│  │    • overall_health: str                              │  │
│  │    • confidence: float                                │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  Tool 3: weather_data_tool                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Purpose: Environmental correlation                   │  │
│  │  ───────────────────────────────────────────────────  │  │
│  │  Input:                                               │  │
│  │    • location                                         │  │
│  │    • start_time, end_time                             │  │
│  │    • metrics                                          │  │
│  │  ───────────────────────────────────────────────────  │  │
│  │  Process:                                             │  │
│  │    1. Query weather API                               │  │
│  │    2. Retrieve conditions                             │  │
│  │    3. Identify extreme events                         │  │
│  │    4. Correlate with anomalies                        │  │
│  │  ───────────────────────────────────────────────────  │  │
│  │  Output:                                              │  │
│  │    • conditions: List[Dict]                           │  │
│  │    • extreme_events: List[str]                        │  │
│  │    • correlation: str                                 │  │
│  │    • confidence: float                                │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## RAG Collection Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Collections (4)                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Collection 1: anomaly_patterns                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  • Historical anomaly cases                           │  │
│  │  • Resolution strategies                              │  │
│  │  • Success rates                                      │  │
│  │  • Lessons learned                                    │  │
│  │  • Similar case studies                               │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  Collection 2: root_cause_database                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  • Equipment failure signatures                       │  │
│  │  • Common root causes by system                       │  │
│  │  • Diagnostic decision trees                          │  │
│  │  • Failure probabilities                              │  │
│  │  • Symptom patterns                                   │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  Collection 3: sensor_specifications                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  • Sensor specifications                              │  │
│  │  • Calibration schedules                              │  │
│  │  • Known failure modes                                │  │
│  │  • Drift patterns                                     │  │
│  │  • Maintenance requirements                           │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  Collection 4: maintenance_procedures                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  • Maintenance schedules                              │  │
│  │  • Preventive procedures                              │  │
│  │  • Maintenance impact patterns                        │  │
│  │  • Post-maintenance signatures                        │  │
│  │  • Correlation analysis                               │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Performance Tracking

```
┌─────────────────────────────────────────────────────────────┐
│                    Metrics & Monitoring                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Agent Metrics:                                              │
│  • total_investigations: int                                 │
│  • total_cost_usd: float                                     │
│  • avg_cost_per_investigation: float                         │
│                                                               │
│  Detector Metrics:                                           │
│  • ai_call_count: int                                        │
│  • tool_call_count: int                                      │
│  • total_cost_usd: float                                     │
│                                                               │
│  Per-Investigation Metrics:                                  │
│  • detection_time_s: float                                   │
│  • investigation_time_s: float                               │
│  • tokens_used: int                                          │
│  • tools_called: int                                         │
│  • rag_queries: int                                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
greenlang/
└── agents/
    ├── base_agents.py                           # Base classes
    ├── categories.py                            # Agent categories
    ├── anomaly_agent_iforest.py                # Detection engine (43KB)
    ├── anomaly_investigation_agent.py          # Investigation agent (40KB)
    ├── ANOMALY_INVESTIGATION_AGENT_README.md   # Usage guide
    └── ...

Documentation/
├── ANOMALY_INVESTIGATION_AGENT_DELIVERY.md     # Delivery summary
└── ANOMALY_INVESTIGATION_ARCHITECTURE.md       # This file
```

## Summary

The Anomaly Investigation Agent implements a clean hybrid architecture:

1. **Deterministic Layer** - Reliable anomaly detection via Isolation Forest
2. **AI Layer** - Evidence-based investigation with RAG + tools
3. **Clear Separation** - Calculations vs. insights
4. **Full Traceability** - Audit trails and confidence scores
5. **Production Ready** - Error handling, monitoring, documentation

**Key Innovation:** Combines the reliability of deterministic ML detection with the analytical power of AI investigation, all within a compliant, auditable framework.
