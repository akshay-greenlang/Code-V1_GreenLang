# GreenLang Framework - Architecture Diagrams

**Version:** 1.0.0
**Date:** 2025-10-17
**Status:** Production

---

## 📐 OVERVIEW

This document provides comprehensive architectural diagrams for the GreenLang Framework, showing class hierarchies, data flow, and system interactions.

---

## 🏗️ 1. FRAMEWORK ARCHITECTURE - HIGH LEVEL

```
┌─────────────────────────────────────────────────────────────────────┐
│                      GreenLang Framework                             │
│                      (Agent Development Platform)                    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
         ┌──────────▼────────┐     │     ┌───────▼────────┐
         │  Core Framework   │     │     │  Utilities     │
         │   (agents/)       │     │     │  & Support     │
         └──────────┬────────┘     │     └───────┬────────┘
                    │              │              │
    ┌───────────────┼──────────┐   │   ┌─────────┼──────────────┐
    │               │          │   │   │         │              │
┌───▼───┐   ┌──────▼─────┐ ┌──▼───▼───▼──┐  ┌──▼─────┐  ┌────▼─────┐
│ Base  │   │ Specialized│ │ Provenance  │  │Validation│ │   I/O    │
│ Agent │   │   Agents   │ │  Framework  │  │Framework │ │ Utilities│
└───────┘   └────────────┘ └─────────────┘  └──────────┘ └──────────┘
    │             │                │              │            │
    │             │                │              │            │
┌───▼─────────────▼────────────────▼──────────────▼────────────▼─────┐
│                    Developer's Custom Agents                        │
│              (BaseDataProcessor, BaseCalculator, etc.)              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 2. BASE AGENT CLASS HIERARCHY

```
                        ┌─────────────┐
                        │   Object    │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │  BaseAgent  │◄─────────────┐
                        │             │              │
                        │ Properties: │         Inherits from
                        │ - config    │              │
                        │ - stats     │              │
                        │ - logger    │              │
                        │             │              │
                        │ Methods:    │              │
                        │ - run()     │              │
                        │ - execute() │              │
                        │ - validate()│              │
                        └──────┬──────┘              │
                               │                     │
                ┌──────────────┼──────────────┐      │
                │              │              │      │
         ┌──────▼─────┐  ┌────▼────┐  ┌─────▼──────┐│
         │    Base    │  │  Base   │  │    Base    ││
         │    Data    │  │Calculator│  │  Reporter  ││
         │ Processor  │  │         │  │            ││
         └──────┬─────┘  └────┬────┘  └─────┬──────┘│
                │             │              │       │
                │             │              │       │
         ┌──────▼─────────────▼──────────────▼───────┘
         │
         │ Your Custom Agents:
         │ - CSVProcessor (extends BaseDataProcessor)
         │ - EmissionsCalculator (extends BaseCalculator)
         │ - MonthlyReporter (extends BaseReporter)
         └────────────────────────────────────────────────
```

### Class Responsibilities

```
┌─────────────────────────────────────────────────────────┐
│ BaseAgent (Abstract)                                    │
│─────────────────────────────────────────────────────────│
│ ✓ Lifecycle management (init, validate, execute)       │
│ ✓ Metrics collection and tracking                      │
│ ✓ Logging with context                                 │
│ ✓ Resource loading with caching                        │
│ ✓ Pre/post execution hooks                             │
│ ✓ Statistics tracking                                  │
└─────────────────────────────────────────────────────────┘
                        │
                        │ extends
                        ▼
┌─────────────────────────────────────────────────────────┐
│ BaseDataProcessor                                       │
│─────────────────────────────────────────────────────────│
│ ✓ Batch processing with configurable size              │
│ ✓ Parallel execution (ThreadPoolExecutor)              │
│ ✓ Progress tracking with tqdm                          │
│ ✓ Error collection and recovery                        │
│ ✓ Record-level validation                              │
│ ✓ Abstract: process_record()                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ BaseCalculator                                          │
│─────────────────────────────────────────────────────────│
│ ✓ High-precision Decimal arithmetic                    │
│ ✓ Calculation caching with LRU eviction                │
│ ✓ Deterministic execution guarantees                   │
│ ✓ Calculation step tracing                             │
│ ✓ Unit conversion utilities                            │
│ ✓ Safe division (zero handling)                        │
│ ✓ Abstract: calculate()                                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ BaseReporter                                            │
│─────────────────────────────────────────────────────────│
│ ✓ Multi-format output (MD, HTML, JSON, Excel)          │
│ ✓ Data aggregation utilities                           │
│ ✓ Section management                                   │
│ ✓ Template-based reporting                             │
│ ✓ Summary generation                                   │
│ ✓ Abstract: aggregate_data(), build_sections()         │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 3. DATA FLOW - AGENT EXECUTION

```
                    User Code
                        │
                        │ calls run()
                        ▼
            ┌───────────────────────┐
            │   Agent.run()         │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  1. Validate Config   │
            │     (enabled?)        │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  2. Run Pre-Hooks     │
            │     (decorators)      │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  3. Validate Input    │
            │     (validate_input)  │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  4. Preprocess        │
            │     (preprocess)      │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  5. EXECUTE           │
            │     (execute)         │◄─── Your business logic here!
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  6. Postprocess       │
            │     (postprocess)     │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  7. Run Post-Hooks    │
            │     (metrics, etc)    │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  8. Collect Metrics   │
            │     (if enabled)      │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  9. Update Stats      │
            │     (success/failure) │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │  10. Return Result    │
            │      (AgentResult)    │
            └───────────────────────┘
```

---

## 🔄 4. BATCH PROCESSING FLOW (BaseDataProcessor)

```
Input: {"records": [...]}
        │
        ▼
┌────────────────────┐
│ 1. Extract Records │
│    from input_data │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ 2. Create Batches  │
│    (batch_size)    │
└─────────┬──────────┘
          │
          ▼
┌─────────────────────────────────────┐
│ 3. Process Each Batch               │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ For each record in batch:   │   │
│  │                             │   │
│  │  ┌──────────────────────┐  │   │
│  │  │ 3a. Validate Record  │  │   │
│  │  └──────────┬───────────┘  │   │
│  │             │               │   │
│  │             ▼               │   │
│  │  ┌──────────────────────┐  │   │
│  │  │ 3b. Process Record   │  │   │
│  │  │     (YOUR CODE)      │  │   │
│  │  └──────────┬───────────┘  │   │
│  │             │               │   │
│  │             ▼               │   │
│  │  ┌──────────────────────┐  │   │
│  │  │ 3c. Collect Result   │  │   │
│  │  │     or Error         │  │   │
│  │  └──────────────────────┘  │   │
│  │                             │   │
│  └─────────────────────────────┘   │
│                                     │
│  Parallel Mode:                    │
│  ┌─────────────────────────────┐   │
│  │ ThreadPoolExecutor          │   │
│  │ - Process batches in ||     │   │
│  │ - Join results              │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
          │
          ▼
┌────────────────────┐
│ 4. Aggregate       │
│    Results         │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ 5. Return          │
│    DataProcessor   │
│    Result          │
└────────────────────┘

Statistics Tracked:
- records_processed
- records_failed
- batches_processed
- execution_time_ms
- success_rate
```

---

## 🧮 5. CALCULATION CACHING FLOW (BaseCalculator)

```
calculate() called with inputs
        │
        ▼
┌────────────────────────┐
│ 1. Generate Cache Key  │
│    hash(inputs)        │
└─────────┬──────────────┘
          │
          ▼
    ┌─────────────┐
    │ Cache Hit?  │
    └─────┬───┬───┘
          │   │
      YES │   │ NO
          │   │
          ▼   ▼
    ┌─────────────┐     ┌──────────────────┐
    │ Return      │     │ 2. Execute       │
    │ Cached      │     │    calculate()   │
    │ Result      │     │    (YOUR CODE)   │
    └─────────────┘     └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │ 3. Round to      │
                        │    precision     │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │ 4. Store in      │
                        │    Cache         │
                        │    (LRU)         │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │ 5. Return Result │
                        └──────────────────┘

Cache Structure:
┌─────────────────────────────────────┐
│ LRU Cache (max_size=128)            │
│                                     │
│ key1 → {result, timestamp, steps}  │
│ key2 → {result, timestamp, steps}  │
│ ...                                 │
│                                     │
│ Eviction: Least Recently Used       │
└─────────────────────────────────────┘
```

---

## 🔒 6. PROVENANCE FRAMEWORK ARCHITECTURE

```
                    Agent Execution
                          │
                          ▼
            ┌─────────────────────────┐
            │  ProvenanceContext      │
            │  - Tracks execution     │
            │  - Collects metadata    │
            └────────────┬────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌────────────────┐ ┌────────────┐ ┌─────────────┐
│ Environment    │ │  Hashing   │ │  Records    │
│ - Python ver   │ │  - SHA256  │ │  - Inputs   │
│ - OS info      │ │  - Merkle  │ │  - Outputs  │
│ - Dependencies │ │  - Files   │ │  - Lineage  │
└────────┬───────┘ └──────┬─────┘ └──────┬──────┘
         │                │               │
         └────────────────┼───────────────┘
                          │
                          ▼
            ┌─────────────────────────┐
            │   ProvenanceRecord      │
            │   - Complete audit      │
            │   - JSON serializable   │
            └────────────┬────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │   Validation            │
            │   - Verify integrity    │
            │   - Check tampering     │
            └────────────┬────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │   Reporting             │
            │   - Markdown            │
            │   - HTML                │
            │   - JSON                │
            └─────────────────────────┘
```

### Provenance Record Structure

```
ProvenanceRecord
├── metadata
│   ├── agent_name: "EmissionsCalculator"
│   ├── version: "1.0.0"
│   ├── timestamp: "2025-10-17T10:00:00Z"
│   └── execution_id: "uuid-..."
├── environment
│   ├── python_version: "3.9.0"
│   ├── os: "Windows 10"
│   ├── dependencies: {...}
│   └── system_info: {...}
├── inputs
│   ├── data: {...}
│   └── input_hash: "sha256..."
├── outputs
│   ├── result: {...}
│   └── output_hash: "sha256..."
├── execution
│   ├── start_time: "..."
│   ├── end_time: "..."
│   ├── duration_ms: 123.45
│   └── status: "success"
└── lineage
    ├── parent_records: [...]
    └── child_records: [...]
```

---

## ✅ 7. VALIDATION FRAMEWORK ARCHITECTURE

```
                ValidationFramework
                        │
                        │ registers
                        ▼
        ┌───────────────────────────────┐
        │     Validator Registry        │
        │                               │
        │  ┌─────────────────────────┐  │
        │  │ Schema Validator        │  │
        │  │ (JSON Schema Draft 7)   │  │
        │  └─────────────────────────┘  │
        │                               │
        │  ┌─────────────────────────┐  │
        │  │ Rules Engine            │  │
        │  │ (12 operators)          │  │
        │  └─────────────────────────┘  │
        │                               │
        │  ┌─────────────────────────┐  │
        │  │ Quality Checks          │  │
        │  │ (completeness, etc)     │  │
        │  └─────────────────────────┘  │
        │                               │
        │  ┌─────────────────────────┐  │
        │  │ Custom Validators       │  │
        │  │ (user-defined)          │  │
        │  └─────────────────────────┘  │
        └───────────────┬───────────────┘
                        │
                        │ validate()
                        ▼
            ┌───────────────────────┐
            │   Input Data          │
            └───────────┬───────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
    Schema          Rules          Quality
    Check           Check          Check
        │               │               │
        └───────────────┼───────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  ValidationResult     │
            │                       │
            │  - valid: bool        │
            │  - errors: []         │
            │  - warnings: []       │
            │  - info: []           │
            └───────────────────────┘
```

### Validation Execution Flow

```
Input Data
    │
    ▼
┌────────────────────────┐
│ Pre-Validators (hooks) │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────────────────┐
│ For Each Registered Validator:    │
│                                    │
│  ┌──────────────────────────────┐ │
│  │ 1. Check if enabled          │ │
│  └────────────┬─────────────────┘ │
│               │                   │
│  ┌────────────▼─────────────────┐ │
│  │ 2. Execute validator         │ │
│  └────────────┬─────────────────┘ │
│               │                   │
│  ┌────────────▼─────────────────┐ │
│  │ 3. Collect errors/warnings   │ │
│  └────────────┬─────────────────┘ │
│               │                   │
│  ┌────────────▼─────────────────┐ │
│  │ 4. Check stop_on_error       │ │
│  └──────────────────────────────┘ │
└────────────────────────────────────┘
            │
            ▼
┌────────────────────────┐
│ Post-Validators (hooks)│
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Aggregate Results     │
│  - Merge errors        │
│  - Set valid flag      │
│  - Add metadata        │
└───────────┬────────────┘
            │
            ▼
    ValidationResult
```

---

## 📁 8. I/O UTILITIES ARCHITECTURE

```
                    DataReader
                        │
                        │ detect_format()
                        ▼
            ┌───────────────────────┐
            │  Format Detection     │
            │  (by extension)       │
            └───────────┬───────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
    ┌───────┐      ┌───────┐      ┌────────┐
    │  CSV  │      │  JSON │      │ Excel  │
    │Reader │      │Reader │      │ Reader │
    └───┬───┘      └───┬───┘      └───┬────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
            ┌──────────────────┐
            │  Pandas DataFrame │
            │  or Dict/List     │
            └──────────────────┘

DataWriter (Reverse Flow)
    Input Data
        │
        ▼
    Format Selection
        │
    ┌───┼───┐
    │   │   │
    ▼   ▼   ▼
   CSV JSON Excel ...
    │   │   │
    └───┼───┘
        │
        ▼
    File Written
```

### Resource Loading with Caching

```
load_resource(path)
        │
        ▼
    ┌─────────────┐
    │ In Cache?   │
    └─────┬───┬───┘
      YES │   │ NO
          │   │
          ▼   ▼
    ┌──────────┐    ┌─────────────────┐
    │ Return   │    │ 1. Read File    │
    │ Cached   │    │ 2. Parse Format │
    └──────────┘    │ 3. Store Cache  │
                    │ 4. Return Data  │
                    └─────────────────┘

Cache Structure:
┌─────────────────────────────────┐
│ _resources = {                  │
│   "path1": data1,               │
│   "path2": data2,               │
│   ...                           │
│ }                               │
└─────────────────────────────────┘
```

---

## 🎨 9. DECORATOR COMPOSITION

```
@traced(save_path="provenance.json")
@deterministic(seed=42)
@cached(ttl_seconds=3600)
def calculate_emissions(energy, factor):
    return energy * factor

Execution Flow:
User calls → traced → deterministic → cached → YOUR CODE
                │          │            │
                │          │            └─ Check cache
                │          └─ Set seed, hash inputs/outputs
                └─ Record provenance

Return flow:
YOUR CODE → cached → deterministic → traced → User receives
              │          │            │
              │          │            └─ Save provenance record
              │          └─ Attach deterministic metadata
              └─ Store in cache
```

### Decorator Interaction Diagram

```
┌─────────────────────────────────────────────────────┐
│ @traced Decorator                                   │
│ ┌─────────────────────────────────────────────────┐ │
│ │ @deterministic Decorator                        │ │
│ │ ┌─────────────────────────────────────────────┐ │ │
│ │ │ @cached Decorator                           │ │ │
│ │ │ ┌─────────────────────────────────────────┐ │ │ │
│ │ │ │   YOUR FUNCTION                         │ │ │ │
│ │ │ │   (calculate_emissions)                 │ │ │ │
│ │ │ └─────────────────────────────────────────┘ │ │ │
│ │ │ Provides: Result caching, LRU eviction      │ │ │
│ │ └─────────────────────────────────────────────┘ │ │
│ │ Provides: Seed setting, input/output hashing    │ │
│ └─────────────────────────────────────────────────┘ │
│ Provides: Provenance tracking, audit trail          │
└─────────────────────────────────────────────────────┘

Result has metadata from ALL decorators:
- Cache hit/miss info
- Deterministic hashes
- Provenance record ID
```

---

## 🔄 10. COMPLETE AGENT PIPELINE (CBAM Example)

```
                    CBAM Pipeline
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
    ┌──────────────┐ ┌─────────┐ ┌───────────┐
    │ Intake Agent │→│  Calc   │→│  Report   │
    │ (DataProc)   │ │ Agent   │ │  Agent    │
    └──────┬───────┘ │ (Calc)  │ │ (Report)  │
           │         └────┬────┘ └─────┬─────┘
           │              │            │
    Read CSV files   Calculate     Generate
    Validate data    emissions     reports
    Enrich with      Cache results Output MD/HTML/Excel
    CN codes
           │              │            │
           └──────────────┼────────────┘
                          │
                Provenance tracked end-to-end
```

### Agent Communication

```
ShipmentIntakeAgent
    │
    │ output: {"shipments": [...]}
    ▼
EmissionsCalculatorAgent
    │
    │ input: {"shipments": [...]}
    │ output: {"shipments": [...], "emissions": {...}}
    ▼
ReportingPackagerAgent
    │
    │ input: {"shipments": [...], "emissions": {...}}
    │ output: {"report_markdown": "...", "report_html": "..."}
    ▼
Final Reports
```

---

## 📈 11. METRICS & MONITORING FLOW

```
Every Agent.run() Execution
            │
            ▼
    ┌───────────────┐
    │ AgentMetrics  │
    │ - exec_time   │
    │ - input_size  │
    │ - output_size │
    │ - records     │
    │ - cache_hits  │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ StatsTracker  │
    │ - executions  │
    │ - successes   │
    │ - failures    │
    │ - avg_time    │
    │ - custom_*    │
    └───────┬───────┘
            │
            ▼
    ┌───────────────────┐
    │ get_stats()       │
    │ Returns aggregate │
    │ statistics        │
    └───────────────────┘

Used for:
- Performance monitoring
- Debugging
- Optimization
- Reporting
```

---

## 🎯 12. DEPLOYMENT ARCHITECTURE

```
┌─────────────────────────────────────────────────────┐
│ Developer Workstation                               │
│                                                     │
│  ┌────────────────────────────────────────────┐    │
│  │ Your Agent Code                            │    │
│  │ (extends BaseDataProcessor, etc.)          │    │
│  └────────────────┬───────────────────────────┘    │
│                   │                                 │
│  ┌────────────────▼───────────────────────────┐    │
│  │ GreenLang Framework                        │    │
│  │ - greenlang.agents                         │    │
│  │ - greenlang.provenance                     │    │
│  │ - greenlang.validation                     │    │
│  │ - greenlang.io                             │    │
│  └────────────────┬───────────────────────────┘    │
│                   │                                 │
│  ┌────────────────▼───────────────────────────┐    │
│  │ Python Runtime (3.8+)                      │    │
│  └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
                     │
                     │ deploy
                     ▼
┌─────────────────────────────────────────────────────┐
│ Production Environment                              │
│                                                     │
│  ┌────────────────────────────────────────────┐    │
│  │ Docker Container                           │    │
│  │  - Your Agent                              │    │
│  │  - GreenLang Framework                     │    │
│  │  - Dependencies                            │    │
│  └────────────────┬───────────────────────────┘    │
│                   │                                 │
│  ┌────────────────▼───────────────────────────┐    │
│  │ Kubernetes Pod / VM                        │    │
│  └────────────────┬───────────────────────────┘    │
│                   │                                 │
│  ┌────────────────▼───────────────────────────┐    │
│  │ Monitoring & Logging                       │    │
│  │ - Metrics collection                       │    │
│  │ - Provenance storage                       │    │
│  │ - Log aggregation                          │    │
│  └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 📚 13. MODULE DEPENDENCY GRAPH

```
greenlang/
    │
    ├── agents/
    │   ├── base.py ─────────────────┐
    │   ├── calculator.py ───→ base  │
    │   ├── data_processor.py →base  │
    │   ├── reporter.py ──────→base  │
    │   └── decorators.py            │
    │       │                        │
    │       └─────────────────┐      │
    │                         ▼      ▼
    ├── provenance/          Uses  Uses
    │   ├── hashing.py        │      │
    │   ├── environment.py    │      │
    │   ├── records.py ◄──────┘      │
    │   ├── validation.py            │
    │   ├── reporting.py             │
    │   └── decorators.py            │
    │                                │
    ├── validation/                  │
    │   ├── framework.py ◄───────────┘
    │   ├── schema.py
    │   ├── rules.py
    │   ├── quality.py
    │   └── decorators.py
    │
    └── io/
        ├── readers.py
        ├── writers.py
        ├── resources.py
        ├── formats.py
        └── streaming.py

External Dependencies:
- pydantic (data validation)
- pandas (data processing)
- tqdm (progress bars)
- openpyxl (Excel support)
```

---

## 🎓 14. LEARNING PATH (From Simple to Complex)

```
Level 1: Basic Agent
    └── Create simple BaseAgent
        └── Override execute()
            └── Return AgentResult

Level 2: Data Processing
    └── Extend BaseDataProcessor
        └── Override process_record()
            └── Handle batches automatically

Level 3: Calculations
    └── Extend BaseCalculator
        └── Override calculate()
            └── Get caching for free

Level 4: Reports
    └── Extend BaseReporter
        └── Override aggregate_data() & build_sections()
            └── Multi-format output

Level 5: Add Provenance
    └── Use @traced decorator
        └── Get audit trail automatically

Level 6: Add Validation
    └── Integrate ValidationFramework
        └── Schema + Business Rules

Level 7: Pipelines
    └── Chain multiple agents
        └── Data flows between stages

Level 8: Production
    └── Add monitoring
        └── Deploy to Kubernetes
            └── Scale horizontally
```

---

## 🔍 15. DEBUGGING & TROUBLESHOOTING FLOW

```
Issue Occurs
    │
    ▼
┌─────────────────────────┐
│ Check Agent Logs        │
│ (logger.debug/info)     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Review AgentMetrics     │
│ (execution time, etc)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Examine Provenance      │
│ (inputs, environment)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Check Validation Errors │
│ (ValidationResult)      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Run Tests               │
│ (unit + integration)    │
└───────────┬─────────────┘
            │
            ▼
    Issue Resolved
```

---

## 📊 SUMMARY

These diagrams show:

1. **High-level architecture** - How modules fit together
2. **Class hierarchies** - Inheritance and responsibilities
3. **Data flows** - How data moves through the system
4. **Execution flows** - Step-by-step processing
5. **Component interactions** - How pieces communicate
6. **Deployment architecture** - Production setup
7. **Learning progression** - How to master the framework

**For more details, see:**
- `docs/API_REFERENCE.md` - Detailed API documentation
- `docs/QUICK_START.md` - Getting started guide
- `examples/` - Working code examples

---

**Document Version:** 1.0.0
**Last Updated:** 2025-10-17
**Maintainer:** GreenLang Core Team
