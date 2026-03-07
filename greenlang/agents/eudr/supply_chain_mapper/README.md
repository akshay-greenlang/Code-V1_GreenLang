# AGENT-EUDR-001: Supply Chain Mapping Master

**Agent ID:** GL-EUDR-SCM-001
**Version:** 1.0.0
**Status:** Production Ready
**Regulation:** EU Regulation 2023/1115 -- EU Deforestation Regulation (EUDR)
**PRD:** PRD-AGENT-EUDR-001

---

## Overview

The Supply Chain Mapping Master is a graph-native supply chain modeling engine
purpose-built for EU Deforestation Regulation (EUDR) compliance. It enables
operators and traders placing EUDR-regulated commodities on the EU market to
model, analyze, and verify their entire supply chain from production plot to
point-of-entry.

The agent models every actor (producer, collector, processor, trader, importer),
every custody transfer, every batch split/merge, and every production plot as
nodes and edges in a directed acyclic graph (DAG). It provides deterministic
risk propagation, automated gap analysis, multi-tier recursive discovery,
plot-level geolocation integration, and regulatory export capabilities.

### Key Value Propositions

- **Full EUDR compliance** for supply chain traceability (Articles 2, 3, 4, 9, 10, 11, 12, 29, 31)
- **Graph-native architecture** -- the supply chain IS a graph from day one
- **Zero-hallucination** -- 100% deterministic, bit-perfect reproducibility with no LLM in the critical path
- **Scale** -- tested for 100,000+ node graphs with sub-second query times
- **All 7 EUDR commodities** -- cattle, cocoa, coffee, palm oil, rubber, soya, wood (plus derived products)

---

## EUDR Compliance Coverage

The agent addresses the following EUDR articles:

| Article | Requirement | Implementation |
|---------|-------------|----------------|
| **Art. 2(1-3)** | Definitions of "deforestation", "deforestation-free" | Deforestation status tracked at plot nodes; Dec 31, 2020 cutoff enforced |
| **Art. 2(30-32)** | "Plot of land", "geolocation" definitions | Plot nodes with GPS coordinates and polygon boundaries (WGS84) |
| **Art. 3** | Prohibition on non-compliant products | Gap analysis identifies non-traceable products |
| **Art. 4(2)** | Due diligence -- collect supply chain information | Multi-tier mapping with automated discovery |
| **Art. 4(2)(f)** | Supply chain information in DDS | Graph export formatted for DDS submission |
| **Art. 9(1)(a-d)** | Geolocation of all plots of land | Plot-level GPS/polygon integration via GeolocationLinker |
| **Art. 9(1)(d)** | Polygon for plots > 4 hectares | Polygon validation engine with PolygonValidation |
| **Art. 10** | Risk assessment | Risk propagation through supply chain graph |
| **Art. 10(2)(f)** | Risk of mixing with unknown-origin products | Batch mixing detection and mass balance verification |
| **Art. 11** | Risk mitigation measures | Risk-based supply chain segmentation |
| **Art. 12** | DDS submission to EU Information System | RegulatoryExporter with DDS JSON/XML/PDF generation |
| **Art. 29** | Country benchmarking (Low/Standard/High) | Country risk classification at graph nodes |
| **Art. 31** | Record keeping for 5 years | Immutable graph snapshots with SHA-256 provenance chain |

---

## Architecture

```
+----------------------------------------------------------------------+
|                    AGENT-EUDR-001: Supply Chain Mapping Master        |
+----------------------------------------------------------------------+
|                                                                      |
|  +------------------+  +-------------------+  +-------------------+  |
|  | SupplyChainGraph |  | MultiTierMapper   |  | GeolocationLinker |  |
|  | Engine           |  | (Feature 2)       |  | (Feature 3)       |  |
|  | (Feature 1)      |  |                   |  |                   |  |
|  | - DAG modeling   |  | - Recursive       |  | - PostGIS queries |  |
|  | - Cycle detect   |  |   discovery       |  | - Polygon valid.  |  |
|  | - Topo sort      |  | - ERP/Questionnaire|  | - Distance calc   |  |
|  | - Serialization  |  |   integration     |  | - Protected areas |  |
|  +------------------+  +-------------------+  +-------------------+  |
|                                                                      |
|  +------------------+  +-------------------+  +-------------------+  |
|  | Batch Trace      |  | RiskPropagation   |  | GapAnalyzer       |  |
|  | (Feature 4)      |  | Engine            |  | (Feature 6)       |  |
|  |                  |  | (Feature 5)       |  |                   |  |
|  | - Forward trace  |  | - Weighted risk   |  | - 10 gap types    |  |
|  | - Backward trace |  | - Highest wins    |  | - Auto-remediate  |  |
|  | - Split/merge    |  | - 4 dimensions    |  | - Trend tracking  |  |
|  | - Mass balance   |  | - Deterministic   |  | - Compliance score|  |
|  +------------------+  +-------------------+  +-------------------+  |
|                                                                      |
|  +------------------+  +-------------------+  +-------------------+  |
|  | Visualization    |  | SupplierOnboarding|  | Regulatory        |  |
|  | Engine           |  | Engine            |  | Exporter          |  |
|  | (Feature 7)      |  | (Feature 8)       |  | (Feature 9)       |  |
|  |                  |  |                   |  |                   |  |
|  | - Force-directed |  | - Invite wizard   |  | - DDS JSON/XML    |  |
|  | - Hierarchical   |  | - Multi-language  |  | - PDF generation  |  |
|  | - Sankey export  |  | - Token-based     |  | - Batch export    |  |
|  | - D3.js/Cytoscape|  | - Bulk import     |  | - Schema validate |  |
|  +------------------+  +-------------------+  +-------------------+  |
|                                                                      |
|  +-------------------------------------------------------------------+
|  | Foundational Layer                                                |
|  | - models.py      (24 Pydantic v2 models, 7 enums, 12 constants)  |
|  | - config.py      (22 GL_EUDR_SCM_* env vars, validated config)   |
|  | - provenance.py  (SHA-256 chain-hashed audit trail)               |
|  | - metrics.py     (15 Prometheus metrics, gl_eudr_scm_ prefix)     |
|  | - setup.py       (Service facade, lifespan, DB/Redis/OTel)        |
|  +-------------------------------------------------------------------+
|                                                                      |
|  +-------------------------------------------------------------------+
|  | API Layer (FastAPI)                                                |
|  | router.py         -> /v1/eudr-scm prefix, 23+ endpoints           |
|  | graph_routes.py   -> POST/GET/DELETE /graphs, GET /export          |
|  | mapping_routes.py -> POST /discover, GET /tiers                    |
|  | traceability_routes.py -> GET /trace/forward, /backward, /batch    |
|  | risk_routes.py    -> POST /risk/propagate, GET /summary, /heatmap  |
|  | gap_routes.py     -> POST /gaps/analyze, GET /gaps, PUT /resolve   |
|  | visualization_routes.py -> GET /layout, GET /sankey                |
|  | onboarding_routes.py -> POST /invite, GET /status, POST /submit   |
|  | dependencies.py   -> JWT auth, RBAC, rate limiting, pagination     |
|  | schemas.py        -> API-level Pydantic request/response models    |
|  +-------------------------------------------------------------------+
+----------------------------------------------------------------------+
```

### Data Flow

```
  External Data Sources          Supply Chain Mapper           Outputs
  =====================          ====================         ========

  AGENT-DATA-003 (ERP) ---+
  AGENT-DATA-005 (EUDR) --+--> MultiTierMapper ----+
  AGENT-DATA-008 (Quest) -+    (recursive discover)|
                               |                   |
  AGENT-DATA-006 (GIS) ------> GeolocationLinker --+--> SupplyChainGraph
  AGENT-DATA-007 (Satellite) -> (plot linkage)     |    (core DAG model)
                               |                   |
                               +-------------------+
                                         |
                          +--------------+--------------+
                          |              |              |
                     RiskPropagation  GapAnalyzer  BatchTrace
                     Engine          (compliance)  (traceability)
                          |              |              |
                          +--------------+--------------+
                                         |
                          +--------------+--------------+
                          |              |              |
                     Visualization  Regulatory    Supplier
                     Engine        Exporter      Onboarding
                          |              |              |
                          v              v              v
                     D3.js/Sankey   DDS JSON/XML  Invite Portal
                     Layout Data    PDF Reports   Token-based
```

---

## Quick Start

### Installation

The Supply Chain Mapper is part of the GreenLang platform. Ensure you have the
greenlang package installed:

```bash
pip install greenlang[eudr]
```

### Basic Usage

```python
from greenlang.agents.eudr.supply_chain_mapper import (
    SupplyChainNode,
    SupplyChainEdge,
    SupplyChainGraph,
    NodeType,
    EUDRCommodity,
    CustodyModel,
)
from decimal import Decimal

# 1. Create a supply chain graph
graph = SupplyChainGraph(
    operator_id="op-eu-chocolate-001",
    commodity=EUDRCommodity.COCOA,
    graph_name="Ghana Cocoa Supply Chain Q1 2026",
)

# 2. Add a producer node (farm/cooperative in Ghana)
producer = SupplyChainNode(
    node_type=NodeType.PRODUCER,
    operator_id="gh-coop-alpha",
    operator_name="Cooperative Alpha",
    country_code="GH",
    coordinates=(6.6885, -1.6244),
    commodities=[EUDRCommodity.COCOA],
    plot_ids=["plot-gh-001", "plot-gh-002"],
    certifications=["RA-2024-GH-001"],
)

# 3. Add a processor node (mill in Ghana)
processor = SupplyChainNode(
    node_type=NodeType.PROCESSOR,
    operator_id="gh-mill-beta",
    operator_name="Beta Processing Mill",
    country_code="GH",
    commodities=[EUDRCommodity.COCOA],
    tier_depth=1,
)

# 4. Add an importer node (EU operator)
importer = SupplyChainNode(
    node_type=NodeType.IMPORTER,
    operator_id="eu-choc-001",
    operator_name="EU Chocolate GmbH",
    country_code="DE",
    commodities=[EUDRCommodity.CHOCOLATE],
    tier_depth=0,
)

# 5. Add nodes to graph
graph.nodes[producer.node_id] = producer
graph.nodes[processor.node_id] = processor
graph.nodes[importer.node_id] = importer
graph.total_nodes = 3

# 6. Create custody transfer edges
edge_1 = SupplyChainEdge(
    source_node_id=producer.node_id,
    target_node_id=processor.node_id,
    commodity=EUDRCommodity.COCOA,
    product_description="Dried cocoa beans (Grade A)",
    quantity=Decimal("5000"),
    unit="kg",
    batch_number="BATCH-GH-2026-001",
    custody_model=CustodyModel.SEGREGATED,
)

edge_2 = SupplyChainEdge(
    source_node_id=processor.node_id,
    target_node_id=importer.node_id,
    commodity=EUDRCommodity.COCOA,
    product_description="Cocoa liquor",
    quantity=Decimal("4200"),
    unit="kg",
    batch_number="BATCH-GH-2026-002",
    custody_model=CustodyModel.SEGREGATED,
)

graph.edges[edge_1.edge_id] = edge_1
graph.edges[edge_2.edge_id] = edge_2
graph.total_edges = 2
graph.max_tier_depth = 2

print(f"Graph: {graph.graph_name}")
print(f"Nodes: {graph.total_nodes}, Edges: {graph.total_edges}")
print(f"Max tier depth: {graph.max_tier_depth}")
```

### Using the Graph Engine

```python
from greenlang.agents.eudr.supply_chain_mapper import (
    SupplyChainGraphEngine,
)
from greenlang.agents.eudr.supply_chain_mapper.config import get_config

config = get_config()
engine = SupplyChainGraphEngine(config)
await engine.initialize()

# Create a graph
graph_id = await engine.create_graph(
    operator_id="op-123",
    commodity="cocoa",
    graph_name="Ghana Cocoa Supply Chain",
)

# Add nodes
node_id = await engine.add_node(
    graph_id=graph_id,
    node_type=NodeType.PRODUCER,
    operator_name="Cooperative Alpha",
    country_code="GH",
)
```

### Using the REST API

```bash
# Create a graph
curl -X POST "https://api.greenlang.io/api/v1/eudr-scm/graphs" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"commodity": "cocoa", "graph_name": "Ghana Cocoa Q1 2026"}'

# Run gap analysis
curl -X POST "https://api.greenlang.io/api/v1/eudr-scm/graphs/$GRAPH_ID/gaps/analyze" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"include_resolved": false}'

# Backward trace to origin plots
curl "https://api.greenlang.io/api/v1/eudr-scm/graphs/$GRAPH_ID/trace/backward/$NODE_ID" \
  -H "Authorization: Bearer $TOKEN"
```

---

## Core Capabilities

### 1. Supply Chain Graph Engine (Feature 1)

The core DAG engine models typed supply chain actors and directed custody
transfer edges with full CRUD, cycle detection, topological sorting, and
graph serialization.

- **8 node types:** Producer, Collector, Processor, Trader, Importer, Certifier, Warehouse, Port
- **Directed edges** with commodity, quantity (Decimal), batch number, custody model, transport mode
- **Cycle detection** to enforce DAG topology
- **Topological sorting** for processing order
- **Serialization** to JSON, GraphML, and internal binary format
- **Graph versioning** with immutable snapshots
- **100,000+ nodes** with <1ms single-node lookup

### 2. Multi-Tier Recursive Mapping (Feature 2)

Automatically discovers and maps supply chain depth from Tier 1 through
Tier N using ERP data, supplier questionnaires, and document extraction.

- **Recursive BFS/DFS** discovery through supplier networks
- **7 commodity-specific archetypes** with expected tier depths
- **Opaque segment detection** where sub-tier visibility is missing
- **Incremental mapping** -- add new tiers without rebuilding
- **Integration protocols** for ERP, questionnaire, PDF, and bulk importers

### 3. Plot-Level Geolocation Integration (Feature 3)

Links every producer node to registered plots with GPS coordinates and
polygon boundaries per EUDR Article 9.

- **WGS84 coordinate validation** with 6+ decimal place precision
- **Polygon enforcement** for plots > 4 hectares
- **PostGIS spatial queries** for bounding box lookups (<100ms for 100K plots)
- **Protected area cross-referencing**
- **Deforestation satellite alert integration** via AGENT-DATA-007
- **Distance metrics** between supply chain nodes

### 4. Many-to-Many Batch Traceability (Feature 4)

Traces every final product back to all contributing production plots through
batch splits, merges, and transformations.

- **Three custody models:** Identity Preserved, Segregated, Mass Balance
- **Forward trace:** "Which products contain commodity from Plot X?"
- **Backward trace:** "Which plots contributed to Product Y?"
- **Batch trace:** "Where did batch BATCH-001 originate and where did it go?"
- **Decimal arithmetic** for mass balance (no floating-point drift)
- **SHA-256 provenance hash** on every batch operation

### 5. Risk Propagation Engine (Feature 5)

Propagates risk scores through the supply chain graph using configurable
weights across four dimensions.

- **Four risk dimensions:** Country (0.30), Commodity (0.20), Supplier (0.25), Deforestation (0.25)
- **Weighted composite score** on 0-100 scale
- **Three risk levels:** Low (<= 30), Standard (31-69), High (>= 70)
- **"Highest risk wins"** propagation principle per EUDR Article 10
- **Deterministic, bit-perfect** -- no LLM or probabilistic components
- **Per-node risk factor breakdown** for audit transparency

### 6. Supply Chain Gap Analysis (Feature 6)

Detects 10 types of compliance gaps, each mapped to a specific EUDR article.

| Gap Type | Severity | EUDR Article |
|----------|----------|--------------|
| Missing Geolocation | Critical | Article 9 |
| Missing Polygon | Critical | Article 9(1)(d) |
| Broken Custody Chain | Critical | Article 4(2)(f) |
| Unverified Actor | High | Article 10 |
| Missing Tier | High | Article 4(2) |
| Mass Balance Discrepancy | High | Article 10(2)(f) |
| Missing Certification | Medium | Article 10 |
| Stale Data | Medium | Article 31 |
| Missing Documentation | Medium | Article 4(2) |
| Orphan Node | Low | Internal |

- **Auto-remediation suggestions** for each gap type
- **Compliance readiness scoring** (0-100%)
- **Trend tracking** over time
- **Severity-based prioritized remediation queue**

### 7. Visualization Engine (Feature 7)

Generates graph layouts and Sankey diagrams for frontend rendering.

- **Hierarchical layout** by tier depth (left-to-right)
- **Force-directed layout** for dense graphs
- **Sankey flow diagrams** showing commodity volumes
- **Color-coded** by node type and risk level
- **Styled edges** by custody model and quantity
- **Compatible with** D3.js, vis-network, Cytoscape.js

### 8. Supplier Onboarding (Feature 8)

Enables operators to invite sub-tier suppliers to self-register their
supply chain data through a token-based portal.

- **Secure token generation** (SHA-256, 48-character)
- **Time-limited invitations** (configurable 1-365 days)
- **Public portal endpoints** (no authentication required for suppliers)
- **Multi-language support** (EN, FR, DE, ES, PT, ID, MS, TH, VI)
- **Self-registration wizard** with GPS coordinates, certifications, sub-suppliers
- **Automatic graph integration** when linked to a graph

### 9. Regulatory Exporter (Feature 9)

Generates EUDR-compliant Due Diligence Statement (DDS) exports.

- **DDS JSON Schema** validation per EU Information System specification
- **XML serialization** for alternative submission format
- **PDF report generation** for audit packages
- **Batch export** for multiple graphs
- **Incremental export** for delta changes
- **Article 4(2) field coverage** verification

---

## Performance Characteristics

| Metric | Target | Implementation |
|--------|--------|----------------|
| Graph query (10K nodes) | < 500ms p99 | BFS/DFS with adjacency lists |
| Single-node lookup | < 1ms | Dictionary-based O(1) access |
| Batch throughput | 50,000 transfers/min | Async processing pipeline |
| Memory (100K nodes) | < 2 GB | Pydantic v2 with dict storage |
| API response (standard) | < 200ms p95 | FastAPI with async handlers |
| Graph construction (10K nodes) | < 5 seconds | Incremental node/edge add |
| Backward trace (50 hops) | < 2 seconds | BFS with visited set |
| Visualization render | < 3 seconds (1K nodes) | Pre-computed layouts |

---

## Integration Points

The Supply Chain Mapper integrates with the following GreenLang agents and
applications:

### Data Agents (Inbound)

| Agent | Integration | Purpose |
|-------|-------------|---------|
| **AGENT-DATA-003** (ERP/Finance) | ERPConnectorProtocol | Tier 1 procurement data import |
| **AGENT-DATA-005** (EUDR Traceability) | ChainOfCustodyEngine, PlotRegistryEngine | Custody transfers, plot geolocation |
| **AGENT-DATA-006** (GIS/Mapping) | PostGIS spatial queries | Geolocation validation, protected areas |
| **AGENT-DATA-007** (Deforestation Satellite) | Alert cross-referencing | Deforestation-free verification |
| **AGENT-DATA-008** (Supplier Questionnaire) | QuestionnaireProcessorProtocol | Sub-tier supplier declarations |
| **AGENT-DATA-001** (PDF Extractor) | PDFExtractorProtocol | Custody document extraction |
| **AGENT-DATA-002** (Excel/CSV) | BulkImporterProtocol | Bulk supplier import |

### Applications (Outbound)

| Application | Integration | Purpose |
|-------------|-------------|---------|
| **GL-EUDR-APP** | SupplyChainMapperService facade | Full integration as supply chain module |
| **GL-EUDR-APP DDS** | RegulatoryExporter | Supply chain section of DDS filing |

### Infrastructure

| Component | Integration | Purpose |
|-----------|-------------|---------|
| PostgreSQL + TimescaleDB | psycopg + psycopg_pool | Persistent graph storage |
| Redis | aioredis | Graph query caching, distributed locks |
| Prometheus | 15 metrics (gl_eudr_scm_* prefix) | Service monitoring |
| Grafana | Pre-built dashboard | Operational visibility |
| OpenTelemetry | Distributed tracing | Request tracing across services |
| Kong API Gateway | Rate limiting, auth | API traffic management |

For detailed integration guides, see [INTEGRATION.md](INTEGRATION.md).

---

## Configuration

All settings are configurable via environment variables with the `GL_EUDR_SCM_`
prefix. See [DEPLOYMENT.md](DEPLOYMENT.md) for the complete reference.

Key configuration groups:

- **Connections:** `GL_EUDR_SCM_DATABASE_URL`, `GL_EUDR_SCM_REDIS_URL`
- **Risk Weights:** `GL_EUDR_SCM_RISK_WEIGHT_COUNTRY` (0.30), `_COMMODITY` (0.20), `_SUPPLIER` (0.25), `_DEFORESTATION` (0.25)
- **Graph Limits:** `GL_EUDR_SCM_MAX_NODES_PER_GRAPH` (100,000), `_MAX_EDGES_PER_GRAPH` (500,000)
- **Performance:** `GL_EUDR_SCM_GRAPH_QUERY_TIMEOUT_MS` (500), `_BATCH_THROUGHPUT_TARGET` (50,000)
- **Risk Thresholds:** `GL_EUDR_SCM_RISK_HIGH_THRESHOLD` (70), `_RISK_LOW_THRESHOLD` (30)
- **Provenance:** `GL_EUDR_SCM_ENABLE_PROVENANCE` (true), `_GENESIS_HASH`

---

## API Reference

The agent exposes 23+ REST API endpoints under `/v1/eudr-scm`. See [API.md](API.md)
for the complete endpoint reference with request/response schemas, authentication
requirements, and cURL examples.

**Endpoint Groups:**

| Group | Endpoints | Description |
|-------|-----------|-------------|
| Graph CRUD | 5 | Create, list, get, delete, export graphs |
| Multi-Tier Mapping | 2 | Trigger discovery, get tier distribution |
| Traceability | 3 | Forward trace, backward trace, batch trace |
| Risk Assessment | 3 | Propagate risk, get summary, get heatmap |
| Gap Analysis | 3 | Analyze gaps, list gaps, resolve gaps |
| Visualization | 2 | Get layout, get Sankey data |
| Supplier Onboarding | 3 | Invite supplier, check status, submit data |
| System | 1 | Health check |

---

## Project Structure

```
greenlang/agents/eudr/supply_chain_mapper/
    __init__.py                 # Public API surface (130+ exports)
    models.py                   # 24 Pydantic v2 models, 7 enums, constants
    config.py                   # SupplyChainMapperConfig (22 GL_EUDR_SCM_* vars)
    provenance.py               # SHA-256 chain-hashed audit trail
    metrics.py                  # 15 Prometheus metrics (gl_eudr_scm_*)
    graph_engine.py             # Core DAG graph engine (Feature 1)
    multi_tier_mapper.py        # Recursive supply chain discovery (Feature 2)
    geolocation_linker.py       # Plot-level geolocation (Feature 3)
    risk_propagation.py         # Deterministic risk propagation (Feature 5)
    gap_analyzer.py             # Compliance gap detection (Feature 6)
    visualization_engine.py     # Graph layout and Sankey (Feature 7)
    supplier_onboarding.py      # Supplier onboarding workflow (Feature 8)
    regulatory_exporter.py      # DDS and regulatory export (Feature 9)
    setup.py                    # Service facade, lifespan, DB/Redis
    api/
        __init__.py
        router.py               # Main router (/v1/eudr-scm)
        dependencies.py         # Auth, RBAC, rate limiting, pagination
        schemas.py              # API request/response models
        graph_routes.py         # Graph CRUD + health endpoints
        mapping_routes.py       # Multi-tier discovery endpoints
        traceability_routes.py  # Forward/backward/batch trace
        risk_routes.py          # Risk propagation + summary
        gap_routes.py           # Gap analysis + resolution
        visualization_routes.py # Layout + Sankey endpoints
        onboarding_routes.py    # Supplier onboarding endpoints
```

---

## Related Documentation

- [API.md](API.md) -- Complete REST API endpoint reference
- [INTEGRATION.md](INTEGRATION.md) -- Integration guide with data agents
- [DEPLOYMENT.md](DEPLOYMENT.md) -- Kubernetes deployment and operations guide
- [Technical Specification](../../../GreenLang%20Development/05-Documentation/AGENT-EUDR-001-Technical-Specification.md) -- Full technical specification

---

## License

Copyright 2026 GreenLang Platform. All rights reserved.
