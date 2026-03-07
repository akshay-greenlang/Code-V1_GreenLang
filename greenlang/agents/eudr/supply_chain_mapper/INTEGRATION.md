# AGENT-EUDR-001: Integration Guide

This document describes the integration points between the Supply Chain Mapping
Master (AGENT-EUDR-001) and the GreenLang data agents, applications, and
infrastructure components it depends on.

---

## Table of Contents

1. [Integration Overview](#integration-overview)
2. [AGENT-DATA-005: EUDR Traceability Connector](#agent-data-005-eudr-traceability-connector)
3. [AGENT-DATA-006: GIS/Mapping Connector](#agent-data-006-gismapping-connector)
4. [AGENT-DATA-007: Deforestation Satellite Connector](#agent-data-007-deforestation-satellite-connector)
5. [AGENT-DATA-003: ERP/Finance Connector](#agent-data-003-erpfinance-connector)
6. [AGENT-DATA-008: Supplier Questionnaire Processor](#agent-data-008-supplier-questionnaire-processor)
7. [GL-EUDR-APP Integration](#gl-eudr-app-integration)
8. [Data Flow Diagrams](#data-flow-diagrams)
9. [API Call Sequences](#api-call-sequences)

---

## Integration Overview

The Supply Chain Mapper operates as a hub that consumes data from five upstream
data agents and exports enriched supply chain intelligence to the GL-EUDR-APP
platform.

```
  Upstream Data Agents                    Supply Chain Mapper                 Downstream
  =====================                   ====================               ==========

  AGENT-DATA-005 (EUDR) -----+
    - ChainOfCustodyEngine    |
    - PlotRegistryEngine      |
    - BatchRecordEngine       |
                              +--->  SupplyChainMapperService
  AGENT-DATA-006 (GIS) ------+        |
    - SpatialQueryEngine      |        |-- SupplyChainGraphEngine
    - ProtectedAreaDB         |        |-- MultiTierMapper
                              |        |-- GeolocationLinker         ---> GL-EUDR-APP
  AGENT-DATA-007 (Satellite) -+        |-- RiskPropagationEngine          - DDS Filing
    - DeforestationAlerts     |        |-- GapAnalyzer                    - Dashboard
    - ChangeDetectionEngine   |        |-- VisualizationEngine            - Reports
                              |        |-- RegulatoryExporter
  AGENT-DATA-003 (ERP) ------+        |-- SupplierOnboardingEngine
    - ProcurementExtractor    |
    - SupplierMasterData      |
                              |
  AGENT-DATA-008 (Questionnaire) +
    - ResponseParser          |
    - SubTierExtractor        |
```

### Integration Protocols

The MultiTierMapper defines four protocol interfaces for decoupled integration:

```python
class ERPConnectorProtocol(Protocol):
    """Interface for ERP/Finance data integration."""
    async def get_supplier_records(
        self, operator_id: str, commodity: str
    ) -> List[SupplierRecord]: ...

class QuestionnaireProcessorProtocol(Protocol):
    """Interface for supplier questionnaire data."""
    async def get_sub_tier_suppliers(
        self, supplier_id: str
    ) -> List[SupplierRecord]: ...

class PDFExtractorProtocol(Protocol):
    """Interface for custody document extraction."""
    async def extract_custody_data(
        self, document_url: str
    ) -> List[Dict[str, Any]]: ...

class BulkImporterProtocol(Protocol):
    """Interface for bulk CSV/Excel supplier import."""
    async def import_suppliers(
        self, file_path: str, format: str
    ) -> List[SupplierRecord]: ...
```

---

## AGENT-DATA-005: EUDR Traceability Connector

**Agent ID:** AGENT-DATA-005
**Purpose:** Chain-of-custody data, plot geolocation, batch records
**Integration Type:** Python module import (same process) and async API

### Data Consumed

| Data Source | Engine | Data Type | Usage |
|-------------|--------|-----------|-------|
| Custody Transfers | ChainOfCustodyEngine | CustodyTransferRecord | Edges in supply chain graph |
| Plot Registry | PlotRegistryEngine | PlotRecord | Producer node geolocation |
| Batch Records | BatchRecordEngine | BatchRecord | Batch split/merge tracing |
| Operator Registry | OperatorRegistryEngine | OperatorRecord | Node identity verification |

### Integration Pattern

```python
from greenlang.eudr_traceability.chain_of_custody import ChainOfCustodyEngine
from greenlang.eudr_traceability.plot_registry import PlotRegistryEngine

# 1. Fetch custody transfers for an operator
coc_engine = ChainOfCustodyEngine(db_pool)
transfers = await coc_engine.get_transfers_by_operator(
    operator_id="op-001",
    commodity="cocoa",
    date_from="2025-01-01",
    date_to="2026-03-01",
)

# 2. Convert transfers to supply chain edges
for transfer in transfers:
    edge = SupplyChainEdge(
        source_node_id=transfer.source_operator_id,
        target_node_id=transfer.target_operator_id,
        commodity=EUDRCommodity(transfer.commodity),
        product_description=transfer.product_description,
        quantity=transfer.quantity,
        unit=transfer.unit,
        batch_number=transfer.batch_id,
        custody_model=CustodyModel(transfer.custody_model),
        transfer_date=transfer.transfer_date,
        hs_code=transfer.hs_code,
    )

# 3. Fetch plot geolocation for producers
plot_engine = PlotRegistryEngine(db_pool)
plots = await plot_engine.get_plots_by_operator(operator_id="op-001")

# 4. Link plots to producer nodes via GeolocationLinker
linker = GeolocationLinker(config)
for plot in plots:
    linkage = await linker.link_plot_to_node(
        plot_id=plot.plot_id,
        coordinates=(plot.latitude, plot.longitude),
        polygon=plot.polygon_wkt,
        node_id=producer_node_id,
    )
```

### Shared Models

The Supply Chain Mapper imports and extends models from AGENT-DATA-005 for
cross-agent consistency:

- `EUDRCommodity` -- shared enumeration of 7 primary + derived commodities
- `CustodyModel` -- Identity Preserved, Segregated, Mass Balance
- `RiskLevel` -- Low, Standard, High (per EUDR Article 29)
- `ComplianceStatus` -- shared compliance state machine

### Data Flow: AGENT-DATA-005 to Supply Chain Mapper

```
AGENT-DATA-005                            Supply Chain Mapper
==============                            ====================

PlotRegistryEngine                        GeolocationLinker
  |-- get_plots_by_operator()  -------->    |-- link_plot_to_node()
  |-- get_plots_in_bbox()      -------->    |-- validate_coordinates()
  |-- validate_polygon()       -------->    |-- check_polygon_requirement()

ChainOfCustodyEngine                      SupplyChainGraphEngine
  |-- get_transfers()          -------->    |-- add_edge()
  |-- get_batch_history()      -------->    |-- trace_batch()
  |-- verify_mass_balance()    -------->    |-- verify_mass_balance()

BatchRecordEngine                         Batch Traceability
  |-- get_batch_lineage()      -------->    |-- trace_forward()
  |-- get_split_records()      -------->    |-- trace_backward()
  |-- get_merge_records()      -------->    |-- resolve_origins()
```

---

## AGENT-DATA-006: GIS/Mapping Connector

**Agent ID:** AGENT-DATA-006
**Purpose:** Spatial operations, protected area verification, map rendering
**Integration Type:** PostGIS queries via psycopg async

### Data Consumed

| Data Source | Function | Usage |
|-------------|----------|-------|
| Spatial Queries | ST_Contains, ST_Distance | Plot-to-node distance validation |
| Protected Areas | Protected area boundaries | Cross-referencing plots against reserves |
| Administrative Boundaries | Country/region polygons | Jurisdiction verification |
| Elevation/Terrain | DEM data | Forest classification support |

### Integration Pattern

```python
from greenlang.agents.eudr.supply_chain_mapper.geolocation_linker import (
    GeolocationLinker,
    PostGISQueryBuilder,
)

# PostGIS spatial query builder
query_builder = PostGISQueryBuilder()

# Check if plot is within a protected area
protected_query = query_builder.build_contains_query(
    point_lat=6.6885,
    point_lon=-1.6244,
    target_table="eudr_protected_areas",
)
# Result: SELECT * FROM eudr_protected_areas
#         WHERE ST_Contains(geom, ST_SetSRID(ST_Point(-1.6244, 6.6885), 4326))

# Calculate distance between two supply chain nodes
distance_query = query_builder.build_distance_query(
    lat1=6.6885, lon1=-1.6244,  # Producer in Ghana
    lat2=51.2277, lon2=6.7735,   # Importer in Germany
)
# Returns distance in meters (Great Circle)

# Find all plots within a bounding box
bbox_query = query_builder.build_bbox_query(
    min_lat=5.0, min_lon=-3.0,
    max_lat=8.0, max_lon=1.0,
    target_table="eudr_production_plots",
)
```

### Spatial Operations Used

| Operation | PostGIS Function | Purpose |
|-----------|-----------------|---------|
| Point-in-polygon | `ST_Contains` | Plot within protected area check |
| Distance calculation | `ST_Distance` / `ST_DistanceSphere` | Node-to-node distance |
| Bounding box search | `ST_MakeEnvelope` + `ST_Intersects` | Regional plot queries |
| Polygon validation | `ST_IsValid` + `ST_Area` | Plot boundary verification |
| Coordinate transform | `ST_Transform` | CRS conversion to WGS84 |

---

## AGENT-DATA-007: Deforestation Satellite Connector

**Agent ID:** AGENT-DATA-007
**Purpose:** Satellite-based deforestation alerts and change detection
**Integration Type:** Async API calls to satellite alert service

### Data Consumed

| Data Source | Format | Usage |
|-------------|--------|-------|
| GLAD Forest Alerts | GeoTIFF/API | Deforestation detection since Dec 2020 |
| Hansen Global Forest Change | Raster tiles | Tree cover loss verification |
| Sentinel-2 Imagery | API | Cloud-free land cover classification |
| RADD Alerts | API | Radar-based deforestation alerts |

### Integration Pattern

```python
from greenlang.agents.eudr.supply_chain_mapper.geolocation_linker import (
    GeolocationLinker,
)

linker = GeolocationLinker(config)

# Check deforestation status for a plot
deforestation_result = await linker.check_deforestation_status(
    plot_id="plot-gh-001",
    coordinates=(6.6885, -1.6244),
    polygon_wkt="POLYGON((...),)",
    cutoff_date="2020-12-31",  # EUDR cutoff
)

# Result:
# {
#     "plot_id": "plot-gh-001",
#     "deforestation_detected": false,
#     "confidence": 0.95,
#     "data_source": "GLAD",
#     "last_checked": "2026-03-01T00:00:00Z",
#     "forest_cover_2020": 0.87,
#     "forest_cover_current": 0.85,
#     "change_pct": -2.3,
# }
```

### Deforestation Risk Integration with Risk Propagation

```
AGENT-DATA-007                          Supply Chain Mapper
==============                          ====================

DeforestationAlerts                     RiskPropagationEngine
  |-- get_alerts_for_polygon() ------>    |-- deforestation_risk_score
  |                                       |    (weight: 0.25)
  |-- get_forest_change()      ------>    |
  |                                       |
  |-- get_confidence_score()   ------>    |-- propagate_to_downstream()
                                          |    (highest risk wins)
                                          |
                                          v
                                     Node risk_score updated
                                     (0-100, affects risk_level)
```

When deforestation is detected at a plot:
1. The GeolocationLinker marks the associated producer node
2. The RiskPropagationEngine assigns a high deforestation risk score
3. Risk propagates downstream through the supply chain graph
4. All downstream products inherit the elevated risk
5. The GapAnalyzer flags affected paths for remediation

---

## AGENT-DATA-003: ERP/Finance Connector

**Agent ID:** AGENT-DATA-003
**Purpose:** Tier 1 procurement data from SAP, Oracle, and Workday
**Integration Type:** ERPConnectorProtocol interface

### Data Consumed

| Data Source | Fields | Usage |
|-------------|--------|-------|
| Purchase Orders | Supplier ID, commodity, quantity, date | Tier 1 edge creation |
| Vendor Master | Supplier name, country, address | Tier 1 node creation |
| Goods Receipts | Batch number, quantity, date | Batch traceability |
| Invoice Data | HS code, CN code, transport mode | Edge metadata enrichment |

### Integration Pattern

```python
from greenlang.agents.eudr.supply_chain_mapper.multi_tier_mapper import (
    MultiTierMapper,
    MultiTierMappingInput,
)

# Configure ERP connector
class SAPConnector:
    """Implements ERPConnectorProtocol for SAP S/4HANA."""

    async def get_supplier_records(
        self, operator_id: str, commodity: str
    ) -> List[SupplierRecord]:
        # Query SAP Purchase Order items
        # Filter by EUDR-relevant material groups
        # Map to SupplierRecord format
        ...

# Use in multi-tier mapping
mapper = MultiTierMapper(config)
mapper.set_erp_connector(SAPConnector())

input_data = MultiTierMappingInput(
    operator_id="op-eu-001",
    commodity="cocoa",
    max_depth=5,
)

result = await mapper.discover_tiers(input_data)
# result.tiers_discovered: 4
# result.new_nodes_added: 127
# result.opaque_segments: 3
```

### ERP Data Mapping

```
SAP S/4HANA                              Supply Chain Mapper
===========                              ====================

MM - Material Master                     EUDRCommodity enum
  |-- Material Group  ----------------->   cocoa, coffee, palm_oil, ...
  |-- HS Code         ----------------->   Edge.hs_code

MM - Vendor Master                       SupplyChainNode
  |-- Vendor Number   ----------------->   operator_id
  |-- Vendor Name     ----------------->   operator_name
  |-- Country Key     ----------------->   country_code
  |-- Region          ----------------->   region

MM - Purchase Order                      SupplyChainEdge
  |-- PO Item         ----------------->   edge creation trigger
  |-- Quantity        ----------------->   quantity (Decimal)
  |-- Unit            ----------------->   unit
  |-- Delivery Date   ----------------->   transfer_date

MM - Goods Receipt                       Batch Traceability
  |-- Batch Number    ----------------->   batch_number
  |-- GR Quantity     ----------------->   mass balance check
```

---

## AGENT-DATA-008: Supplier Questionnaire Processor

**Agent ID:** AGENT-DATA-008
**Purpose:** Sub-tier supplier declarations from questionnaire responses
**Integration Type:** QuestionnaireProcessorProtocol interface

### Data Consumed

| Data Source | Fields | Usage |
|-------------|--------|-------|
| Questionnaire Responses | Sub-supplier names, countries, commodities | Tier 2+ node discovery |
| Certification Data | Cert type, cert number, expiry | Node certifications |
| Plot Declarations | GPS coordinates, plot IDs | Producer geolocation |
| Processing Data | Input/output quantities, batch numbers | Mass balance verification |

### Integration Pattern

```python
from greenlang.agents.eudr.supply_chain_mapper.multi_tier_mapper import (
    MultiTierMapper,
)

class QuestionnaireAdapter:
    """Implements QuestionnaireProcessorProtocol."""

    async def get_sub_tier_suppliers(
        self, supplier_id: str
    ) -> List[SupplierRecord]:
        # Query AGENT-DATA-008 for questionnaire responses
        # Parse sub-tier supplier declarations
        # Return normalized SupplierRecord list
        ...

mapper = MultiTierMapper(config)
mapper.set_questionnaire_processor(QuestionnaireAdapter())

# During recursive discovery, the mapper calls:
# sub_suppliers = await questionnaire_adapter.get_sub_tier_suppliers("supp-001")
# For each sub-supplier, a new node is created and the process recurses
```

### Recursive Discovery Flow

```
                    Tier 0 (Importer)
                         |
         ERP Connector   |   (AGENT-DATA-003)
                         v
                    Tier 1 (Direct Suppliers)
                         |
    Questionnaire        |   (AGENT-DATA-008)
    Processor            |
                         v
                    Tier 2 (Sub-Suppliers)
                         |
    Questionnaire  +     |   (AGENT-DATA-008)
    PDF Extractor  |     |   (AGENT-DATA-001)
                   |     v
                    Tier 3 (Sub-Sub-Suppliers)
                         |
                         v
                    Tier N (Producers)
                         |
    Plot Registry        |   (AGENT-DATA-005)
    GIS Connector        |   (AGENT-DATA-006)
    Satellite            |   (AGENT-DATA-007)
                         v
                    Production Plots
                    (GPS + Polygon)
```

---

## GL-EUDR-APP Integration

The Supply Chain Mapper integrates with the GL-EUDR-APP platform through the
`SupplyChainMapperService` facade and the FastAPI router.

### Service Facade

```python
from greenlang.agents.eudr.supply_chain_mapper.setup import (
    SupplyChainMapperService,
    get_service,
    lifespan,
)

# In GL-EUDR-APP main.py:
from fastapi import FastAPI

app = FastAPI(lifespan=lifespan)

# The lifespan context manager handles:
# 1. Load configuration from GL_EUDR_SCM_* env vars
# 2. Connect to PostgreSQL (psycopg_pool)
# 3. Connect to Redis
# 4. Initialize all 9 engines
# 5. Start health check background task
# 6. On shutdown: close all connections

# Mount the API router
from greenlang.agents.eudr.supply_chain_mapper.api.router import get_router

app.include_router(get_router(), prefix="/api")
```

### DDS Generation Integration

The RegulatoryExporter provides supply chain data for the Due Diligence
Statement (DDS) filing:

```python
from greenlang.agents.eudr.supply_chain_mapper import (
    RegulatoryExporter,
    create_exporter,
    OperatorInfo,
    ProductInfo,
    DeclarationInfo,
)

exporter = create_exporter(config)

# Generate DDS supply chain section
dds_result = await exporter.export_dds(
    graph_id="graph-001",
    operator=OperatorInfo(
        name="EU Chocolate GmbH",
        eori="DE123456789",
        country="DE",
    ),
    product=ProductInfo(
        description="Cocoa liquor",
        hs_code="1803.10.00",
        commodity="cocoa",
        quantity="4200",
        unit="kg",
    ),
    declaration=DeclarationInfo(
        declaration_type="initial",
        reference_period_start="2026-01-01",
        reference_period_end="2026-03-31",
    ),
)

# Validate against EU Information System schema
validation = exporter.validate_dds(dds_result.dds_data)
assert validation.is_valid

# Export as JSON for submission
json_output = exporter.serialize_json(dds_result.dds_data)

# Export as XML alternative
xml_output = exporter.serialize_xml(dds_result.dds_data)

# Generate PDF audit package
pdf_result = await exporter.generate_pdf_report(
    graph_id="graph-001",
    include_supply_chain_map=True,
    include_risk_assessment=True,
    include_gap_analysis=True,
)
```

---

## Data Flow Diagrams

### Complete Data Flow: Commodity Import to DDS Filing

```
  Step 1: Data Collection
  =======================
  SAP/Oracle ERP ---> AGENT-DATA-003 ---> Purchase Orders, Vendor Master
  Questionnaires ---> AGENT-DATA-008 ---> Sub-tier Declarations
  PDF Documents  ---> AGENT-DATA-001 ---> Custody Transfer Docs
  CSV/Excel      ---> AGENT-DATA-002 ---> Bulk Supplier Lists

  Step 2: Supply Chain Graph Construction
  =======================================
  MultiTierMapper
    |-- Tier 1: from ERP data (AGENT-DATA-003)
    |-- Tier 2+: from questionnaires (AGENT-DATA-008)
    |-- Tier N: from documents (AGENT-DATA-001)
    |-- Producers: from bulk imports (AGENT-DATA-002)
    v
  SupplyChainGraphEngine
    |-- create_graph()
    |-- add_node() per actor
    |-- add_edge() per custody transfer
    |-- verify_dag() (cycle detection)
    v
  SupplyChainGraph (DAG in memory + PostgreSQL)

  Step 3: Geolocation Enrichment
  ==============================
  PlotRegistryEngine (AGENT-DATA-005) ---> Plot GPS + Polygons
  GIS Connector (AGENT-DATA-006)      ---> Spatial Validation
  Satellite (AGENT-DATA-007)          ---> Deforestation Check
    v
  GeolocationLinker
    |-- link_plot_to_node()
    |-- validate_coordinates()
    |-- check_polygon_requirement()
    |-- cross_reference_protected_areas()
    |-- check_deforestation_status()

  Step 4: Analysis
  ================
  RiskPropagationEngine
    |-- calculate_node_risk() per node
    |-- propagate_downstream() (highest wins)
    v
  GapAnalyzer
    |-- detect_missing_geolocation()
    |-- detect_broken_custody_chain()
    |-- detect_unverified_actors()
    |-- detect_mass_balance_discrepancy()
    |-- calculate_compliance_readiness()
    v
  Compliance Readiness Score (0-100%)

  Step 5: Export and Submission
  ============================
  RegulatoryExporter
    |-- format_dds_supply_chain_section()
    |-- validate_against_schema()
    |-- serialize_json() / serialize_xml()
    |-- generate_pdf_report()
    v
  Due Diligence Statement (DDS)
    |-- Submit to EU Information System
```

### Risk Propagation Data Flow

```
  Input Risk Scores                       Propagation              Output
  =================                       ===========              ======

  Country Risk DB -----> country_risk     |
    (Art. 29 benchmarks)  (weight: 0.30)  |
                                          |
  Commodity Risk ------> commodity_risk   |---> Weighted Sum
    (inherent risk)       (weight: 0.20)  |     per node
                                          |     (0-100 scale)
  Supplier Score ------> supplier_risk    |
    (historical data)     (weight: 0.25)  |
                                          |
  Deforestation -------> deforestation    |
    (AGENT-DATA-007)      (weight: 0.25)  |

                         Propagation Rule:
                         "Highest risk wins" ---> Downstream
                                                  nodes inherit
                                                  max upstream
                                                  risk

  Classification:
    0-30   = LOW risk      (simplified due diligence)
    31-69  = STANDARD risk (standard due diligence)
    70-100 = HIGH risk     (enhanced due diligence)
```

---

## API Call Sequences

### Sequence 1: Create Graph and Run Full Analysis

```
Client                    API                     Engines
  |                        |                         |
  |-- POST /graphs ------->|                         |
  |                        |-- create_graph() ------>|
  |<-- 201 {graph_id} ----|                         |
  |                        |                         |
  |-- POST /discover ----->|                         |
  |                        |-- MultiTierMapper ----->|
  |                        |   discover_tiers()      |
  |<-- 202 {result} ------|                         |
  |                        |                         |
  |-- POST /risk/prop ---->|                         |
  |                        |-- RiskPropagation ----->|
  |                        |   propagate()           |
  |<-- 200 {results} -----|                         |
  |                        |                         |
  |-- POST /gaps/analyze ->|                         |
  |                        |-- GapAnalyzer --------->|
  |                        |   analyze()             |
  |<-- 200 {gaps} --------|                         |
  |                        |                         |
  |-- GET /export -------->|                         |
  |                        |-- RegulatoryExporter -->|
  |                        |   export_dds()          |
  |<-- 200 {dds_data} ----|                         |
```

### Sequence 2: Supplier Onboarding Flow

```
Operator                  API                    Supplier
  |                        |                        |
  |-- POST /invite ------->|                        |
  |                        |-- generate_token() --->|
  |<-- 201 {token, url} --|                        |
  |                        |                        |
  |-- (email invitation) -------------------------------->|
  |                        |                        |
  |                        |<-- GET /onboarding/{token} --|
  |                        |-- 200 {invitation} --->|
  |                        |                        |
  |                        |<-- POST /submit -------|
  |                        |-- add_node_to_graph() -|
  |                        |-- 200 {node_id} ------>|
  |                        |                        |
  |-- GET /graphs/{id} --->|                        |
  |<-- 200 (updated graph)-|                        |
```

### Sequence 3: Backward Trace to Origin Plots

```
Client                    API                     Graph Engine
  |                        |                         |
  |-- GET /trace/backward  |                         |
  |   /{graph_id}/{node_id}|                         |
  |                        |-- BFS backward -------->|
  |                        |   from node_id          |
  |                        |   follow incoming edges  |
  |                        |   collect plot_ids       |
  |                        |   from producer nodes    |
  |                        |                         |
  |<-- 200 TraceResult ----|                         |
  |   visited_nodes: [...]  |                         |
  |   visited_edges: [...]  |                         |
  |   origin_plot_ids: [...] |                         |
  |   trace_depth: 4        |                         |
  |   is_complete: true      |                         |
```

---

## Error Handling

All integrations follow the GreenLang error handling pattern:

1. **Timeout:** If an upstream agent does not respond within the configured
   timeout (`GL_EUDR_SCM_GRAPH_QUERY_TIMEOUT_MS`), the operation is retried
   once and then fails with a clear error message.

2. **Data Quality:** If upstream data fails validation (invalid coordinates,
   missing required fields), the data is logged as a gap (via GapAnalyzer)
   rather than silently dropped.

3. **Partial Failure:** Multi-tier discovery continues even if individual
   tier lookups fail. Failed tiers are flagged as "opaque segments" in the
   mapping result.

4. **Circuit Breaker:** Repeated failures to an upstream agent trigger a
   circuit breaker that prevents cascading failures. The circuit resets
   after a configurable cooldown period.

---

## Authentication Between Agents

All inter-agent communication uses the GreenLang internal service mesh
authentication:

- **Internal calls** (same Kubernetes namespace): mTLS via service mesh
- **Cross-namespace calls:** JWT service tokens with `agent:internal` scope
- **Database access:** PostgreSQL connection pool with role-based credentials
- **Redis access:** Redis AUTH with per-service credentials

The Supply Chain Mapper does not store or forward external user credentials.
All API endpoints authenticate the end user via SEC-001 (JWT) and authorize
via SEC-002 (RBAC) before delegating to internal engines.

---

## Related Documentation

- [README.md](README.md) -- Agent overview and quick start
- [API.md](API.md) -- Complete REST API reference
- [DEPLOYMENT.md](DEPLOYMENT.md) -- Deployment and operations guide
