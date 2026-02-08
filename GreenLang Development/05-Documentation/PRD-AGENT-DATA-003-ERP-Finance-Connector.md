# PRD: AGENT-DATA-003 - ERP/Finance Connector

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-DATA-003 |
| **Agent ID** | GL-DATA-X-004 |
| **Component** | ERP/Finance Connector Agent (Spend Data, Purchase Orders, Inventory, Scope 3 Mapping, Emissions Estimation) |
| **Category** | Data Intake Agent |
| **Priority** | P0 - Critical (primary enterprise data integration for Scope 3 spend-based emissions) |
| **Status** | Layer 1 Complete (~837 lines), Integration Gap-Fill Required |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS requires enterprise financial data from ERP systems for Scope 3
spend-based emissions calculations. Organizations use SAP, Oracle, NetSuite, Dynamics,
Workday and other ERPs as their source of truth for procurement data. Without a
production-grade ERP connector agent:

- **No ERP system integration**: Manual data export from SAP, Oracle, NetSuite, etc.
- **No spend data extraction**: Vendor transactions not automatically captured
- **No purchase order processing**: PO line items not parsed for material analysis
- **No inventory tracking**: Material quantities and values not synchronized
- **No Scope 3 category mapping**: Spend not mapped to GHG Protocol 15 categories
- **No emissions estimation**: Spend-based EEIO calculations not automated
- **No vendor classification**: Vendors not categorized by environmental impact
- **No material mapping**: Materials not linked to emission factors
- **No currency normalization**: Multi-currency spend not standardized to USD
- **No connection management**: ERP credentials and endpoints not centrally managed
- **No audit trail**: Data extraction operations not tracked for compliance

## 3. Existing Implementation

### 3.1 Layer 1: Foundation Agent
**File**: `greenlang/agents/data/erp_connector_agent.py` (~837 lines)
- `ERPConnectorAgent` (BaseAgent subclass, AGENT_ID: GL-DATA-X-004)
- 4 enums: ERPSystem(10: sap_s4hana/sap_ecc/oracle_cloud/oracle_ebs/netsuite/dynamics_365/workday/sage/quickbooks/simulated), Scope3Category(16: 15 GHG Protocol categories + unclassified), TransactionType(6: purchase_order/invoice/goods_receipt/payment/credit_memo/journal_entry), SpendCategory(11: direct_materials/indirect_materials/services/energy/transport/travel/capital_equipment/it_services/professional_services/facilities/other)
- 10 Pydantic models: ERPConnectionConfig, VendorMapping, MaterialMapping, PurchaseOrderLine, PurchaseOrder, SpendRecord, InventoryItem, ERPQueryInput, ERPQueryOutput
- SPEND_TO_SCOPE3_MAPPING: 11 rules mapping SpendCategory to Scope3Category
- DEFAULT_EMISSION_FACTORS: 11 EEIO factors (kgCO2e per USD) by SpendCategory
- Operations: register_connection, register_vendor_mapping, register_material_mapping, query_spend, set_emission_factor
- Scope 3 mapping, emissions calculation, spend/PO/inventory queries
- Simulated data generation for testing
- SHA-256 provenance hashing
- In-memory storage (no database persistence)

### 3.2 Layer 1 Tests
None found.

## 4. Identified Gaps

### Gap 1: No Integration Module
No `greenlang/erp_connector/` package providing a clean SDK.

### Gap 2: No Prometheus Metrics
No `greenlang/erp_connector/metrics.py` following the standard 12-metric pattern.

### Gap 3: No Service Setup Facade
No `configure_erp_connector(app)` / `get_erp_connector(app)` pattern.

### Gap 4: No Real ERP Adapters
Layer 1 only has simulated data; no production ERP system adapters.

### Gap 5: No REST API Router
No `greenlang/erp_connector/api/router.py` with FastAPI endpoints.

### Gap 6: No K8s Deployment Manifests
No `deployment/kubernetes/erp-connector-service/` manifests.

### Gap 7: No Database Migration
No `V033__erp_connector_service.sql` for persistent connection/transaction storage.

### Gap 8: No Monitoring
No Grafana dashboard or alert rules.

### Gap 9: No CI/CD Pipeline
No `.github/workflows/erp-connector-ci.yml`.

### Gap 10: No Operational Runbooks
No `docs/runbooks/` for ERP connector operations.

### Gap 11: No Currency Converter
No multi-currency normalization engine.

### Gap 12: No Batch Sync
No scheduled batch synchronization with ERP systems.

## 5. Architecture (Final State)

### 5.1 Integration Module
```
greenlang/erp_connector/
  __init__.py                  # Public API exports
  config.py                    # ERPConnectorConfig with GL_ERP_CONNECTOR_ env prefix
  models.py                    # Pydantic v2 models (re-export + enhance from Layer 1)
  connection_manager.py        # ConnectionManager: ERP connection lifecycle, health checks
  spend_extractor.py           # SpendExtractor: vendor transactions, AP data, spend analysis
  purchase_order_engine.py     # PurchaseOrderEngine: PO parsing, line items, GR matching
  inventory_tracker.py         # InventoryTracker: material quantities, valuations, movements
  scope3_mapper.py             # Scope3Mapper: GHG Protocol category classification engine
  emissions_calculator.py      # EmissionsCalculator: spend-based EEIO calculations
  currency_converter.py        # CurrencyConverter: multi-currency normalization
  provenance.py                # ProvenanceTracker: SHA-256 hash chain for audit
  metrics.py                   # 12 Prometheus self-monitoring metrics
  setup.py                     # ERPConnectorService facade, configure/get
  api/
    __init__.py
    router.py                  # FastAPI router (20 endpoints)
```

### 5.2 Database Schema (V033)
```sql
CREATE SCHEMA erp_connector_service;
-- erp_connections (connection registry with encrypted credentials)
-- vendor_mappings (vendor to Scope 3 category mappings)
-- material_mappings (material to emission factor mappings)
-- spend_records (hypertable - extracted spend transactions)
-- purchase_orders (PO headers with lifecycle status)
-- purchase_order_lines (PO line items with material details)
-- inventory_snapshots (point-in-time inventory positions)
-- sync_jobs (hypertable - synchronization job tracking)
-- emission_calculations (spend-based emission results)
-- erp_audit_log (hypertable - connector operation audit trail)
```

### 5.3 Prometheus Self-Monitoring Metrics (12)
| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_erp_connections_total` | Counter | Total ERP connections established by system |
| 2 | `gl_erp_sync_duration_seconds` | Histogram | Sync operation latency |
| 3 | `gl_erp_spend_records_total` | Counter | Total spend records extracted |
| 4 | `gl_erp_purchase_orders_total` | Counter | Total purchase orders processed |
| 5 | `gl_erp_scope3_mappings_total` | Counter | Scope 3 classifications by category |
| 6 | `gl_erp_emissions_calculated_total` | Counter | Emissions calculations performed |
| 7 | `gl_erp_sync_errors_total` | Counter | Sync errors by ERP system |
| 8 | `gl_erp_currency_conversions_total` | Counter | Currency conversions performed |
| 9 | `gl_erp_inventory_items_total` | Counter | Inventory items tracked |
| 10 | `gl_erp_batch_syncs_total` | Counter | Batch sync operations by status |
| 11 | `gl_erp_active_connections` | Gauge | Currently active ERP connections |
| 12 | `gl_erp_sync_queue_size` | Gauge | Sync operations waiting in queue |

### 5.4 API Endpoints (20)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/connections` | Register an ERP connection |
| GET | `/v1/connections` | List ERP connections |
| GET | `/v1/connections/{connection_id}` | Get connection details |
| POST | `/v1/connections/{connection_id}/test` | Test ERP connection |
| DELETE | `/v1/connections/{connection_id}` | Remove ERP connection |
| POST | `/v1/spend/sync` | Sync spend data from ERP |
| GET | `/v1/spend` | Query spend records (with filters) |
| GET | `/v1/spend/summary` | Get spend summary by Scope 3 category |
| POST | `/v1/purchase-orders/sync` | Sync purchase orders from ERP |
| GET | `/v1/purchase-orders` | Query purchase orders |
| GET | `/v1/purchase-orders/{po_number}` | Get PO details with line items |
| POST | `/v1/inventory/sync` | Sync inventory from ERP |
| GET | `/v1/inventory` | Query inventory items |
| POST | `/v1/mappings/vendors` | Create/update vendor mapping |
| GET | `/v1/mappings/vendors` | List vendor mappings |
| POST | `/v1/mappings/materials` | Create/update material mapping |
| POST | `/v1/emissions/calculate` | Calculate emissions for spend data |
| GET | `/v1/emissions/summary` | Get emissions summary |
| GET | `/v1/statistics` | Get service statistics |
| GET | `/health` | Service health check |

### 5.5 Key Design Principles
1. **Zero-hallucination**: All category mapping uses explicit rules, NO LLM for classifications
2. **Multi-ERP support**: Pluggable adapters for SAP, Oracle, NetSuite, Dynamics, Workday
3. **GHG Protocol alignment**: Full 15-category Scope 3 classification
4. **EEIO emission factors**: EPA/EXIOBASE spend-based factors with source tracking
5. **Vendor classification**: Hierarchical vendor-to-category mapping with override chains
6. **Material linkage**: Material-level emission factors when available
7. **Multi-currency**: Automatic normalization to reporting currency
8. **Connection security**: Encrypted credential storage, connection health monitoring
9. **Batch synchronization**: Scheduled/on-demand ERP data sync with incremental pull
10. **Complete audit trail**: Every extraction and calculation logged with SHA-256 provenance

## 6. Completion Plan

### Phase 1: Core Integration (Backend Developer)
1. Create `greenlang/erp_connector/__init__.py` - Public API exports
2. Create `greenlang/erp_connector/config.py` - ERPConnectorConfig with GL_ERP_CONNECTOR_ env prefix
3. Create `greenlang/erp_connector/models.py` - Pydantic v2 models (re-export + enhance Layer 1)
4. Create `greenlang/erp_connector/connection_manager.py` - ConnectionManager with health checks
5. Create `greenlang/erp_connector/spend_extractor.py` - SpendExtractor with vendor analysis
6. Create `greenlang/erp_connector/purchase_order_engine.py` - PO processing engine
7. Create `greenlang/erp_connector/inventory_tracker.py` - Inventory sync and tracking
8. Create `greenlang/erp_connector/scope3_mapper.py` - GHG Protocol Scope 3 classifier
9. Create `greenlang/erp_connector/emissions_calculator.py` - Spend-based EEIO calculator
10. Create `greenlang/erp_connector/currency_converter.py` - Multi-currency normalizer
11. Create `greenlang/erp_connector/provenance.py` - ProvenanceTracker
12. Create `greenlang/erp_connector/metrics.py` - 12 Prometheus metrics
13. Create `greenlang/erp_connector/api/router.py` - FastAPI router with 20 endpoints
14. Create `greenlang/erp_connector/setup.py` - ERPConnectorService facade

### Phase 2: Infrastructure (DevOps Engineer)
1. Create `deployment/database/migrations/sql/V033__erp_connector_service.sql`
2. Create K8s manifests in `deployment/kubernetes/erp-connector-service/`
3. Create monitoring dashboards and alerts
4. Create CI/CD pipeline
5. Create operational runbooks

### Phase 3: Tests (Test Engineer)
1-16. Create unit, integration, and load tests (550+ tests target)

## 7. Success Criteria
- Integration module provides clean SDK for all ERP data operations
- All 12 Prometheus metrics instrumented
- Standard GreenLang deployment pattern (K8s, monitoring, CI/CD)
- V033 database migration for persistent ERP data storage
- 20 REST API endpoints operational
- 550+ tests passing
- Multi-ERP connection management with health monitoring
- Scope 3 category mapping with >95% accuracy on classified vendors
- Spend-based emissions calculation using EEIO factors
- Multi-currency normalization
- Complete audit trail for every data extraction

## 8. Integration Points

### 8.1 Upstream Dependencies
- **AGENT-FOUND-003 Unit Normalizer**: Normalize extracted quantities
- **AGENT-FOUND-005 Citations**: Track emission factor sources
- **AGENT-FOUND-006 Access Guard**: Authorization for ERP data access
- **AGENT-FOUND-010 Observability**: Metrics, tracing, logging
- **AGENT-DATA-001 PDF Extractor**: Process invoice PDFs from ERP

### 8.2 Downstream Consumers
- **Scope 3 Calculation Agents**: Consume classified spend data
- **Supply Chain Agents**: Consume vendor and material data
- **Compliance Agents**: Consume procurement records
- **CSRD/CBAM Reporting**: Structured Scope 3 emissions data
- **Admin Dashboard**: ERP sync status visualization

### 8.3 Infrastructure Integration
- **PostgreSQL**: Persistent connection, transaction, mapping storage (V033 migration)
- **Redis**: Connection pool caching, sync queue, emission factor cache
- **S3**: Raw ERP data export storage
- **Prometheus**: 12 self-monitoring metrics
- **Grafana**: ERP connector service dashboard
- **Alertmanager**: 15 alert rules
- **K8s**: Standard deployment with HPA
