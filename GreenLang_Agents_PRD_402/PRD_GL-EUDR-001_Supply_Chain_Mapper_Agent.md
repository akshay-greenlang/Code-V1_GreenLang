# PRD: Supply Chain Mapper Agent (GL-EUDR-001)

**Agent family:** EUDRTraceabilityFamily
**Layer:** Supply Chain Traceability
**Primary domains:** Supply chain mapping, network analysis, supplier relationships
**Priority:** P0 (highest)
**Doc version:** 2.0
**Last updated:** 2026-01-30 (Asia/Kolkata)
**Status:** APPROVED FOR DEVELOPMENT
**Owner:** EUDR Platform Team

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-30 | GreenLang EUDR Team | Initial PRD |
| 2.0 | 2026-01-30 | GreenLang EUDR Team | Merged stakeholder decisions, added technical designs |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals and Non-Goals](#3-goals-and-non-goals)
4. [Stakeholders and Users](#4-stakeholders-and-users)
5. [Definitions and Glossary](#5-definitions-and-glossary)
6. [Regulatory Requirements](#6-regulatory-requirements)
7. [Architecture Decisions](#7-architecture-decisions)
8. [Data Model](#8-data-model)
9. [Functional Requirements](#9-functional-requirements)
10. [Non-Functional Requirements](#10-non-functional-requirements)
11. [API Specification](#11-api-specification)
12. [Technical Design: Entity Resolution](#12-technical-design-entity-resolution)
13. [Technical Design: LLM Integration](#13-technical-design-llm-integration)
14. [Algorithms](#14-algorithms)
15. [UI/UX Specification](#15-uiux-specification)
16. [Performance & Scale Strategy](#16-performance--scale-strategy)
17. [Access Control](#17-access-control)
18. [Agent Orchestration](#18-agent-orchestration)
19. [Notification System](#19-notification-system)
20. [Testing Strategy](#20-testing-strategy)
21. [Rollout Plan](#21-rollout-plan)
22. [Success Metrics](#22-success-metrics)
23. [Dependencies and Integration](#23-dependencies-and-integration)
24. [Risk Mitigation](#24-risk-mitigation)

---

## 1. Executive Summary

**Supply Chain Mapper Agent (GL-EUDR-001)** is the foundational agent in the EUDR Supply Chain Traceability pipeline. It maps the complete supply chain network from raw material origin (farm/forest plots) through all intermediaries to the final EU importer. This agent creates a comprehensive graph representation of supplier relationships, material flows, and geographic origins essential for EUDR compliance.

### Why This Agent is Critical

- EUDR requires complete traceability from production plot to EU market entry
- Multi-tier supply chains (often 5-10 tiers) must be fully mapped
- 7 commodity categories each have unique supply chain structures
- ~100,000 EU operators need supply chain visibility by December 2025

### Key Architecture Decisions (Stakeholder Approved)

| Decision | Choice |
|----------|--------|
| Graph Storage | PostgreSQL primary, Neo4j as read-optimized cache |
| Entity Resolution | Hybrid rules + ML (precision-first) |
| Coverage Thresholds | Risk-weighted with DDS hard gates |
| Multi-Tier Suppliers | Role-based edges with tier context |
| Visualization | All 4 paradigms (graph, Sankey, table, map) |
| LLM Integration | Entity extraction, fuzzy matching, NL queries |
| Deployment | Hybrid K8s API + dedicated graph workers |
| Timeline | 12 weeks |

---

## 2. Problem Statement

EU Deforestation Regulation (2023/1115) mandates that operators prove their commodities are:
1. **Deforestation-free** (no deforestation after December 31, 2020)
2. **Legally produced** (compliant with all origin country laws)

Without a dedicated Supply Chain Mapper:
- Companies cannot identify all suppliers in complex multi-tier chains
- Material origins remain obscured by intermediaries
- Due diligence becomes impossible without visibility
- Risk assessment lacks the foundation of supply chain knowledge

### Primary Risk: Data Availability

Stakeholder interviews identified **data availability** as the top implementation risk. Suppliers may refuse to disclose upstream relationships. Mitigation strategies (prioritized):
1. **Contractual requirements** - Disclosure obligations in supplier contracts
2. **Supplier incentives** - Faster payments, preferred status for data sharing
3. **Alternative data sources** - Customs, shipping manifests, certifications
4. **Industry collaboration** - Pre-competitive data sharing consortiums

---

## 3. Goals and Non-Goals

### 3.1 Goals (must deliver)

1. **Multi-tier supply chain mapping**
   - Map all supplier tiers from plot of origin to EU border
   - Support 7 EUDR commodity categories
   - Handle complex transformations (e.g., cocoa beans → chocolate)

2. **Network graph construction**
   - Build directed acyclic graphs (DAG) of material flows
   - Identify all nodes (suppliers, processors, traders)
   - Track edges (transactions, transformations)
   - Handle suppliers appearing at multiple tiers (role-based edges)

3. **Origin plot identification**
   - Link final products to specific production plots
   - Support polygon and point geolocation data
   - Validate plot coordinates against commodity type
   - Support cooperative/smallholder aggregation models

4. **Temporal tracking**
   - Track supply chain changes over time
   - Maintain historical supplier relationships
   - Version supply chain snapshots for audit
   - Support as-of queries and snapshot diffs

5. **Entity resolution**
   - Deduplicate suppliers across data sources
   - Hybrid rules + ML approach for >95% accuracy
   - Handle suppliers without tax IDs

### 3.2 Non-Goals (explicitly out of scope)

- Deforestation risk assessment (GL-EUDR-015)
- Document verification (GL-EUDR-030 series)
- DDS report generation (GL-EUDR-040 series)
- Direct ERP integration (handled by GL-DATA-X agents)
- What-if scenario modeling (v2)
- Node-level ACLs (v2)

---

## 4. Stakeholders and Users

### 4.1 Primary Stakeholders
- **EUDR Compliance Officers:** Primary users needing supply chain visibility
- **Procurement Teams:** Manage supplier relationships
- **Sustainability Officers:** Ensure ethical sourcing
- **Legal/Risk Teams:** Due diligence requirements
- **External Auditors:** Verification and certification

### 4.2 Primary User Personas

| Persona | Primary Task | Key Views |
|---------|-------------|-----------|
| Supply Chain Analyst | Maps and validates supplier networks | Graph canvas, Table |
| Compliance Manager | Ensures complete traceability coverage | Coverage dashboard, Alerts |
| Procurement Officer | Registers new suppliers with plot data | Forms, Map view |
| Field Auditor | Validates supply chain on-site | Offline PWA, Map view |

---

## 5. Definitions and Glossary

| Term | Definition |
|---|---|
| **Tier** | Level in supply chain (Tier 1 = direct supplier, Tier 2 = supplier's supplier, etc.) |
| **Plot of Origin** | Specific geographic location where commodity was produced (farm, forest, plantation) |
| **Chain of Custody** | Documentation trail proving material flow from origin to destination |
| **Mass Balance** | Accounting system tracking volumes through transformations |
| **Commodity Category** | One of 7 EUDR categories: cattle, cocoa, coffee, palm oil, rubber, soy, wood |
| **Transformation** | Process changing commodity form (e.g., logs → lumber, beans → powder) |
| **Node** | Entity in supply chain graph (producer, processor, trader, importer) |
| **Edge** | Relationship/transaction between nodes |
| **Entity Resolution** | Process of deduplicating and merging supplier records from multiple sources |
| **Golden Record** | Single authoritative record created by merging data from multiple sources |
| **Snapshot** | Immutable point-in-time capture of supply chain graph state |

---

## 6. Regulatory Requirements (EUDR 2023/1115)

### 6.1 Article 9 - Due Diligence Requirements
- Collection of information about commodities
- Traceability to plot of production
- Geolocation of all plots of land

### 6.2 Article 10 - Information Requirements
- Country of production
- Geolocation coordinates
- Quantity and description
- Names and addresses of suppliers

### 6.3 Covered Commodities

| Category | Example Products | Typical Supply Chain Depth |
|---|---|---|
| Cattle | Beef, leather, gelatin | 3-5 tiers |
| Cocoa | Chocolate, cocoa butter | 4-7 tiers |
| Coffee | Roasted, instant | 4-6 tiers |
| Palm Oil | Food products, biofuels | 5-8 tiers |
| Rubber | Tires, industrial | 4-6 tiers |
| Soy | Oil, meal, protein | 3-5 tiers |
| Wood | Furniture, paper, pulp | 4-7 tiers |

### 6.4 Deadline Handling

| Operator Type | Deadline | System Handling |
|---------------|----------|-----------------|
| Large Operators | December 30, 2024 | Full feature requirements |
| SMEs | June 30, 2025 | Same system, deadline-aware UI |

Implementation: Tag each importer with `operator_size: LARGE | SME` to enforce appropriate gates and display deadline countdowns.

---

## 7. Architecture Decisions

### 7.1 Graph Storage Strategy

| Aspect | Decision |
|--------|----------|
| **Primary Store** | PostgreSQL (authoritative, ACID-compliant) |
| **Graph Cache** | Neo4j (read-optimized for traversals) |
| **Sync Direction** | Write to PostgreSQL → sync to Neo4j |
| **Rationale** | ACID guarantees, standard tooling, easier backup/recovery |

### 7.2 Entity Resolution Strategy

| Aspect | Decision |
|--------|----------|
| **Approach** | Hybrid rules + ML (precision-first) |
| **Auto-Merge Threshold** | ≥0.98 with strong feature match |
| **Review Threshold** | 0.85-0.98 → human review queue |
| **No Merge** | <0.85 score |
| **Strong Features** | Tax ID, DUNS, EORI, exact address+country |

### 7.3 Multi-Tier Supplier Handling

| Aspect | Decision |
|--------|----------|
| **Node Model** | Single canonical node per deduplicated entity |
| **Edge Model** | Multiple edges with role/tier context metadata |
| **Tier Storage** | `node.tier` = tier_min; `node.metadata.tiers` = all appearances |

### 7.4 Coverage Thresholds

| Gate | Standard Risk | High Risk |
|------|--------------|-----------|
| Risk Assessment | ≥95% mapping, ≥90% plot | ≥98% mapping, ≥95% plot |
| DDS Submission | No HIGH severity gaps | No HIGH severity gaps |

### 7.5 Cycle Detection

| Aspect | Decision |
|--------|----------|
| **Primary Resolution** | Temporal (use transaction_date) |
| **Fallback** | Keep higher confidence_score edge |
| **Always** | Emit CYCLE_DETECTED gap for review |

### 7.6 Data Refusal Handling

| Scenario | Decision |
|----------|----------|
| Supplier refuses Tier 2+ disclosure | Partial acceptance |
| Implementation | Flag `disclosure_status: PARTIAL`, exclude from DDS claims |

### 7.7 Cooperative Modeling

| Aspect | Decision |
|--------|----------|
| **Farmers** | Individual PRODUCER nodes with own plots |
| **Cooperative** | AGGREGATOR/Trader node |
| **Relationship** | Farmer -(AGGREGATES)→ Cooperative -(SUPPLIES)→ Next Tier |
| **Shared Plots** | v1.1: Add `plot_producers` join table |

---

## 8. Data Model

### 8.1 Core Entities (PostgreSQL)

```sql
-- Supply Chain Nodes
CREATE TABLE supply_chain_nodes (
    node_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_type VARCHAR(50) NOT NULL,
    name VARCHAR(500) NOT NULL,
    country_code CHAR(2) NOT NULL,
    address JSONB,
    tax_id VARCHAR(100),
    duns_number VARCHAR(9),
    eori_number VARCHAR(17),
    certifications JSONB DEFAULT '[]',
    commodities TEXT[] NOT NULL,

    -- Tier information (multi-tier support)
    tier INTEGER,  -- tier_min (shortest path to importer)
    tier_max INTEGER,  -- furthest tier appearance
    all_tiers INTEGER[],  -- all tier positions [1, 3, 5]

    -- Verification & Risk
    risk_score DECIMAL(3,2),
    verification_status VARCHAR(50) DEFAULT 'UNVERIFIED',
    disclosure_status VARCHAR(50) DEFAULT 'FULL',  -- FULL, PARTIAL, NONE

    -- Operator classification (EUDR deadline handling)
    operator_size VARCHAR(10),  -- LARGE, SME

    -- Golden record tracking
    field_sources JSONB DEFAULT '{}',  -- {"tax_id": "SAP", "address": "QUESTIONNAIRE"}

    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_node_type CHECK (
        node_type IN ('PRODUCER', 'PROCESSOR', 'TRADER', 'IMPORTER', 'AGGREGATOR')
    ),
    CONSTRAINT valid_disclosure CHECK (
        disclosure_status IN ('FULL', 'PARTIAL', 'NONE')
    )
);

-- Supply Chain Edges (ENHANCED with provenance)
CREATE TABLE supply_chain_edges (
    edge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_node_id UUID REFERENCES supply_chain_nodes(node_id),
    target_node_id UUID REFERENCES supply_chain_nodes(node_id),
    edge_type VARCHAR(50) NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    quantity DECIMAL(15,3),
    quantity_unit VARCHAR(20),
    transaction_date DATE,
    documents JSONB DEFAULT '[]',
    verified BOOLEAN DEFAULT FALSE,
    confidence_score DECIMAL(3,2) DEFAULT 0.5,

    -- NEW: Data provenance tracking
    data_source VARCHAR(50) DEFAULT 'SUPPLIER_DECLARED',
    inference_method VARCHAR(100),
    inference_evidence JSONB DEFAULT '[]',

    -- NEW: Multi-tier context
    edge_context JSONB DEFAULT '{}',  -- {is_direct_to_importer, observed_tier, relationship_path}

    -- NEW: DDS eligibility
    dds_eligible BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_edge_type CHECK (
        edge_type IN ('SUPPLIES', 'PROCESSES', 'TRADES', 'IMPORTS', 'AGGREGATES')
    ),
    CONSTRAINT valid_commodity CHECK (
        commodity IN ('CATTLE', 'COCOA', 'COFFEE', 'PALM_OIL', 'RUBBER', 'SOY', 'WOOD')
    ),
    CONSTRAINT valid_data_source CHECK (
        data_source IN (
            'SUPPLIER_DECLARED',
            'INFERRED_CUSTOMS',
            'INFERRED_SHIPPING',
            'INFERRED_CERTIFICATION',
            'INFERRED_SATELLITE',
            'THIRD_PARTY_DATA'
        )
    )
);

-- Plot References (origin plots)
CREATE TABLE origin_plots (
    plot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    producer_node_id UUID REFERENCES supply_chain_nodes(node_id),
    plot_identifier VARCHAR(255),
    geometry GEOMETRY(GEOMETRY, 4326) NOT NULL,
    area_hectares DECIMAL(10,2),
    commodity VARCHAR(50) NOT NULL,
    country_code CHAR(2) NOT NULL,
    validation_status VARCHAR(50) DEFAULT 'PENDING',
    deforestation_risk_score DECIMAL(3,2),
    last_assessment_date DATE,
    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_geometry CHECK (ST_IsValid(geometry))
);

-- NEW: Shared plot support for cooperatives (v1.1)
CREATE TABLE plot_producers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID REFERENCES origin_plots(plot_id),
    producer_node_id UUID REFERENCES supply_chain_nodes(node_id),
    share_percentage DECIMAL(5,2),
    tenure_type VARCHAR(50),  -- OWNER, LEASE, COMMUNITY
    valid_from DATE,
    valid_to DATE,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(plot_id, producer_node_id, valid_from)
);

-- Supply Chain Snapshots (versioning)
CREATE TABLE supply_chain_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_date TIMESTAMP DEFAULT NOW(),
    importer_node_id UUID REFERENCES supply_chain_nodes(node_id),
    commodity VARCHAR(50) NOT NULL,
    graph_hash VARCHAR(64) NOT NULL,
    node_count INTEGER,
    edge_count INTEGER,
    plot_count INTEGER,
    coverage_percentage DECIMAL(5,2),
    mapping_completeness DECIMAL(5,2),
    plot_coverage DECIMAL(5,2),
    snapshot_data JSONB NOT NULL,

    -- NEW: Snapshot trigger
    trigger_type VARCHAR(50),  -- SCHEDULED, DDS_SUBMISSION, COVERAGE_DROP, MANUAL

    created_by VARCHAR(100),

    UNIQUE(importer_node_id, commodity, snapshot_date)
);

-- NEW: Entity Resolution Candidates
CREATE TABLE entity_resolution_candidates (
    candidate_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_a_id UUID REFERENCES supply_chain_nodes(node_id),
    node_b_id UUID REFERENCES supply_chain_nodes(node_id),
    similarity_score DECIMAL(3,2) NOT NULL,
    matching_features JSONB NOT NULL,  -- ["tax_id_match", "address_similar"]
    resolution_status VARCHAR(50) DEFAULT 'PENDING',  -- PENDING, AUTO_MERGED, REVIEWED_MERGE, REVIEWED_NO_MERGE
    resolved_by VARCHAR(100),
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(node_a_id, node_b_id)
);

-- NEW: Supply chain gaps tracking
CREATE TABLE supply_chain_gaps (
    gap_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID REFERENCES supply_chain_snapshots(snapshot_id),
    node_id UUID REFERENCES supply_chain_nodes(node_id),
    gap_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    description TEXT,
    remediation_suggestion TEXT,
    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_gap_type CHECK (
        gap_type IN (
            'UNVERIFIED_SUPPLIER', 'MISSING_PLOT_DATA', 'MISSING_COORDINATES',
            'PARTIAL_DISCLOSURE', 'EXPIRED_CERTIFICATION', 'CYCLE_DETECTED',
            'MISSING_TIER_DATA', 'UNVERIFIED_TRANSFORMATION'
        )
    ),
    CONSTRAINT valid_severity CHECK (
        severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    )
);

-- Indexes
CREATE INDEX idx_nodes_type ON supply_chain_nodes(node_type);
CREATE INDEX idx_nodes_country ON supply_chain_nodes(country_code);
CREATE INDEX idx_nodes_commodity ON supply_chain_nodes USING GIN(commodities);
CREATE INDEX idx_nodes_verification ON supply_chain_nodes(verification_status);
CREATE INDEX idx_nodes_disclosure ON supply_chain_nodes(disclosure_status);
CREATE INDEX idx_edges_source ON supply_chain_edges(source_node_id);
CREATE INDEX idx_edges_target ON supply_chain_edges(target_node_id);
CREATE INDEX idx_edges_commodity ON supply_chain_edges(commodity);
CREATE INDEX idx_edges_data_source ON supply_chain_edges(data_source);
CREATE INDEX idx_edges_dds_eligible ON supply_chain_edges(dds_eligible);
CREATE INDEX idx_plots_geometry ON origin_plots USING GIST(geometry);
CREATE INDEX idx_plots_producer ON origin_plots(producer_node_id);
CREATE INDEX idx_snapshots_importer ON supply_chain_snapshots(importer_node_id);
CREATE INDEX idx_snapshots_date ON supply_chain_snapshots(snapshot_date);
CREATE INDEX idx_er_candidates_status ON entity_resolution_candidates(resolution_status);
CREATE INDEX idx_gaps_snapshot ON supply_chain_gaps(snapshot_id);
CREATE INDEX idx_gaps_severity ON supply_chain_gaps(severity);
```

### 8.2 Graph Schema (Neo4j - Read Cache)

```cypher
// Node creation with multi-tier support
CREATE (p:Producer {
    node_id: $node_id,
    name: $name,
    country_code: $country_code,
    tier: $tier_min,
    all_tiers: $all_tiers,
    verification_status: $verification_status
})

// Relationship with provenance
MATCH (source:Producer {node_id: $source_id})
MATCH (target:Processor {node_id: $target_id})
CREATE (source)-[:SUPPLIES {
    edge_id: $edge_id,
    commodity: $commodity,
    quantity: $quantity,
    date: $date,
    verified: $verified,
    confidence_score: $confidence_score,
    data_source: $data_source,
    dds_eligible: $dds_eligible
}]->(target)

// Tier calculation query (handles multi-tier appearances)
MATCH path = (n)-[*]->(importer:Importer {node_id: $importer_id})
WITH n, collect(DISTINCT length(path)) as tiers
SET n.tier = min(tiers),
    n.tier_max = max(tiers),
    n.all_tiers = tiers

// Cycle detection
MATCH (a)-[r1]->(b)-[r2]->(a)
RETURN a, b, r1, r2
```

---

## 9. Functional Requirements

### 9.1 Supply Chain Discovery
- **FR-001 (P0):** Ingest supplier data from ERP systems and manual uploads
- **FR-002 (P0):** Auto-discover multi-tier relationships from transaction data
- **FR-003 (P0):** Support all 7 EUDR commodity categories
- **FR-004 (P0):** Deduplicate suppliers using hybrid entity resolution (rules + ML)
- **FR-005 (P0):** Merge data from multiple sources using field-level golden record strategy
- **FR-006 (P0):** Track data provenance for all edges (source, inference method)

### 9.2 Graph Construction
- **FR-010 (P0):** Build directed graph of supplier relationships
- **FR-011 (P0):** Calculate tier levels based on distance from origin
- **FR-012 (P0):** Track material flow quantities through chain
- **FR-013 (P0):** Handle commodity transformations (input → output mappings)
- **FR-014 (P0):** Support suppliers appearing at multiple tiers (role-based edges)
- **FR-015 (P0):** Detect and resolve cycles with temporal/confidence fallback
- **FR-016 (P1):** Support mass balance calculations through processing

### 9.3 Plot Linking
- **FR-020 (P0):** Associate producers with origin plots (geolocation)
- **FR-021 (P0):** Validate plot coordinates (WGS-84, 6 decimal precision)
- **FR-022 (P0):** Support both point and polygon geometries
- **FR-023 (P0):** Model cooperative/smallholder aggregation (individual farmers + AGGREGATES edges)
- **FR-024 (P1):** Detect duplicate/overlapping plots
- **FR-025 (P1):** Support shared plots via plot_producers join table (v1.1)

### 9.4 Coverage Analysis
- **FR-030 (P0):** Calculate traceability coverage percentage (risk-weighted)
- **FR-031 (P0):** Identify and classify gaps by severity (CRITICAL, HIGH, MEDIUM, LOW)
- **FR-032 (P0):** Generate coverage reports per commodity
- **FR-033 (P0):** Enforce coverage gates (95%/90% for risk assessment, no HIGH for DDS)
- **FR-034 (P1):** Alert when coverage falls below threshold
- **FR-035 (P1):** Prioritize gap closure by volume/risk

### 9.5 Versioning and Audit
- **FR-040 (P0):** Version supply chain snapshots (immutable)
- **FR-041 (P0):** Track all changes with timestamps
- **FR-042 (P0):** Enable point-in-time queries via as-of API
- **FR-043 (P0):** Generate diff reports between snapshots
- **FR-044 (P0):** Auto-create snapshots on key events (DDS submission, coverage drop)
- **FR-045 (P1):** Export audit-ready supply chain documentation

### 9.6 Data Refusal Handling
- **FR-050 (P0):** Accept partial disclosure from suppliers (Tier 1 only)
- **FR-051 (P0):** Mark partially disclosed suppliers appropriately
- **FR-052 (P0):** Exclude partial disclosure volumes from DDS-eligible claims

---

## 10. Non-Functional Requirements

### 10.1 Performance
- **NFR-001 (P0):** Process 10,000 supplier records in < 5 minutes
- **NFR-002 (P0):** Graph queries return in < 2 seconds
- **NFR-003 (P0):** Support supply chains with 100,000+ nodes
- **NFR-004 (P0):** Incremental updates without full rebuild
- **NFR-005 (P1):** Progressive loading for UI (2-3 tiers initially)

### 10.2 Accuracy
- **NFR-010 (P0):** Entity resolution accuracy > 95%
- **NFR-011 (P0):** Zero false negatives on direct supplier links
- **NFR-012 (P0):** Coordinate validation with zero tolerance for invalid WGS-84

### 10.3 Reliability
- **NFR-020 (P0):** 99.9% service availability
- **NFR-021 (P0):** Automatic recovery from transient failures
- **NFR-022 (P1):** Graceful degradation under load
- **NFR-023 (P1):** Time-boxed queries with partial results

### 10.4 Security
- **NFR-030 (P0):** Encrypt supplier data at rest and in transit
- **NFR-031 (P0):** Role-based access control (Org + Commodity + Tier levels)
- **NFR-032 (P0):** Audit logging for all data access

### 10.5 Internationalization
- **NFR-040 (P1):** Multi-language UI support (EN, ES, PT, FR, ID, VI)
- **NFR-041 (P1):** Offline PWA capability for field auditors

---

## 11. API Specification

### 11.1 REST Endpoints

```yaml
openapi: 3.0.0
info:
  title: GL-EUDR-001 Supply Chain Mapper API
  version: 2.0.0
  description: Supply chain mapping and traceability API for EUDR compliance

servers:
  - url: /api/v1

paths:
  # ==================== NODES ====================
  /supply-chain/nodes:
    post:
      summary: Create supply chain node
      tags: [Nodes]
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateNodeRequest'
      responses:
        201:
          description: Node created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SupplyChainNode'

    get:
      summary: List supply chain nodes
      tags: [Nodes]
      parameters:
        - name: commodity
          in: query
          schema:
            type: string
            enum: [CATTLE, COCOA, COFFEE, PALM_OIL, RUBBER, SOY, WOOD]
        - name: tier
          in: query
          schema:
            type: integer
        - name: country
          in: query
          schema:
            type: string
        - name: verification_status
          in: query
          schema:
            type: string
            enum: [VERIFIED, UNVERIFIED, PENDING]
        - name: disclosure_status
          in: query
          schema:
            type: string
            enum: [FULL, PARTIAL, NONE]

  /supply-chain/nodes/{node_id}:
    get:
      summary: Get node details
      tags: [Nodes]
    patch:
      summary: Update node
      tags: [Nodes]
    delete:
      summary: Delete node
      tags: [Nodes]

  # ==================== EDGES ====================
  /supply-chain/edges:
    post:
      summary: Create supply chain edge
      tags: [Edges]
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateEdgeRequest'

    get:
      summary: List edges
      tags: [Edges]
      parameters:
        - name: data_source
          in: query
          schema:
            type: string
            enum: [SUPPLIER_DECLARED, INFERRED_CUSTOMS, INFERRED_SHIPPING, INFERRED_CERTIFICATION, INFERRED_SATELLITE, THIRD_PARTY_DATA]
        - name: dds_eligible
          in: query
          schema:
            type: boolean

  # ==================== GRAPH ====================
  /supply-chain/graph:
    get:
      summary: Get full supply chain graph
      tags: [Graph]
      parameters:
        - name: importer_id
          in: query
          required: true
          schema:
            type: string
            format: uuid
        - name: commodity
          in: query
          required: true
          schema:
            type: string
        - name: depth
          in: query
          schema:
            type: integer
            default: 10
        - name: include_inferred
          in: query
          schema:
            type: boolean
            default: true
          description: Include edges from inferred data sources
      responses:
        200:
          description: Supply chain graph
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SupplyChainGraph'

  # ==================== COVERAGE ====================
  /supply-chain/coverage:
    get:
      summary: Calculate traceability coverage
      tags: [Coverage]
      parameters:
        - name: importer_id
          in: query
          required: true
          schema:
            type: string
            format: uuid
        - name: commodity
          in: query
          required: true
          schema:
            type: string
        - name: risk_level
          in: query
          schema:
            type: string
            enum: [LOW, STANDARD, HIGH]
            default: STANDARD
      responses:
        200:
          description: Coverage analysis
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CoverageReport'

  /supply-chain/coverage/gates:
    get:
      summary: Check coverage gates (risk assessment, DDS submission)
      tags: [Coverage]
      parameters:
        - name: importer_id
          in: query
          required: true
        - name: commodity
          in: query
          required: true
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CoverageGateResult'

  # ==================== SNAPSHOTS (ENHANCED) ====================
  /supply-chain/snapshots:
    post:
      summary: Create supply chain snapshot
      tags: [Snapshots]
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required: [importer_id, commodity]
              properties:
                importer_id:
                  type: string
                  format: uuid
                commodity:
                  type: string
                trigger_type:
                  type: string
                  enum: [MANUAL, SCHEDULED, DDS_SUBMISSION, COVERAGE_DROP]
                  default: MANUAL
      responses:
        201:
          description: Snapshot created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Snapshot'

    get:
      summary: Query snapshots with as-of support
      tags: [Snapshots]
      parameters:
        - name: importer_id
          in: query
          required: true
          schema:
            type: string
            format: uuid
        - name: commodity
          in: query
          required: true
          schema:
            type: string
        - name: as_of
          in: query
          schema:
            type: string
            format: date-time
          description: Return latest snapshot before this timestamp
        - name: policy
          in: query
          schema:
            type: string
            enum: [latest_before, closest, exact]
            default: latest_before
      responses:
        200:
          description: Snapshot list or single snapshot if as_of specified
          content:
            application/json:
              schema:
                oneOf:
                  - type: array
                    items:
                      $ref: '#/components/schemas/SnapshotSummary'
                  - $ref: '#/components/schemas/Snapshot'

  /supply-chain/snapshots/{snapshot_id}:
    get:
      summary: Get snapshot details
      tags: [Snapshots]
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Snapshot'

  /supply-chain/snapshots/{snapshot_id}/diff/{compare_snapshot_id}:
    get:
      summary: Get diff between two snapshots
      tags: [Snapshots]
      responses:
        200:
          description: Snapshot diff
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SnapshotDiff'

  # ==================== ENTITY RESOLUTION ====================
  /supply-chain/entity-resolution/candidates:
    get:
      summary: List entity resolution candidates
      tags: [Entity Resolution]
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [PENDING, AUTO_MERGED, REVIEWED_MERGE, REVIEWED_NO_MERGE]
        - name: min_score
          in: query
          schema:
            type: number
            minimum: 0
            maximum: 1
      responses:
        200:
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/EntityResolutionCandidate'

  /supply-chain/entity-resolution/candidates/{candidate_id}/resolve:
    post:
      summary: Resolve entity resolution candidate
      tags: [Entity Resolution]
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required: [decision]
              properties:
                decision:
                  type: string
                  enum: [MERGE, NO_MERGE]
                reason:
                  type: string

  /supply-chain/entity-resolution/run:
    post:
      summary: Trigger entity resolution batch run
      tags: [Entity Resolution]
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                scope:
                  type: string
                  enum: [ALL, NEW_NODES, SPECIFIC_NODES]
                node_ids:
                  type: array
                  items:
                    type: string
                    format: uuid

  # ==================== NATURAL LANGUAGE QUERIES ====================
  /supply-chain/query/natural-language:
    post:
      summary: Query supply chain using natural language
      tags: [LLM]
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required: [query, importer_id, commodity]
              properties:
                query:
                  type: string
                  example: "Show me all palm oil suppliers in Indonesia with expired certifications"
                importer_id:
                  type: string
                  format: uuid
                commodity:
                  type: string
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/NLQueryResult'

components:
  schemas:
    CreateNodeRequest:
      type: object
      required: [node_type, name, country_code, commodities]
      properties:
        node_type:
          type: string
          enum: [PRODUCER, PROCESSOR, TRADER, IMPORTER, AGGREGATOR]
        name:
          type: string
        country_code:
          type: string
          pattern: '^[A-Z]{2}$'
        commodities:
          type: array
          items:
            type: string
        address:
          type: object
        tax_id:
          type: string
        duns_number:
          type: string
        eori_number:
          type: string
        operator_size:
          type: string
          enum: [LARGE, SME]
        disclosure_status:
          type: string
          enum: [FULL, PARTIAL, NONE]
          default: FULL

    CreateEdgeRequest:
      type: object
      required: [source_node_id, target_node_id, edge_type, commodity]
      properties:
        source_node_id:
          type: string
          format: uuid
        target_node_id:
          type: string
          format: uuid
        edge_type:
          type: string
          enum: [SUPPLIES, PROCESSES, TRADES, IMPORTS, AGGREGATES]
        commodity:
          type: string
        quantity:
          type: number
        quantity_unit:
          type: string
        transaction_date:
          type: string
          format: date
        data_source:
          type: string
          enum: [SUPPLIER_DECLARED, INFERRED_CUSTOMS, INFERRED_SHIPPING, INFERRED_CERTIFICATION, INFERRED_SATELLITE, THIRD_PARTY_DATA]
          default: SUPPLIER_DECLARED
        inference_method:
          type: string
        inference_evidence:
          type: array
          items:
            type: object

    SupplyChainGraph:
      type: object
      properties:
        nodes:
          type: array
          items:
            $ref: '#/components/schemas/SupplyChainNode'
        edges:
          type: array
          items:
            $ref: '#/components/schemas/SupplyChainEdge'
        metadata:
          type: object
          properties:
            total_nodes:
              type: integer
            total_edges:
              type: integer
            max_tier:
              type: integer
            commodities:
              type: array
              items:
                type: string
            has_cycles:
              type: boolean
            inferred_edge_count:
              type: integer

    CoverageReport:
      type: object
      properties:
        importer_id:
          type: string
        commodity:
          type: string
        overall_coverage:
          type: number
        mapping_completeness:
          type: number
        plot_coverage:
          type: number
        tier_coverage:
          type: object
          additionalProperties:
            type: number
        volume_coverage:
          type: number
        gaps:
          type: array
          items:
            $ref: '#/components/schemas/Gap'
        gap_summary:
          type: object
          properties:
            critical:
              type: integer
            high:
              type: integer
            medium:
              type: integer
            low:
              type: integer

    CoverageGateResult:
      type: object
      properties:
        can_proceed_to_risk_assessment:
          type: boolean
        can_submit_dds:
          type: boolean
        blocking_gaps:
          type: array
          items:
            $ref: '#/components/schemas/Gap'
        risk_level_applied:
          type: string

    Gap:
      type: object
      properties:
        gap_id:
          type: string
        node_id:
          type: string
        gap_type:
          type: string
        severity:
          type: string
          enum: [CRITICAL, HIGH, MEDIUM, LOW]
        description:
          type: string
        remediation_suggestion:
          type: string

    SnapshotDiff:
      type: object
      properties:
        base_snapshot_id:
          type: string
        compare_snapshot_id:
          type: string
        nodes_added:
          type: array
          items:
            type: string
        nodes_removed:
          type: array
          items:
            type: string
        nodes_modified:
          type: array
          items:
            type: object
            properties:
              node_id:
                type: string
              changes:
                type: object
        edges_added:
          type: array
          items:
            type: string
        edges_removed:
          type: array
          items:
            type: string
        coverage_change:
          type: object
          properties:
            previous:
              type: number
            current:
              type: number
            delta:
              type: number

    EntityResolutionCandidate:
      type: object
      properties:
        candidate_id:
          type: string
        node_a:
          $ref: '#/components/schemas/SupplyChainNode'
        node_b:
          $ref: '#/components/schemas/SupplyChainNode'
        similarity_score:
          type: number
        matching_features:
          type: array
          items:
            type: string
        resolution_status:
          type: string

    NLQueryResult:
      type: object
      properties:
        original_query:
          type: string
        interpreted_query:
          type: string
          description: How the LLM interpreted the query
        generated_filter:
          type: object
          description: The structured filter generated from NL
        results:
          type: array
          items:
            $ref: '#/components/schemas/SupplyChainNode'
        result_count:
          type: integer
```

---

## 12. Technical Design: Entity Resolution

### 12.1 Overview

Entity resolution is the process of identifying and merging supplier records that refer to the same real-world entity across multiple data sources. This is critical for EUDR compliance as suppliers may appear in ERP systems, questionnaires, certifications, and customs data with variations in names and identifiers.

### 12.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Entity Resolution Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Data       │    │  Candidate   │    │   Scoring    │       │
│  │  Ingestion   │───▶│  Generation  │───▶│   Engine     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Normalization│    │   Blocking   │    │  Decision    │       │
│  │   Engine     │    │   Strategy   │    │   Policy     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│                              ┌─────────────────────────────┐    │
│                              │      Resolution Action       │    │
│                              │  AUTO_MERGE │ REVIEW │ SKIP  │    │
│                              └─────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 12.3 Implementation

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from uuid import UUID
from enum import Enum
import hashlib
import re
from difflib import SequenceMatcher


class ResolutionDecision(Enum):
    AUTO_MERGE = "AUTO_MERGE"
    REVIEW = "REVIEW"
    NO_MERGE = "NO_MERGE"


@dataclass
class MatchFeature:
    feature_name: str
    score: float
    weight: float
    is_strong: bool = False
    evidence: Optional[str] = None


@dataclass
class MatchResult:
    node_a_id: UUID
    node_b_id: UUID
    overall_score: float
    features: List[MatchFeature]
    decision: ResolutionDecision
    strong_feature_matched: bool


class EntityResolutionEngine:
    """
    Hybrid entity resolution engine combining deterministic rules
    and ML-based scoring for supplier deduplication.
    """

    # Thresholds (stakeholder approved)
    AUTO_MERGE_THRESHOLD = 0.98
    REVIEW_THRESHOLD = 0.85

    # Strong features that increase confidence
    STRONG_FEATURES = {
        "tax_id_match",
        "duns_match",
        "eori_match",
        "exact_address_country_match",
        "email_domain_match"
    }

    # Feature weights for scoring
    FEATURE_WEIGHTS = {
        "tax_id_match": 0.30,
        "duns_match": 0.25,
        "eori_match": 0.25,
        "name_similarity": 0.15,
        "address_similarity": 0.10,
        "country_match": 0.05,
        "phone_match": 0.05,
        "email_domain_match": 0.10,
        "certification_overlap": 0.05
    }

    def __init__(self, db_session, embedding_service=None):
        self.db = db_session
        self.embedding_service = embedding_service  # For ML-based matching

    def find_candidates(
        self,
        node_id: UUID,
        scope: str = "ALL"
    ) -> List[UUID]:
        """
        Find potential duplicate candidates using blocking strategy.
        Blocking reduces comparison space from O(n²) to O(n).
        """
        node = self.get_node(node_id)
        candidates = set()

        # Block 1: Same country + first 3 chars of normalized name
        name_prefix = self._normalize_name(node.name)[:3].upper()
        candidates.update(
            self._query_by_block(
                f"country:{node.country_code}:name_prefix:{name_prefix}"
            )
        )

        # Block 2: Same tax ID (if exists)
        if node.tax_id:
            normalized_tax = self._normalize_tax_id(node.tax_id, node.country_code)
            candidates.update(
                self._query_by_tax_id(normalized_tax)
            )

        # Block 3: Same DUNS/EORI
        if node.duns_number:
            candidates.update(self._query_by_duns(node.duns_number))
        if node.eori_number:
            candidates.update(self._query_by_eori(node.eori_number))

        # Block 4: Embedding similarity (if ML enabled)
        if self.embedding_service:
            similar_by_embedding = self.embedding_service.find_similar(
                self._create_entity_text(node),
                top_k=50,
                threshold=0.8
            )
            candidates.update(similar_by_embedding)

        # Remove self
        candidates.discard(node_id)

        return list(candidates)

    def score_pair(
        self,
        node_a_id: UUID,
        node_b_id: UUID
    ) -> MatchResult:
        """
        Calculate similarity score between two nodes.
        """
        node_a = self.get_node(node_a_id)
        node_b = self.get_node(node_b_id)
        features = []

        # 1. Tax ID Match (deterministic)
        if node_a.tax_id and node_b.tax_id:
            tax_a = self._normalize_tax_id(node_a.tax_id, node_a.country_code)
            tax_b = self._normalize_tax_id(node_b.tax_id, node_b.country_code)
            if tax_a == tax_b:
                features.append(MatchFeature(
                    feature_name="tax_id_match",
                    score=1.0,
                    weight=self.FEATURE_WEIGHTS["tax_id_match"],
                    is_strong=True,
                    evidence=f"Tax ID: {tax_a}"
                ))

        # 2. DUNS Match (deterministic)
        if node_a.duns_number and node_b.duns_number:
            if node_a.duns_number == node_b.duns_number:
                features.append(MatchFeature(
                    feature_name="duns_match",
                    score=1.0,
                    weight=self.FEATURE_WEIGHTS["duns_match"],
                    is_strong=True,
                    evidence=f"DUNS: {node_a.duns_number}"
                ))

        # 3. EORI Match (deterministic)
        if node_a.eori_number and node_b.eori_number:
            if node_a.eori_number == node_b.eori_number:
                features.append(MatchFeature(
                    feature_name="eori_match",
                    score=1.0,
                    weight=self.FEATURE_WEIGHTS["eori_match"],
                    is_strong=True,
                    evidence=f"EORI: {node_a.eori_number}"
                ))

        # 4. Name Similarity (fuzzy)
        name_score = self._calculate_name_similarity(node_a.name, node_b.name)
        features.append(MatchFeature(
            feature_name="name_similarity",
            score=name_score,
            weight=self.FEATURE_WEIGHTS["name_similarity"],
            evidence=f"Name sim: {name_score:.2f}"
        ))

        # 5. Address Similarity (fuzzy)
        if node_a.address and node_b.address:
            addr_score = self._calculate_address_similarity(
                node_a.address, node_b.address
            )
            is_exact = addr_score > 0.95 and node_a.country_code == node_b.country_code
            features.append(MatchFeature(
                feature_name="address_similarity",
                score=addr_score,
                weight=self.FEATURE_WEIGHTS["address_similarity"],
                is_strong=is_exact,
                evidence=f"Address sim: {addr_score:.2f}"
            ))
            if is_exact:
                features.append(MatchFeature(
                    feature_name="exact_address_country_match",
                    score=1.0,
                    weight=0.15,
                    is_strong=True
                ))

        # 6. Country Match
        if node_a.country_code == node_b.country_code:
            features.append(MatchFeature(
                feature_name="country_match",
                score=1.0,
                weight=self.FEATURE_WEIGHTS["country_match"]
            ))

        # 7. Certification Overlap
        cert_overlap = self._calculate_certification_overlap(
            node_a.certifications, node_b.certifications
        )
        if cert_overlap > 0:
            features.append(MatchFeature(
                feature_name="certification_overlap",
                score=cert_overlap,
                weight=self.FEATURE_WEIGHTS["certification_overlap"],
                evidence=f"Cert overlap: {cert_overlap:.2f}"
            ))

        # Calculate overall score
        total_weight = sum(f.weight for f in features)
        overall_score = sum(f.score * f.weight for f in features) / total_weight if total_weight > 0 else 0

        # Check for strong feature match
        strong_matched = any(f.is_strong for f in features)

        # Make decision
        decision = self._make_decision(overall_score, strong_matched)

        return MatchResult(
            node_a_id=node_a_id,
            node_b_id=node_b_id,
            overall_score=overall_score,
            features=features,
            decision=decision,
            strong_feature_matched=strong_matched
        )

    def _make_decision(
        self,
        score: float,
        strong_feature_matched: bool
    ) -> ResolutionDecision:
        """
        Decision policy (precision-first):
        - AUTO_MERGE: ≥0.98 AND strong feature match
        - REVIEW: 0.85-0.98 OR (≥0.98 without strong feature)
        - NO_MERGE: <0.85
        """
        if score >= self.AUTO_MERGE_THRESHOLD:
            if strong_feature_matched:
                return ResolutionDecision.AUTO_MERGE
            return ResolutionDecision.REVIEW
        elif score >= self.REVIEW_THRESHOLD:
            return ResolutionDecision.REVIEW
        return ResolutionDecision.NO_MERGE

    def merge_nodes(
        self,
        survivor_id: UUID,
        victim_id: UUID,
        resolution_method: str
    ) -> UUID:
        """
        Merge two nodes into one, using field-level golden record strategy.
        """
        survivor = self.get_node(survivor_id)
        victim = self.get_node(victim_id)

        # Apply field-level golden record
        merged_data = self._apply_golden_record(survivor, victim)

        # Update survivor with merged data
        self._update_node(survivor_id, merged_data)

        # Redirect all edges from victim to survivor
        self._redirect_edges(victim_id, survivor_id)

        # Mark victim as merged (soft delete)
        self._mark_as_merged(victim_id, survivor_id)

        # Log the merge for audit
        self._log_merge(survivor_id, victim_id, resolution_method)

        return survivor_id

    def _apply_golden_record(self, survivor, victim) -> Dict:
        """
        Field-level golden record: pick best value for each field.
        """
        FIELD_SOURCE_PRIORITY = {
            "tax_id": ["SAP", "QUESTIONNAIRE", "MANUAL"],
            "address": ["QUESTIONNAIRE", "SAP", "MANUAL"],
            "certifications": ["CERTIFICATION_BODY", "QUESTIONNAIRE", "SAP"],
            "name": ["SAP", "QUESTIONNAIRE", "MANUAL"]
        }

        merged = {}
        field_sources = {}

        for field, priorities in FIELD_SOURCE_PRIORITY.items():
            survivor_val = getattr(survivor, field, None)
            victim_val = getattr(victim, field, None)
            survivor_source = survivor.field_sources.get(field)
            victim_source = victim.field_sources.get(field)

            # Pick value from higher priority source
            if survivor_val and victim_val:
                survivor_priority = priorities.index(survivor_source) if survivor_source in priorities else 99
                victim_priority = priorities.index(victim_source) if victim_source in priorities else 99

                if victim_priority < survivor_priority:
                    merged[field] = victim_val
                    field_sources[field] = victim_source
                else:
                    merged[field] = survivor_val
                    field_sources[field] = survivor_source
            else:
                merged[field] = survivor_val or victim_val
                field_sources[field] = survivor_source or victim_source

        merged["field_sources"] = field_sources
        return merged

    def _normalize_name(self, name: str) -> str:
        """Normalize company name for comparison."""
        name = name.upper()
        # Remove common suffixes
        for suffix in ["LTD", "LLC", "INC", "CORP", "PTE", "SDN BHD", "PT", "CV"]:
            name = re.sub(rf'\b{suffix}\.?\b', '', name)
        # Remove punctuation and extra spaces
        name = re.sub(r'[^\w\s]', '', name)
        name = ' '.join(name.split())
        return name

    def _normalize_tax_id(self, tax_id: str, country: str) -> str:
        """Normalize tax ID based on country format."""
        tax_id = re.sub(r'[\s\-\.]', '', tax_id.upper())
        return f"{country}:{tax_id}"

    def _calculate_name_similarity(self, name_a: str, name_b: str) -> float:
        """Calculate name similarity using multiple methods."""
        norm_a = self._normalize_name(name_a)
        norm_b = self._normalize_name(name_b)

        # Exact match after normalization
        if norm_a == norm_b:
            return 1.0

        # Sequence matcher
        seq_score = SequenceMatcher(None, norm_a, norm_b).ratio()

        # Token overlap (Jaccard)
        tokens_a = set(norm_a.split())
        tokens_b = set(norm_b.split())
        if tokens_a and tokens_b:
            jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        else:
            jaccard = 0

        # Return weighted average
        return 0.6 * seq_score + 0.4 * jaccard

    def _calculate_address_similarity(self, addr_a: dict, addr_b: dict) -> float:
        """Calculate address similarity."""
        scores = []

        # Compare each component
        for component in ["street", "city", "postal_code", "region"]:
            val_a = addr_a.get(component, "").upper()
            val_b = addr_b.get(component, "").upper()
            if val_a and val_b:
                scores.append(SequenceMatcher(None, val_a, val_b).ratio())

        return sum(scores) / len(scores) if scores else 0

    def _calculate_certification_overlap(
        self,
        certs_a: List[Dict],
        certs_b: List[Dict]
    ) -> float:
        """Calculate overlap in certifications."""
        if not certs_a or not certs_b:
            return 0

        cert_ids_a = {c.get("certificate_number") for c in certs_a if c.get("certificate_number")}
        cert_ids_b = {c.get("certificate_number") for c in certs_b if c.get("certificate_number")}

        if not cert_ids_a or not cert_ids_b:
            return 0

        return len(cert_ids_a & cert_ids_b) / len(cert_ids_a | cert_ids_b)

    # Placeholder methods for database operations
    def get_node(self, node_id: UUID): pass
    def _query_by_block(self, block_key: str) -> List[UUID]: pass
    def _query_by_tax_id(self, tax_id: str) -> List[UUID]: pass
    def _query_by_duns(self, duns: str) -> List[UUID]: pass
    def _query_by_eori(self, eori: str) -> List[UUID]: pass
    def _create_entity_text(self, node) -> str: pass
    def _update_node(self, node_id: UUID, data: Dict): pass
    def _redirect_edges(self, from_id: UUID, to_id: UUID): pass
    def _mark_as_merged(self, victim_id: UUID, survivor_id: UUID): pass
    def _log_merge(self, survivor_id: UUID, victim_id: UUID, method: str): pass
```

### 12.4 Batch Processing

```python
class EntityResolutionBatchProcessor:
    """
    Batch processor for running entity resolution on new or all nodes.
    """

    def __init__(self, engine: EntityResolutionEngine, db_session):
        self.engine = engine
        self.db = db_session

    async def run_batch(
        self,
        scope: str = "NEW_NODES",
        node_ids: Optional[List[UUID]] = None
    ) -> Dict:
        """
        Run entity resolution batch.

        Args:
            scope: ALL, NEW_NODES, or SPECIFIC_NODES
            node_ids: List of specific node IDs if scope is SPECIFIC_NODES
        """
        stats = {
            "nodes_processed": 0,
            "candidates_found": 0,
            "auto_merged": 0,
            "sent_to_review": 0,
            "no_match": 0
        }

        # Get nodes to process
        if scope == "SPECIFIC_NODES" and node_ids:
            nodes = node_ids
        elif scope == "NEW_NODES":
            nodes = self._get_new_nodes()
        else:
            nodes = self._get_all_nodes()

        for node_id in nodes:
            stats["nodes_processed"] += 1

            # Find candidates
            candidates = self.engine.find_candidates(node_id)
            stats["candidates_found"] += len(candidates)

            # Score each candidate
            for candidate_id in candidates:
                result = self.engine.score_pair(node_id, candidate_id)

                if result.decision == ResolutionDecision.AUTO_MERGE:
                    # Auto-merge immediately
                    self.engine.merge_nodes(
                        survivor_id=node_id,
                        victim_id=candidate_id,
                        resolution_method="AUTO_MERGE"
                    )
                    stats["auto_merged"] += 1

                elif result.decision == ResolutionDecision.REVIEW:
                    # Queue for human review
                    self._create_review_candidate(result)
                    stats["sent_to_review"] += 1

                else:
                    stats["no_match"] += 1

        return stats

    def _get_new_nodes(self) -> List[UUID]:
        """Get nodes not yet processed for entity resolution."""
        pass

    def _get_all_nodes(self) -> List[UUID]:
        """Get all nodes."""
        pass

    def _create_review_candidate(self, result: MatchResult):
        """Create entity resolution candidate record for review."""
        pass
```

---

## 13. Technical Design: LLM Integration

### 13.1 Overview

LLM integration provides three specific capabilities:
1. **Entity Extraction** - Extract supplier information from unstructured documents
2. **Fuzzy Matching Assist** - Help entity resolution when rules are inconclusive
3. **Natural Language Queries** - Allow users to query supply chains in plain English

**Important:** Core compliance logic (coverage calculation, risk scoring, DDS eligibility) remains deterministic. LLM outputs are always validated and never directly modify compliance-critical data.

### 13.2 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     LLM Integration Layer                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Entity Extractor │  │ Matching Assist │  │  NL Query Engine │  │
│  │                  │  │                  │  │                  │  │
│  │ • Invoice Parser │  │ • Uncertain     │  │ • Query Parser   │  │
│  │ • PDF Extraction │  │   pair scoring  │  │ • Filter Builder │  │
│  │ • OCR + NER      │  │ • Explanation   │  │ • Result Format  │  │
│  └────────┬─────────┘  └────────┬────────┘  └────────┬─────────┘  │
│           │                     │                     │           │
│           ▼                     ▼                     ▼           │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │                    Validation Layer                           ││
│  │  • Schema validation   • Confidence thresholds                ││
│  │  • Human review flags  • Audit logging                        ││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 13.3 Implementation

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from uuid import UUID
from enum import Enum
import json


@dataclass
class ExtractedSupplier:
    name: Optional[str]
    address: Optional[Dict[str, str]]
    country_code: Optional[str]
    tax_id: Optional[str]
    contact_email: Optional[str]
    contact_phone: Optional[str]
    commodities: List[str]
    confidence: float
    extraction_notes: List[str]


@dataclass
class NLQueryResult:
    original_query: str
    interpreted_query: str
    generated_filter: Dict[str, Any]
    cypher_query: Optional[str]
    results: List[Dict]
    result_count: int
    confidence: float


class LLMIntegrationService:
    """
    LLM integration service for supply chain operations.
    Uses Claude/GPT-4 for intelligent extraction and querying.
    """

    def __init__(self, llm_client, db_session, graph_service):
        self.llm = llm_client
        self.db = db_session
        self.graph = graph_service

    # ==================== ENTITY EXTRACTION ====================

    async def extract_supplier_from_document(
        self,
        document_content: str,
        document_type: str,
        document_metadata: Optional[Dict] = None
    ) -> ExtractedSupplier:
        """
        Extract supplier information from unstructured document.

        Args:
            document_content: Text content of document (from OCR if needed)
            document_type: INVOICE, CERTIFICATE, CONTRACT, DECLARATION
            document_metadata: Optional metadata about the document
        """
        prompt = f"""Extract supplier information from this {document_type}.

Document content:
{document_content}

Extract the following fields if present:
- Supplier/Vendor name
- Full address (street, city, postal code, country)
- Country code (ISO 2-letter)
- Tax ID / VAT number
- Contact email
- Contact phone
- Commodities mentioned

Return as JSON with this structure:
{{
    "name": "string or null",
    "address": {{"street": "", "city": "", "postal_code": "", "country": ""}},
    "country_code": "XX or null",
    "tax_id": "string or null",
    "contact_email": "string or null",
    "contact_phone": "string or null",
    "commodities": ["list of commodities"],
    "confidence": 0.0-1.0,
    "notes": ["any extraction uncertainties"]
}}

Be conservative with confidence. Only report high confidence (>0.8) if the information is clearly and unambiguously stated."""

        response = await self.llm.complete(prompt)

        try:
            extracted = json.loads(response)
            return ExtractedSupplier(
                name=extracted.get("name"),
                address=extracted.get("address"),
                country_code=extracted.get("country_code"),
                tax_id=extracted.get("tax_id"),
                contact_email=extracted.get("contact_email"),
                contact_phone=extracted.get("contact_phone"),
                commodities=extracted.get("commodities", []),
                confidence=extracted.get("confidence", 0.5),
                extraction_notes=extracted.get("notes", [])
            )
        except json.JSONDecodeError:
            return ExtractedSupplier(
                name=None, address=None, country_code=None,
                tax_id=None, contact_email=None, contact_phone=None,
                commodities=[], confidence=0.0,
                extraction_notes=["Failed to parse LLM response"]
            )

    # ==================== FUZZY MATCHING ASSIST ====================

    async def assist_entity_matching(
        self,
        node_a: Dict,
        node_b: Dict,
        feature_scores: Dict[str, float]
    ) -> Dict:
        """
        Use LLM to help evaluate uncertain entity matches.

        Only called when deterministic rules give scores in review range (0.85-0.98).
        """
        prompt = f"""Evaluate if these two supplier records refer to the same real-world entity.

Record A:
- Name: {node_a.get('name')}
- Country: {node_a.get('country_code')}
- Address: {json.dumps(node_a.get('address', {}))}
- Tax ID: {node_a.get('tax_id', 'Not provided')}

Record B:
- Name: {node_b.get('name')}
- Country: {node_b.get('country_code')}
- Address: {json.dumps(node_b.get('address', {}))}
- Tax ID: {node_b.get('tax_id', 'Not provided')}

Current matching scores:
{json.dumps(feature_scores, indent=2)}

Analyze whether these are likely the same entity. Consider:
1. Name variations (abbreviations, translations, typos)
2. Address formatting differences
3. Common patterns in your knowledge of company naming

Return JSON:
{{
    "likely_same_entity": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "key_factors": ["list of decisive factors"],
    "recommendation": "MERGE" or "NO_MERGE" or "NEEDS_HUMAN_REVIEW"
}}"""

        response = await self.llm.complete(prompt)

        try:
            result = json.loads(response)
            return {
                "llm_assessment": result,
                "feature_scores": feature_scores,
                "combined_recommendation": self._combine_assessment(
                    feature_scores, result
                )
            }
        except json.JSONDecodeError:
            return {
                "llm_assessment": None,
                "feature_scores": feature_scores,
                "combined_recommendation": "NEEDS_HUMAN_REVIEW"
            }

    def _combine_assessment(
        self,
        feature_scores: Dict[str, float],
        llm_result: Dict
    ) -> str:
        """Combine rule-based scores with LLM assessment."""
        if llm_result.get("confidence", 0) < 0.7:
            return "NEEDS_HUMAN_REVIEW"

        if llm_result.get("recommendation") == "MERGE":
            # Boost confidence if LLM agrees
            return "MERGE" if llm_result.get("confidence", 0) > 0.85 else "NEEDS_HUMAN_REVIEW"

        return llm_result.get("recommendation", "NEEDS_HUMAN_REVIEW")

    # ==================== NATURAL LANGUAGE QUERIES ====================

    async def process_natural_language_query(
        self,
        query: str,
        importer_id: UUID,
        commodity: str
    ) -> NLQueryResult:
        """
        Process natural language query about supply chain.

        Examples:
        - "Show me all palm oil suppliers in Indonesia"
        - "Which suppliers have expired certifications?"
        - "Find producers within 10km of deforestation alerts"
        """
        # Step 1: Parse query intent
        parse_prompt = f"""Parse this supply chain query and extract structured filters.

Query: "{query}"
Context: Importer ID {importer_id}, Commodity {commodity}

Available filter fields:
- node_type: PRODUCER, PROCESSOR, TRADER, IMPORTER, AGGREGATOR
- country_code: ISO 2-letter country codes
- commodity: CATTLE, COCOA, COFFEE, PALM_OIL, RUBBER, SOY, WOOD
- tier: integer (0 = origin, higher = closer to importer)
- verification_status: VERIFIED, UNVERIFIED, PENDING
- disclosure_status: FULL, PARTIAL, NONE
- certification_status: VALID, EXPIRED, NONE
- data_source: SUPPLIER_DECLARED, INFERRED_CUSTOMS, etc.
- risk_score: 0.0-1.0 (higher = riskier)

Return JSON:
{{
    "interpreted_query": "human-readable interpretation",
    "filters": {{
        "node_type": ["list"] or null,
        "country_code": ["list"] or null,
        "tier": {{"min": int, "max": int}} or null,
        "verification_status": ["list"] or null,
        "certification_expired": true/false or null,
        "risk_score_min": float or null,
        "custom_conditions": ["additional conditions in natural language"]
    }},
    "sort_by": "field_name" or null,
    "sort_order": "asc" or "desc",
    "limit": integer or null
}}"""

        parse_response = await self.llm.complete(parse_prompt)

        try:
            parsed = json.loads(parse_response)
        except json.JSONDecodeError:
            return NLQueryResult(
                original_query=query,
                interpreted_query="Failed to parse query",
                generated_filter={},
                cypher_query=None,
                results=[],
                result_count=0,
                confidence=0.0
            )

        # Step 2: Generate database query
        filters = parsed.get("filters", {})
        cypher_query = self._build_cypher_query(importer_id, commodity, filters)

        # Step 3: Execute query
        results = await self.graph.execute_query(cypher_query)

        return NLQueryResult(
            original_query=query,
            interpreted_query=parsed.get("interpreted_query", query),
            generated_filter=filters,
            cypher_query=cypher_query,
            results=results,
            result_count=len(results),
            confidence=0.9 if results else 0.7
        )

    def _build_cypher_query(
        self,
        importer_id: UUID,
        commodity: str,
        filters: Dict
    ) -> str:
        """Build Cypher query from parsed filters."""
        where_clauses = [
            f"importer.node_id = '{importer_id}'",
            f"r.commodity = '{commodity}'"
        ]

        if filters.get("node_type"):
            types = filters["node_type"]
            where_clauses.append(f"n.node_type IN {types}")

        if filters.get("country_code"):
            countries = filters["country_code"]
            where_clauses.append(f"n.country_code IN {countries}")

        if filters.get("verification_status"):
            statuses = filters["verification_status"]
            where_clauses.append(f"n.verification_status IN {statuses}")

        if filters.get("tier"):
            tier = filters["tier"]
            if tier.get("min") is not None:
                where_clauses.append(f"n.tier >= {tier['min']}")
            if tier.get("max") is not None:
                where_clauses.append(f"n.tier <= {tier['max']}")

        where_clause = " AND ".join(where_clauses)

        query = f"""
        MATCH path = (n)-[r*]->(importer:Importer)
        WHERE {where_clause}
        RETURN DISTINCT n
        ORDER BY n.tier ASC
        LIMIT 100
        """

        return query
```

### 13.4 Safety & Audit

```python
class LLMAuditLogger:
    """
    Audit logging for all LLM operations.
    Required for compliance and debugging.
    """

    def log_extraction(
        self,
        document_id: str,
        input_content_hash: str,
        output: ExtractedSupplier,
        model_used: str,
        latency_ms: int
    ):
        """Log entity extraction operation."""
        pass

    def log_matching_assist(
        self,
        node_a_id: UUID,
        node_b_id: UUID,
        llm_recommendation: str,
        final_decision: str,
        model_used: str
    ):
        """Log matching assistance operation."""
        pass

    def log_nl_query(
        self,
        user_id: str,
        original_query: str,
        generated_filter: Dict,
        result_count: int,
        model_used: str
    ):
        """Log natural language query."""
        pass
```

---

## 14. Algorithms

### 14.1 Multi-Tier Discovery Algorithm

```python
from collections import deque
from typing import List, Set, Dict, Optional
from uuid import UUID
from dataclasses import dataclass


@dataclass
class DiscoveryResult:
    graph: 'SupplyChainGraph'
    cycles_detected: List[Dict]
    coverage_metrics: Dict


def discover_supply_chain(
    importer_id: UUID,
    commodity: str,
    max_depth: int = 10,
    include_inferred: bool = True
) -> DiscoveryResult:
    """
    Discover full supply chain from importer back to origin plots.
    Uses BFS with transaction data mining and cycle detection.
    """
    graph = SupplyChainGraph()
    visited = set()
    tier_appearances = {}  # node_id -> list of tiers
    cycles = []
    queue = deque([(importer_id, 0)])

    while queue:
        node_id, tier = queue.popleft()

        if tier > max_depth:
            continue

        # Track multi-tier appearances
        if node_id in tier_appearances:
            tier_appearances[node_id].append(tier)
            # Node already processed at different tier - skip re-processing
            # but record the additional tier appearance
            continue
        else:
            tier_appearances[node_id] = [tier]

        visited.add(node_id)
        node = get_node(node_id)
        graph.add_node(node)

        # Find suppliers from transaction data
        suppliers = find_suppliers(node_id, commodity, include_inferred)

        for supplier in suppliers:
            # Check for cycles
            if supplier.id in visited:
                cycle = detect_and_resolve_cycle(
                    node_id, supplier.id, commodity, graph
                )
                if cycle:
                    cycles.append(cycle)
                continue

            edge = create_edge(
                source_id=supplier.id,
                target_id=node_id,
                commodity=commodity,
                data_source=supplier.source,
                confidence=supplier.confidence
            )

            # Add edge context for multi-tier support
            edge.edge_context = {
                "observed_tier": tier + 1,
                "is_direct_to_importer": tier == 0,
                "relationship_path": f"VIA:{node_id}" if tier > 0 else "DIRECT"
            }

            graph.add_edge(edge)
            queue.append((supplier.id, tier + 1))

    # Update nodes with tier information
    for node_id, tiers in tier_appearances.items():
        node = graph.get_node(node_id)
        node.tier = min(tiers)  # tier_min
        node.tier_max = max(tiers)
        node.all_tiers = sorted(tiers)
        node.metadata["tiers"] = tiers

    # Link origin plots to producer nodes
    for node in graph.nodes:
        if node.node_type == "PRODUCER":
            plots = find_plots(node.node_id, commodity)
            for plot in plots:
                graph.add_plot(node.node_id, plot)

    return DiscoveryResult(
        graph=graph,
        cycles_detected=cycles,
        coverage_metrics=calculate_coverage(graph, commodity)
    )


def detect_and_resolve_cycle(
    node_a_id: UUID,
    node_b_id: UUID,
    commodity: str,
    graph: 'SupplyChainGraph'
) -> Optional[Dict]:
    """
    Detect and resolve cycles using temporal/confidence strategy.
    """
    # Check if bidirectional edges exist
    edge_a_to_b = graph.get_edge(node_a_id, node_b_id, commodity)
    edge_b_to_a = graph.get_edge(node_b_id, node_a_id, commodity)

    if not (edge_a_to_b and edge_b_to_a):
        return None

    # Resolve using temporal information
    if edge_a_to_b.transaction_date and edge_b_to_a.transaction_date:
        if edge_a_to_b.transaction_date < edge_b_to_a.transaction_date:
            keep, suppress = edge_a_to_b, edge_b_to_a
            resolution_method = "temporal"
        else:
            keep, suppress = edge_b_to_a, edge_a_to_b
            resolution_method = "temporal"
    else:
        # Fallback to confidence score
        if edge_a_to_b.confidence_score >= edge_b_to_a.confidence_score:
            keep, suppress = edge_a_to_b, edge_b_to_a
        else:
            keep, suppress = edge_b_to_a, edge_a_to_b
        resolution_method = "confidence"

    # Remove suppressed edge
    graph.remove_edge(suppress.edge_id)

    return {
        "type": "CYCLE_DETECTED",
        "nodes": [str(node_a_id), str(node_b_id)],
        "kept_edge": str(keep.edge_id),
        "suppressed_edge": str(suppress.edge_id),
        "resolution_method": resolution_method,
        "severity": "MEDIUM"
    }
```

### 14.2 Coverage Calculation with Gates

```python
@dataclass
class CoverageReport:
    overall_coverage: float
    mapping_completeness: float
    plot_coverage: float
    tier_coverage: Dict[int, float]
    volume_coverage: float
    gaps: List['Gap']
    gap_summary: Dict[str, int]
    can_proceed_to_risk: bool
    can_submit_dds: bool


class CoverageGate:
    """Coverage gate logic for risk assessment and DDS submission."""

    BASELINE_MAPPING = 0.95
    BASELINE_PLOT = 0.90
    HIGH_RISK_MAPPING = 0.98
    HIGH_RISK_PLOT = 0.95

    def calculate_coverage(
        self,
        graph: 'SupplyChainGraph',
        commodity: str,
        risk_level: str = "STANDARD"
    ) -> CoverageReport:
        """Calculate comprehensive coverage metrics."""

        # 1. Mapping completeness (suppliers with verified upstream)
        total_nodes = len(graph.nodes)
        verified_nodes = len([n for n in graph.nodes if n.verification_status == "VERIFIED"])
        mapping_completeness = verified_nodes / total_nodes if total_nodes > 0 else 0

        # 2. Plot coverage (producers with plot data)
        producers = [n for n in graph.nodes if n.node_type == "PRODUCER"]
        producers_with_plots = [p for p in producers if graph.get_plots(p.node_id)]
        plot_coverage = len(producers_with_plots) / len(producers) if producers else 1.0

        # 3. Tier coverage
        tier_coverage = {}
        for tier in range(graph.max_tier + 1):
            tier_nodes = [n for n in graph.nodes if n.tier == tier]
            verified = [n for n in tier_nodes if n.verification_status == "VERIFIED"]
            tier_coverage[tier] = len(verified) / len(tier_nodes) if tier_nodes else 0

        # 4. Volume coverage (verified edges)
        total_volume = sum(e.quantity or 0 for e in graph.edges)
        verified_volume = sum(e.quantity or 0 for e in graph.edges if e.verified)
        volume_coverage = verified_volume / total_volume if total_volume else 0

        # 5. Identify gaps
        gaps = self._identify_gaps(graph)
        gap_summary = self._summarize_gaps(gaps)

        # 6. Calculate overall coverage (weighted)
        overall_coverage = (
            0.30 * mapping_completeness +
            0.35 * plot_coverage +
            0.20 * volume_coverage +
            0.15 * (sum(tier_coverage.values()) / len(tier_coverage) if tier_coverage else 0)
        )

        # 7. Check gates
        can_proceed_to_risk = self._check_risk_assessment_gate(
            mapping_completeness, plot_coverage, risk_level
        )
        can_submit_dds = self._check_dds_gate(gaps)

        return CoverageReport(
            overall_coverage=overall_coverage * 100,
            mapping_completeness=mapping_completeness * 100,
            plot_coverage=plot_coverage * 100,
            tier_coverage={k: v * 100 for k, v in tier_coverage.items()},
            volume_coverage=volume_coverage * 100,
            gaps=gaps,
            gap_summary=gap_summary,
            can_proceed_to_risk=can_proceed_to_risk,
            can_submit_dds=can_submit_dds
        )

    def _check_risk_assessment_gate(
        self,
        mapping: float,
        plot: float,
        risk_level: str
    ) -> bool:
        """Check if coverage meets risk assessment threshold."""
        if risk_level == "HIGH":
            return mapping >= self.HIGH_RISK_MAPPING and plot >= self.HIGH_RISK_PLOT
        return mapping >= self.BASELINE_MAPPING and plot >= self.BASELINE_PLOT

    def _check_dds_gate(self, gaps: List['Gap']) -> bool:
        """Check if any HIGH severity gaps block DDS submission."""
        high_gaps = [g for g in gaps if g.severity in ["CRITICAL", "HIGH"]]
        return len(high_gaps) == 0

    def _identify_gaps(self, graph: 'SupplyChainGraph') -> List['Gap']:
        """Identify all traceability gaps."""
        gaps = []

        for node in graph.nodes:
            # Unverified supplier
            if node.verification_status != "VERIFIED":
                gaps.append(Gap(
                    node_id=node.node_id,
                    gap_type="UNVERIFIED_SUPPLIER",
                    severity="MEDIUM",
                    description=f"Supplier {node.name} is not verified",
                    remediation="Request verification documents from supplier"
                ))

            # Missing plot data for producers
            if node.node_type == "PRODUCER" and not graph.get_plots(node.node_id):
                gaps.append(Gap(
                    node_id=node.node_id,
                    gap_type="MISSING_PLOT_DATA",
                    severity="HIGH",
                    description=f"Producer {node.name} has no plot geolocation data",
                    remediation="Collect plot coordinates from supplier"
                ))

            # Partial disclosure
            if node.disclosure_status == "PARTIAL":
                gaps.append(Gap(
                    node_id=node.node_id,
                    gap_type="PARTIAL_DISCLOSURE",
                    severity="MEDIUM",
                    description=f"Supplier {node.name} has not disclosed Tier 2+ suppliers",
                    remediation="Request full supply chain disclosure or use alternative sourcing"
                ))

        return gaps

    def _summarize_gaps(self, gaps: List['Gap']) -> Dict[str, int]:
        """Summarize gaps by severity."""
        return {
            "critical": len([g for g in gaps if g.severity == "CRITICAL"]),
            "high": len([g for g in gaps if g.severity == "HIGH"]),
            "medium": len([g for g in gaps if g.severity == "MEDIUM"]),
            "low": len([g for g in gaps if g.severity == "LOW"])
        }
```

---

## 15. UI/UX Specification

### 15.1 Visualization Paradigms

The system supports **four visualization paradigms** to serve different user tasks:

| Paradigm | Primary Use Case | Key Features |
|----------|------------------|--------------|
| **Interactive Graph** | Explore relationships | Force-directed layout, zoom/pan, click-to-expand, node filtering |
| **Sankey Diagram** | Analyze volume flows | Width-proportional streams, tier grouping, commodity coloring |
| **Tabular View** | Data analysis | Filter, sort, export, expandable rows, bulk actions |
| **Map View** | Geographic analysis | Plot locations, cluster markers, country heat maps |

### 15.2 Multi-Language Support

| Language | Code | Priority |
|----------|------|----------|
| English | en | P0 |
| Spanish | es | P1 |
| Portuguese | pt | P1 |
| French | fr | P1 |
| Indonesian | id | P1 |
| Vietnamese | vi | P1 |

### 15.3 Offline Capability

**Use Case:** Field auditors visiting remote farms/facilities with limited connectivity.

**Implementation:**
- Progressive Web App (PWA) with service worker
- Downloadable supply chain snapshots
- Offline-first data access
- Background sync when connectivity restored
- Read-only mode for offline access

---

## 16. Performance & Scale Strategy

### 16.1 Large Graph Handling (50,000+ Nodes)

| Strategy | Implementation |
|----------|----------------|
| **Progressive Loading** | Load 2-3 tiers initially; lazy-load deeper tiers on expand |
| **Pre-Computed Views** | Materialize common subgraphs during snapshot creation |
| **Server-Side Pagination** | Never send full graph to client; virtual scrolling |
| **Time-Boxed Queries** | 10-second timeout; return partial results with "more available" |

### 16.2 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   API Pod    │  │   API Pod    │  │   API Pod    │       │
│  │  (Scalable)  │  │  (Scalable)  │  │  (Scalable)  │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│         └────────────────┬┴─────────────────┘                │
│                          │                                   │
│                          ▼                                   │
│              ┌───────────────────────┐                       │
│              │     Load Balancer     │                       │
│              └───────────────────────┘                       │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Dedicated Graph Workers                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │  │
│  │  │  Neo4j Op   │  │  Discovery  │  │   Entity    │    │  │
│  │  │   Worker    │  │   Worker    │  │  Resolution │    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘    │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │   PostgreSQL     │  sync   │      Neo4j       │          │
│  │   (Primary)      │ ──────▶ │   (Read Cache)   │          │
│  └──────────────────┘         └──────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## 17. Access Control

### 17.1 Permission Levels

| Level | Description | Example |
|-------|-------------|---------|
| **Organization** | Users see only their company's data | Baseline for all users |
| **Commodity** | Restrict to specific commodities | Palm oil team sees only palm oil chains |
| **Tier** | Restrict depth visibility | Procurement sees Tier 1 only |

### 17.2 Implementation

```python
class SupplyChainPermissions:
    def can_view_node(self, user: User, node: SupplyChainNode) -> bool:
        # Organization check
        if node.owner_org != user.organization:
            return False

        # Commodity check
        if user.allowed_commodities:
            if not any(c in user.allowed_commodities for c in node.commodities):
                return False

        # Tier check
        if user.max_tier_depth is not None:
            if node.tier > user.max_tier_depth:
                return False

        return True
```

---

## 18. Agent Orchestration

### 18.1 Pipeline Sequence

```
┌─────────────────────────────────────────────────────────────────┐
│                    EUDR Compliance Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Data Ingestion (GL-DATA-X agents)                           │
│     └──▶ ERP data, questionnaires, documents                    │
│                          │                                       │
│                          ▼                                       │
│  2. GL-EUDR-001: Supply Chain Mapper                            │
│     └──▶ Build graph, create snapshot, calculate coverage       │
│                          │                                       │
│                          ▼                                       │
│  3. Coverage Gate Check                                          │
│     ├── IF gaps with missing plots ──▶ GL-EUDR-002 (Geolocation)│
│     ├── IF transformation gaps ──▶ GL-EUDR-003 (Traceability)   │
│     └── IF supplier gaps ──▶ GL-EUDR-004 (Verification)         │
│                          │                                       │
│                          ▼                                       │
│  4. Coverage Threshold Met?                                      │
│     ├── NO: Return to step 3, notify users                      │
│     └── YES: Continue                                           │
│                          │                                       │
│                          ▼                                       │
│  5. GL-EUDR-015: Risk Assessment                                │
│     └──▶ Calculate risk scores, flag high-risk suppliers        │
│                          │                                       │
│                          ▼                                       │
│  6. Risk Acceptable?                                             │
│     ├── NO: Mitigation required                                 │
│     └── YES: Continue                                           │
│                          │                                       │
│                          ▼                                       │
│  7. GL-EUDR-040: DDS Reporting                                  │
│     └──▶ Generate Due Diligence Statement                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 18.2 Handoff Contract

All downstream agents receive `snapshot_id` as input and produce artifacts linked back to that snapshot. Downstream agents are **read-only** on the snapshot.

---

## 19. Notification System

### 19.1 Alert Tiering

| Severity | Channel | Timing | Examples |
|----------|---------|--------|----------|
| **CRITICAL** | Push (Slack, Teams, webhooks) | Immediate | Coverage drops below DDS threshold, supplier sanctioned |
| **WARNING** | Email digest | Daily | Certification expiring, new gap detected |
| **INFO** | In-app only | On login | Snapshot created, minor coverage change |

---

## 20. Testing Strategy

### 20.1 Test Data Sources

| Source | Purpose |
|--------|---------|
| **Synthetic Generation** | Scale testing with known properties |
| **Industry Datasets** | Trase, Open Supply Hub for realistic structures |
| **Golden Test Cases** | Hand-crafted: 5-tier chain, cooperative, cycle, partial disclosure |

### 20.2 Test Categories

- **Unit Tests:** Node/edge CRUD, entity resolution scoring, coverage calculation
- **Integration Tests:** PostgreSQL ↔ Neo4j sync, API endpoints, snapshot creation
- **Golden Tests:** Known supply chain structures, expected coverage results
- **Performance Tests:** 100K node graphs, concurrent queries, incremental updates

---

## 21. Rollout Plan (12 Weeks)

| Phase | Weeks | Deliverables |
|-------|-------|--------------|
| **1** | 1-3 | Core graph data model, node/edge CRUD, single-tier mapping, PostgreSQL schema |
| **2** | 4-6 | Multi-tier discovery algorithm, entity resolution (rules + ML), plot linking, Neo4j sync |
| **3** | 7-9 | Coverage analysis with gates, gap identification, versioning/snapshots, as-of API |
| **4** | 10-12 | Performance optimization, LLM integration, UI views, downstream agent integration |

---

## 22. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Mapping Completeness | >95% of suppliers mapped to origin | Automated coverage report |
| Plot Coverage | >90% of producers linked to plots | Automated coverage report |
| Entity Resolution Accuracy | >95% | Manual audit of merged records |
| Query Latency | <2 seconds for graph retrieval | APM monitoring |
| Coverage Calculation | <30 seconds for 10K node graph | Performance tests |
| Auto-Merge Rate | >70% of candidates (precision-first) | Entity resolution stats |
| DDS Gate Pass Rate | >80% of submissions on first attempt | Pipeline metrics |

---

## 23. Dependencies and Integration

### 23.1 Upstream Dependencies

| Agent | Purpose |
|-------|---------|
| GL-DATA-X-001 | Document ingestion |
| GL-DATA-X-003 | ERP connector |
| GL-DATA-SUP-001 | Supplier questionnaire processor |

### 23.2 Downstream Consumers

| Agent | What It Needs |
|-------|---------------|
| GL-EUDR-002 | Snapshot ID, list of producers missing plots |
| GL-EUDR-003 | Snapshot ID, transformation edges |
| GL-EUDR-015 | Snapshot ID, full graph for risk scoring |
| GL-EUDR-040 | Snapshot ID, coverage report, DDS-eligible paths |

---

## 24. Risk Mitigation

### 24.1 Data Availability Risk (PRIMARY)

| Priority | Strategy | Implementation |
|----------|----------|----------------|
| **1** | Contractual Requirements | Disclosure obligation, SLA, flow-down clause |
| **2** | Supplier Incentives | Faster payments, preferred status for data sharing |
| **3** | Alternative Data Sources | Customs, shipping manifests, certifications, satellite |
| **4** | Industry Collaboration | Pre-competitive data sharing consortiums |

### 24.2 Technical Risks

| Risk | Mitigation |
|------|------------|
| Entity resolution false positives | Precision-first thresholds (0.98 for auto-merge), human review queue |
| Graph query performance | PostgreSQL primary + Neo4j cache, pre-computed views, time-boxed queries |
| Neo4j sync lag | Async sync with eventual consistency, fallback to PostgreSQL for critical paths |
| LLM hallucination | Validation layer, never use LLM for compliance-critical decisions |

---

## Appendix A: Decision Summary Matrix

| Area | Decision | Confidence |
|------|----------|------------|
| Graph Storage | PostgreSQL primary, Neo4j cache | HIGH |
| Entity Resolution | Hybrid rules + ML | HIGH |
| Coverage Threshold | Risk-weighted with DDS gate | HIGH |
| Multi-Tier Suppliers | Role-based edges | HIGH |
| Visualization | All 4 paradigms | HIGH |
| Orchestration | Orchestrator pattern | HIGH |
| Data Refusal | Partial acceptance | MEDIUM |
| Cooperatives | Individual farmers + aggregation | HIGH |
| Conflicts | Field-level golden record | HIGH |
| Provenance | Multi-layered tracking | HIGH |
| Scale Strategy | All 4 techniques | HIGH |
| Access Control | Org + Commodity + Tier | HIGH |
| Temporal Queries | Periodic snapshots | HIGH |
| Cycle Detection | Temporal resolution + warning | HIGH |
| LLM Integration | 3 specific tasks | HIGH |
| Deployment | Hybrid K8s + dedicated | HIGH |
| Notifications | Tiered by severity | HIGH |
| EUDR Deadlines | Operator classification flag | HIGH |
| Data Mitigation | Contracts > Incentives > Alt Sources | HIGH |
| Timeline | 12 weeks | HIGH |
| Ownership | EUDR Platform Team | HIGH |

---

## Appendix B: Resolved Open Questions

| Original Question | Resolution |
|-------------------|------------|
| How to handle suppliers at multiple tiers? | Role-based edges with tier context metadata |
| What is sufficient coverage threshold? | Risk-weighted: 95%/90% baseline; 100% for DDS path |
| How to handle cooperative shared plots? | Individual farmers + AGGREGATES edges; plot_producers join table in v1.1 |
| Entity resolution without tax IDs? | Hybrid rules + ML with precision-first thresholds |

---

*Document Version: 2.0*
*Created: 2026-01-30*
*Status: APPROVED FOR DEVELOPMENT*
*Owner: EUDR Platform Team*
*Next Review: After Phase 1 completion*
