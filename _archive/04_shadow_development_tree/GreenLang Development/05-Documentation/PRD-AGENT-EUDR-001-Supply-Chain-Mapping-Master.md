# PRD: AGENT-EUDR-001 -- Supply Chain Mapping Master

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-001 |
| **Agent ID** | GL-EUDR-SCM-001 |
| **Component** | Supply Chain Mapping Master Agent |
| **Category** | EUDR Regulatory Agent -- Supply Chain Intelligence |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-06 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-06 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) requires every operator and trader placing cattle, cocoa, coffee, palm oil, rubber, soya, or wood (and their derived products) on the EU market to demonstrate that those products are deforestation-free and legally produced, backed by a Due Diligence Statement (DDS) filed with the EU Information System. Central to this obligation is the ability to trace every product back to the specific plot of land where it was produced (Article 9), through every intermediary in the supply chain (Articles 4, 10).

Real-world EUDR supply chains are not linear. A single chocolate bar may contain cocoa from dozens of cooperatives spanning multiple countries, routed through aggregators, processors, traders, and refiners before reaching an EU importer. These many-to-many, multi-tier, multi-commodity supply chain topologies cannot be managed with spreadsheets or single-tier supplier databases. Today, most EU importers face the following gaps:

- **No graph-based supply chain model**: Existing systems track direct (Tier 1) suppliers only; they cannot represent the recursive, branching networks that connect production plots to final products.
- **No plot-level geolocation linkage**: Supply chain records are disconnected from the GPS/polygon geolocation data mandated by Article 9.
- **No many-to-many batch traceability**: When commodities from multiple plots are aggregated, split, and re-aggregated through processors and traders, the link between the final product and every contributing plot is lost.
- **No automated supply chain discovery**: Operators manually chase supplier questionnaires; there is no systematic method to discover and map sub-tier suppliers.
- **No supply chain risk propagation**: Risk assessed at the country or plot level is not propagated through the supply chain graph to identify which final products inherit that risk.
- **No visualization of complex topologies**: Without a graph visualization, compliance officers cannot identify gaps, bottlenecks, or opaque segments in the chain.

Without solving these problems, EU operators face penalties of up to 4% of annual EU turnover, confiscation of goods, temporary exclusion from public procurement, and public naming.

### 1.2 Solution Overview

Agent-EUDR-001: Supply Chain Mapping Master is a specialized agent that builds, maintains, analyzes, and visualizes the full multi-tier supply chain graph for all seven EUDR-regulated commodities and their derived products. It operates as a graph-native engine that models every actor (producer, collector, processor, trader, importer), every transfer of custody, every batch split/merge, and every production plot as nodes and edges in a directed acyclic graph (DAG). The agent integrates deeply with the existing AGENT-DATA-005 EUDR Traceability Connector for chain-of-custody data, AGENT-DATA-006 GIS/Mapping Connector for geospatial operations, and AGENT-DATA-007 Deforestation Satellite Connector for deforestation verification. It feeds enriched supply chain data into the GL-EUDR-APP platform.

Core capabilities:

1. **Graph-native supply chain modeling** -- Nodes (actors, plots, batches) and edges (custody transfers, commodity flows) with full attribute storage, supporting DAG traversal, cycle detection, and topological sorting.
2. **Multi-tier recursive mapping** -- Automatically discovers and maps supply chain depth from Tier 1 through Tier N, using supplier declarations, ERP data, and questionnaire responses.
3. **Plot-to-product traceability** -- Links every final product to every contributing production plot with GPS/polygon geolocation, maintaining the link through batch splits, merges, and transformations.
4. **Many-to-many topology support** -- Models complex real-world scenarios: one plot supplying many processors, one processor receiving from many plots, batch aggregation across origins.
5. **Risk propagation engine** -- Propagates country risk, commodity risk, supplier risk, and deforestation risk through the supply chain graph so that downstream products inherit upstream risk signals.
6. **Supply chain gap analysis** -- Identifies missing tiers, unverified actors, plots without geolocation, and custody chain breaks.
7. **Visualization and export** -- Generates interactive supply chain maps, Sankey diagrams, and graph exports (GeoJSON, GraphML, JSON-LD) for auditors and regulators.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Supply chain depth mapped | Tier 1 through Tier N (median >= 4 tiers) | Average tier depth across all mapped chains |
| Plot-to-product traceability | 100% of compliant products linked to origin plots | % of products with complete origin plot linkage |
| Many-to-many resolution | Support for 10,000+ nodes per supply chain graph | Max node count in production graphs |
| Supply chain gap detection | Identify 95%+ of missing links | Precision/recall against manually audited chains |
| Risk propagation accuracy | 100% deterministic, reproducible | Bit-perfect reproducibility tests |
| Graph traversal performance | < 500ms for 10,000-node graph queries | p99 latency under load |
| Visualization render time | < 3 seconds for 1,000-node interactive graph | Frontend render benchmarks |
| EUDR compliance coverage | All 7 commodities + derived products | Commodity coverage matrix |
| Regulatory acceptance | 100% of generated supply chain data accepted in DDS | EU Information System submission validation |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, with an estimated supply chain compliance technology market of 3-5 billion EUR.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of the 7 regulated commodities requiring digital supply chain mapping tools, estimated at 800M-1.2B EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1, representing 50M-80M EUR in supply chain mapping module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) of EUDR-regulated commodities
- Multinational food and beverage companies (cocoa, coffee, palm oil, soya)
- Timber and paper industry operators
- Automotive and tire manufacturers (rubber)
- Meat and leather importers (cattle)

**Secondary:**
- Customs brokers and freight forwarders handling EUDR-regulated goods
- Commodity traders and intermediaries
- Certification bodies (FSC, RSPO, Rainforest Alliance, UTZ)
- Compliance consultants and auditors
- SME importers (1,000-10,000 shipments/year) -- enforcement from June 30, 2026

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual / Spreadsheet | No cost; familiar | Cannot model multi-tier; no geolocation linkage; error-prone | Graph-native, automated, zero-error |
| Generic SCM platforms (SAP Ariba, Oracle SCM) | Enterprise integration; Tier 1 visibility | Not EUDR-specific; no plot-level geo; no deforestation check | Purpose-built for EUDR Article 9/10; plot-level integration |
| Niche EUDR tools (Preferred by Nature, Ecosphere+) | Commodity expertise | Single-commodity; limited graph modeling; manual tiers | All 7 commodities; automated multi-tier; graph-native |
| Blockchain traceability (Sourcemap, OpenSC) | Immutable audit trail | Expensive; slow; limited analytics; no risk propagation | Faster; cheaper; full risk propagation; deterministic |
| In-house custom builds | Tailored to org | 12-18 month build; no regulatory updates; no scale | Ready now; continuous regulatory updates; production-grade |

### 2.4 Differentiation Strategy

1. **Graph-native architecture** -- Not a bolted-on feature; the supply chain IS a graph from day one.
2. **Regulatory fidelity** -- Every data field maps to a specific EUDR Article requirement.
3. **Integration depth** -- Pre-built connectors to AGENT-DATA-005 (traceability), AGENT-DATA-006 (GIS), AGENT-DATA-007 (satellite), AGENT-DATA-003 (ERP), and the GL-EUDR-APP platform.
4. **Zero-hallucination risk propagation** -- Deterministic, auditable, reproducible risk calculations with no LLM in the critical path.
5. **Scale** -- Tested for supply chain graphs of 100,000+ nodes with sub-second query times.

---

## 3. Goals and Objectives

### 3.1 Business Goals

| # | Goal | Metric | Timeline |
|---|------|--------|----------|
| BG-1 | Enable EU operators to achieve EUDR compliance for supply chain traceability | 100% of customers pass Article 9/10 audits | Q2 2026 |
| BG-2 | Reduce time-to-map a multi-tier supply chain from weeks to hours | 90% reduction in mapping time | Q2 2026 |
| BG-3 | Become the reference supply chain mapping solution for EUDR | 500+ enterprise customers | Q4 2026 |
| BG-4 | Reduce compliance penalties for customers | Zero EUDR penalties for active customers | Ongoing |

### 3.2 Product Goals

| # | Goal | Description |
|---|------|-------------|
| PG-1 | Complete multi-tier mapping | Map supply chains from EU point-of-entry back to production plot for all 7 commodities |
| PG-2 | Many-to-many resolution | Model and resolve complex supply chain topologies with batch splitting and merging |
| PG-3 | Geolocation integration | Link every supply chain node to GPS/polygon data per EUDR Article 9 |
| PG-4 | Risk propagation | Propagate risk scores through the supply chain graph to identify inherited risk |
| PG-5 | Gap analysis | Automatically identify missing tiers, unverified actors, and incomplete geolocation |
| PG-6 | Regulatory export | Generate supply chain data in the format required for DDS submission |
| PG-7 | Visualization | Provide interactive supply chain graph visualization with geographic overlay |

### 3.3 Technical Goals

| # | Goal | Target |
|---|------|--------|
| TG-1 | Graph query performance | < 500ms p99 for 10,000-node graphs |
| TG-2 | Batch processing throughput | 50,000 custody transfers per minute |
| TG-3 | Memory efficiency | < 2 GB for 100,000-node graph in memory |
| TG-4 | API response time | < 200ms p95 for standard queries |
| TG-5 | Test coverage | >= 85% line coverage, >= 90% branch coverage |
| TG-6 | Zero-hallucination | 100% deterministic, bit-perfect reproducibility |

---

## 4. User Personas

### Persona 1: Compliance Officer -- Maria (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Regulatory Compliance at a large EU chocolate manufacturer |
| **Company** | 5,000 employees, sourcing cocoa from 12 countries |
| **EUDR Pressure** | Board-level mandate to achieve EUDR compliance by enforcement date |
| **Pain Points** | Cannot see beyond Tier 1 suppliers; no visibility into cooperative-level sourcing; manual Excel mapping takes 6 weeks per commodity; cannot link cocoa bags to specific farm plots |
| **Goals** | Full supply chain visibility from farm to factory; automated gap identification; audit-ready documentation |
| **Technical Skill** | Moderate -- comfortable with web applications but not a developer |

### Persona 2: Supply Chain Analyst -- Lukas (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Supply Chain Analyst at an EU timber importer |
| **Company** | 800 employees, importing tropical and temperate wood from 20+ countries |
| **EUDR Pressure** | Must map complex multi-step processing chains (forest -> sawmill -> veneer -> furniture) |
| **Pain Points** | Wood passes through 5-7 intermediaries; batch mixing at sawmills destroys traceability; no way to model mass balance across processors |
| **Goals** | Model the full processing chain with batch split/merge; maintain plot-level linkage through transformations; generate compliant custody documentation |
| **Technical Skill** | High -- comfortable with data tools, APIs, and basic scripting |

### Persona 3: Procurement Manager -- Ana (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Procurement Director at a palm oil refinery |
| **Company** | 3,000 employees, sourcing from 200+ plantations across Indonesia and Malaysia |
| **EUDR Pressure** | Must onboard suppliers with plot-level geolocation data; must verify deforestation-free status |
| **Pain Points** | Suppliers are reluctant to share sub-tier information; no standardized onboarding workflow; manual collection of GPS coordinates from plantations |
| **Goals** | Streamlined supplier onboarding with geolocation capture; automated risk screening of new suppliers; visibility into supplier sub-tiers |
| **Technical Skill** | Low-moderate -- uses ERP and web applications |

### Persona 4: External Auditor -- Dr. Hofmann (Tertiary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm |
| **EUDR Pressure** | Must verify operator supply chain claims for regulatory audits |
| **Pain Points** | Operators provide incomplete, inconsistent supply chain documentation; no standardized way to verify traceability claims |
| **Goals** | Access read-only supply chain graphs with provenance hashes; verify plot-to-product traceability; validate risk assessment completeness |
| **Technical Skill** | Moderate -- comfortable with audit software and document review |

---

## 5. Regulatory Requirements

### 5.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 2(1-3)** | Definitions of "deforestation", "deforestation-free", "forest degradation" | Deforestation status tracked at plot nodes; integrated with AGENT-DATA-007 satellite verification against Dec 31, 2020 cutoff |
| **Art. 2(13-14)** | Definitions of "operator" and "trader" -- entities placing/making available products on EU market | Node types (Importer, Trader) mapped to EUDR operator/trader roles with role-based obligations |
| **Art. 2(17)** | "Traceability" definition -- ability to trace the relevant commodities and products through all stages of production, processing, and distribution | Full supply chain graph from plot to point-of-entry |
| **Art. 2(30-32)** | Definitions of "plot of land", "geolocation" and "geo-localisation" | Plot nodes with GPS coordinates and polygon boundaries per geolocation specification |
| **Art. 3** | Prohibition on non-compliant products | Gap analysis engine identifies non-traceable products |
| **Art. 4(2)** | Due diligence -- collect information on supply chain | Multi-tier mapping with automated discovery |
| **Art. 4(2)(f)** | Information on the supply chain of the relevant commodity or product | Graph export for DDS submission |
| **Art. 9(1)(a-d)** | Geolocation of all plots of land | Plot-level GPS/polygon integration via AGENT-DATA-005 |
| **Art. 9(1)(d)** | Polygon coordinates for plots > 4 hectares | Polygon validation engine |
| **Art. 10(1)** | Risk assessment -- assess and identify risk of non-compliance | Risk propagation through supply chain graph |
| **Art. 10(2)(a)** | Complexity of the relevant supply chain | Graph complexity metrics and analysis |
| **Art. 10(2)(e)** | Concerns about the country of production | Country-level risk scoring at plot nodes |
| **Art. 10(2)(f)** | Risk of circumvention or mixing with products of unknown origin | Batch mixing detection and mass balance verification |
| **Art. 11** | Risk mitigation measures | Risk-based supply chain segmentation and action planning |
| **Art. 12** | Submission of DDS to the EU Information System | Supply chain data formatted for DDS export |
| **Art. 29** | Country benchmarking (Low/Standard/High risk) | Country risk classification at graph nodes |
| **Art. 31** | Record keeping for 5 years | Immutable graph history with timestamp-based versioning |

### 5.2 Covered Commodities and Supply Chain Archetypes

| Commodity | Typical Supply Chain Depth | Key Actors | Mapping Complexity |
|-----------|---------------------------|------------|-------------------|
| **Cattle** | 4-6 tiers | Ranch -> Feedlot -> Slaughterhouse -> Packer -> Trader -> Importer | High (animal movement, pasture rotation) |
| **Cocoa** | 5-7 tiers | Smallholder -> Cooperative -> Collector -> Processor -> Trader -> Importer | Very High (thousands of smallholders per cooperative) |
| **Coffee** | 4-6 tiers | Farm -> Wet Mill -> Dry Mill -> Exporter -> Trader -> Importer | High (altitude/origin segregation) |
| **Palm Oil** | 4-5 tiers | Plantation -> Mill -> Refinery -> Trader -> Importer | High (RSPO mass balance challenges) |
| **Rubber** | 4-6 tiers | Smallholder -> Collector -> Processor -> Trader -> Importer | High (latex aggregation destroys traceability) |
| **Soya** | 4-5 tiers | Farm -> Silo -> Crusher -> Trader -> Importer | Medium-High (large volumes, co-mingling) |
| **Wood** | 5-8 tiers | Forest -> Sawmill -> Veneer/Plywood -> Furniture -> Trader -> Importer | Very High (multi-step processing, species mixing) |

### 5.3 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Baseline date for all deforestation-free verification |
| June 29, 2023 | Regulation entered into force | Legal basis for all compliance checks |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | Operators must have supply chain mapping operational |
| June 30, 2026 | Enforcement for SMEs | SME onboarding wave; agent must handle scale |
| Ongoing (quarterly) | Country benchmarking updates by EC | Agent must consume and apply updated country risk lists |

---

## 6. Features and Requirements

### 6.1 Must-Have Features (P0 -- Launch Blockers)

All 9 features below are P0 launch blockers. The agent cannot ship without all 9 features operational. Features 1-6 form the core graph intelligence engine; Features 7-9 form the experience and integration layer that delivers that intelligence to users and regulators.

**P0 Features 1-6: Core Graph Intelligence Engine**

---

#### Feature 1: Supply Chain Graph Engine

**User Story:**
```
As a compliance officer,
I want to model my entire supply chain as an interactive graph with nodes and edges,
So that I can see every actor, transfer, and plot involved in bringing a product to the EU market.
```

**Acceptance Criteria:**
- [ ] Models supply chain actors as typed nodes: Producer, Collector, Processor, Trader, Importer, Certifier
- [ ] Models custody transfers as directed edges with commodity, quantity, date, batch number
- [ ] Models production plots as leaf nodes with GPS coordinates and polygon boundaries
- [ ] Models batch records with split/merge relationships
- [ ] Supports directed acyclic graph (DAG) topology with cycle detection
- [ ] Supports graph operations: add_node, add_edge, remove_node, remove_edge, update_attributes
- [ ] Maintains graph versioning with immutable snapshots for audit trail
- [ ] Supports graph serialization to/from JSON, GraphML, and internal binary format
- [ ] Handles graphs with 100,000+ nodes without performance degradation
- [ ] Provides topological sorting for processing order determination

**Non-Functional Requirements:**
- Performance: Graph construction < 5 seconds for 10,000 nodes; single-node lookup < 1ms
- Memory: < 2 GB for 100,000-node graph
- Reproducibility: Deterministic graph construction from same input data
- Auditability: Every graph mutation recorded with timestamp and actor

**Dependencies:**
- NetworkX or custom graph library for in-memory graph operations
- AGENT-DATA-005 ChainOfCustodyEngine for custody transfer data
- AGENT-DATA-005 PlotRegistryEngine for plot geolocation data

**Estimated Effort:** 3 weeks (1 senior backend engineer)

**Edge Cases:**
- Circular supply chains (A supplies B supplies A) -- Detect and flag as error
- Orphan nodes (actors with no edges) -- Flag for review
- Duplicate node detection (same operator appearing under different IDs) -- Fuzzy match and merge

---

#### Feature 2: Multi-Tier Recursive Mapping

**User Story:**
```
As a supply chain analyst,
I want the system to automatically discover and map suppliers beyond my direct Tier 1 relationships,
So that I can achieve the full supply chain visibility required by EUDR Article 4(2).
```

**Acceptance Criteria:**
- [ ] Maps Tier 1 suppliers from operator's own procurement records (ERP/CSV)
- [ ] Maps Tier 2+ suppliers from supplier declarations (AGENT-DATA-008)
- [ ] Maps sub-tier relationships from supplier questionnaire responses
- [ ] Supports manual entry for tiers where automated discovery is not possible
- [ ] Recursively builds graph depth: Tier 1 -> Tier 2 -> ... -> Tier N -> Plot
- [ ] Tracks mapping completeness per tier (% of expected suppliers mapped)
- [ ] Identifies "opaque" segments where sub-tier visibility is missing
- [ ] Supports incremental mapping (add new tiers to existing graph without rebuilding)
- [ ] Handles the 7 commodity-specific supply chain archetypes described in Section 5.2
- [ ] Generates tier-depth report showing distribution of supply chain depth

**Non-Functional Requirements:**
- Completeness: Map at least 4 tiers deep for 80%+ of supply chains within 30 days
- Latency: Recursive discovery completes within 10 minutes for 1,000 Tier 1 suppliers
- Data Quality: Flag unverified tiers with confidence scores

**Dependencies:**
- AGENT-DATA-003 ERP/Finance Connector for Tier 1 procurement data
- AGENT-DATA-008 Supplier Questionnaire Processor for sub-tier declarations
- AGENT-DATA-001 PDF Invoice Extractor for custody documents
- AGENT-DATA-002 Excel/CSV Normalizer for bulk supplier imports

**Estimated Effort:** 4 weeks (1 senior backend engineer, 1 data engineer)

---

#### Feature 3: Plot-Level Geolocation Integration

**User Story:**
```
As a compliance officer,
I want every supply chain endpoint to be linked to a specific production plot with GPS coordinates,
So that I can fulfill EUDR Article 9 geolocation requirements in my Due Diligence Statement.
```

**Acceptance Criteria:**
- [ ] Links supply chain producer nodes to registered plots in the PlotRegistryEngine
- [ ] Validates plot geolocation: WGS84 coordinates within valid ranges
- [ ] Enforces polygon requirement for plots > 4 hectares per Article 9(1)(d)
- [ ] Displays plot locations on interactive map within supply chain graph view
- [ ] Calculates distance metrics between supply chain nodes for logistics validation
- [ ] Flags producers without registered plots as "geolocation missing"
- [ ] Supports bulk import of plot geolocation data from CSV, GeoJSON, and Shapefile
- [ ] Integrates with AGENT-DATA-006 GIS/Mapping Connector for spatial operations
- [ ] Cross-references plot locations against protected area boundaries
- [ ] Cross-references plot locations against deforestation satellite alerts (AGENT-DATA-007)

**Non-Functional Requirements:**
- Precision: Coordinates stored with 6+ decimal places (approximately 0.11m accuracy)
- Spatial Indexing: Plot lookups within a bounding box < 100ms for 100,000 plots
- Compliance: 100% of plots > 4 ha must have polygon data

**Dependencies:**
- AGENT-DATA-005 PlotRegistryEngine (existing -- production ready)
- AGENT-DATA-006 GIS/Mapping Connector (existing -- production ready)
- AGENT-DATA-007 Deforestation Satellite Connector (existing -- production ready)
- PostGIS extension for spatial queries

**Estimated Effort:** 2 weeks (1 backend engineer, 1 GIS specialist)

---

#### Feature 4: Many-to-Many Batch Traceability

**User Story:**
```
As a supply chain analyst,
I want to trace every final product back to all contributing production plots,
even when commodities have been aggregated, split, and re-aggregated through multiple processors,
So that I can maintain EUDR-compliant traceability through complex batch transformations.
```

**Acceptance Criteria:**
- [ ] Models three chain of custody models: Identity Preserved, Segregated, Mass Balance
- [ ] Tracks batch splitting: one input batch becomes N output batches with quantity conservation
- [ ] Tracks batch merging: N input batches become one output batch with origin preservation
- [ ] Tracks batch transformation: input commodity transforms to derived product (e.g., cocoa beans -> chocolate)
- [ ] Maintains deterministic mass balance: total output quantity <= total input quantity (within tolerance)
- [ ] Preserves origin plot linkage through unlimited levels of splits and merges
- [ ] Generates forward trace: "Which products contain commodity from Plot X?"
- [ ] Generates backward trace: "Which plots contributed to Product Y?"
- [ ] Flags mass balance discrepancies (output > input) as compliance alerts
- [ ] Supports partial traceability scoring when some origins are unknown (mass balance model)

**Non-Functional Requirements:**
- Accuracy: Mass balance verification uses Decimal arithmetic (no floating-point drift)
- Performance: Full backward trace < 2 seconds for chains with 50 split/merge operations
- Auditability: SHA-256 provenance hash on every batch operation

**Dependencies:**
- AGENT-DATA-005 ChainOfCustodyEngine (existing -- production ready)
- AGENT-DATA-005 BatchRecord model (existing)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

#### Feature 5: Risk Propagation Engine

**User Story:**
```
As a compliance officer,
I want risk scores from upstream suppliers and plots to automatically propagate downstream through the supply chain,
So that I can identify which final products inherit high-risk signals and require enhanced due diligence.
```

**Acceptance Criteria:**
- [ ] Propagates country risk (Article 29 benchmarking: Low/Standard/High) from plot nodes downstream
- [ ] Propagates commodity risk based on deforestation association for each commodity type
- [ ] Propagates supplier risk based on compliance history, certifications, and declarations
- [ ] Propagates deforestation risk based on satellite verification results from AGENT-DATA-007
- [ ] Calculates composite risk score at every graph node using deterministic weighted formula
- [ ] Applies "highest risk wins" principle: a product is only as safe as its riskiest input
- [ ] Identifies risk concentration: which suppliers/plots drive the most downstream risk
- [ ] Generates risk heatmap overlaying supply chain graph
- [ ] Triggers enhanced due diligence requirements when risk exceeds configurable threshold
- [ ] All risk calculations are deterministic (same input -> same output), with no LLM involvement

**Risk Calculation Formula:**
```
Node_Risk = max(
    Inherited_Risk_from_Parents,
    Own_Country_Risk * W_country,
    Own_Commodity_Risk * W_commodity,
    Own_Supplier_Risk * W_supplier,
    Own_Deforestation_Risk * W_deforestation
)

Where:
- W_country = 0.30 (configurable)
- W_commodity = 0.20 (configurable)
- W_supplier = 0.25 (configurable)
- W_deforestation = 0.25 (configurable)
- Inherited_Risk = max(risk of all parent nodes)
```

**Non-Functional Requirements:**
- Determinism: Bit-perfect reproducibility across runs
- Performance: Full graph risk propagation < 3 seconds for 10,000-node graph
- Configurability: Risk weights adjustable per operator without code changes

**Dependencies:**
- AGENT-DATA-005 RiskAssessmentEngine (existing -- production ready)
- AGENT-DATA-007 Deforestation Satellite Connector for deforestation risk inputs

**Estimated Effort:** 2 weeks (1 backend engineer)

---

#### Feature 6: Supply Chain Gap Analysis

**User Story:**
```
As a compliance officer,
I want the system to automatically identify gaps and weaknesses in my supply chain mapping,
So that I can prioritize remediation efforts and achieve full EUDR compliance.
```

**Acceptance Criteria:**
- [ ] Detects missing tiers (e.g., producer -> ??? -> processor) where intermediaries are unknown
- [ ] Detects unverified actors (nodes without compliance status or identity verification)
- [ ] Detects missing geolocation (producer nodes without GPS coordinates)
- [ ] Detects missing polygon data for plots > 4 hectares
- [ ] Detects broken custody chains (products with no traceable link to origin plots)
- [ ] Detects missing documentation (nodes without custody transfer records)
- [ ] Detects mass balance discrepancies (output exceeding input quantities)
- [ ] Generates compliance readiness score per commodity supply chain (0-100)
- [ ] Generates prioritized remediation action list sorted by risk impact
- [ ] Tracks gap closure over time with trend reporting

**Gap Severity Classification:**
| Gap Type | Severity | EUDR Article | Auto-Remediation |
|----------|----------|-------------|-----------------|
| Missing producer geolocation | Critical | Art. 9 | Trigger supplier questionnaire |
| Missing polygon for > 4 ha plot | Critical | Art. 9(1)(d) | Flag for GIS team |
| Broken custody chain | Critical | Art. 4(2)(f) | Trigger supplier investigation |
| Unverified intermediary | High | Art. 10 | Send verification request |
| Missing tier (opaque segment) | High | Art. 4(2) | Trigger sub-tier discovery |
| Mass balance discrepancy | High | Art. 10(2)(f) | Flag for manual review |
| Missing certification | Medium | Art. 10 | Request certification upload |
| Stale data (> 12 months) | Medium | Art. 31 | Trigger data refresh |

**Non-Functional Requirements:**
- Detection: Identify 95%+ of gaps (measured against manual audit baseline)
- Speed: Full gap analysis < 30 seconds for 10,000-node graph
- Reporting: Export gap analysis as PDF, CSV, or JSON

**Dependencies:**
- Features 1-5 (graph engine, multi-tier mapping, geolocation, batch traceability, risk)

**Estimated Effort:** 2 weeks (1 backend engineer)

---

**P0 Features 7-9: Experience and Integration Layer**

> Features 7, 8, and 9 are P0 launch blockers. Without visualization, onboarding, and regulatory export, the core graph engine cannot deliver end-user value. These features are the delivery mechanism through which compliance officers, analysts, and procurement managers interact with the graph engine.

---

#### Feature 7: Interactive Supply Chain Visualization

**User Story:**
```
As a compliance officer,
I want an interactive visual map of my supply chain that shows every actor, flow, and risk level,
So that I can quickly understand the structure, identify problems, and communicate to stakeholders.
```

**Acceptance Criteria:**
- [ ] Renders supply chain as interactive force-directed graph with zoom, pan, and node selection
- [ ] Color-codes nodes by risk level (green = low, yellow = standard, red = high)
- [ ] Color-codes edges by compliance status (solid = verified, dashed = pending, red = broken)
- [ ] Displays geographic overlay: nodes positioned on world map at their country/GPS location
- [ ] Supports Sankey diagram view showing commodity flow volumes between actors
- [ ] Supports drill-down: click a node to see its details, sub-graph, and risk factors
- [ ] Supports filtering: by commodity, country, risk level, compliance status, tier depth
- [ ] Supports time-slider: view the supply chain as it existed at any historical date
- [ ] Exports visualizations as PNG, SVG, and PDF for audit documentation
- [ ] Renders on desktop and tablet browsers (minimum 1024px viewport)

**Non-Functional Requirements:**
- Performance: Render 1,000-node graph in < 3 seconds; 10,000 nodes with virtualization
- Accessibility: WCAG 2.1 AA compliance for color-blind users (pattern-based differentiation)
- Responsiveness: Interactive updates (filter, zoom) < 200ms

**Dependencies:**
- Frontend framework: React (existing GL-EUDR-APP frontend)
- Graph visualization library: D3.js or vis-network or Cytoscape.js
- Map library: Leaflet (existing in GL-EUDR-APP)
- Backend: Graph API endpoints from Feature 1

**Estimated Effort:** 4 weeks (1 frontend engineer, 1 UX designer)

---

#### Feature 8: Supplier Onboarding and Discovery Workflow

**User Story:**
```
As a procurement manager,
I want a structured workflow to onboard new suppliers with EUDR-required information,
So that I can efficiently collect the supply chain and geolocation data needed for compliance.
```

**Acceptance Criteria:**
- [ ] Provides multi-step onboarding wizard: company info -> commodities -> plots -> certifications -> declarations
- [ ] Generates unique supplier onboarding link for self-service data entry
- [ ] Validates submitted data against EUDR requirements in real-time
- [ ] Captures plot GPS coordinates via mobile-friendly form (smartphone GPS capture)
- [ ] Requests sub-tier supplier information as part of onboarding
- [ ] Tracks onboarding completion status per supplier (% fields completed)
- [ ] Sends automated reminders for incomplete onboarding
- [ ] Auto-creates supply chain graph nodes and edges from onboarding data
- [ ] Supports bulk supplier import from CSV/Excel for existing supplier databases

**Non-Functional Requirements:**
- Completion Rate: Target 70%+ onboarding completion within 14 days of invitation
- Mobile: GPS coordinate capture works on iOS and Android mobile browsers
- Multi-language: Support English, French, German, Spanish, Portuguese, Indonesian

**Dependencies:**
- AGENT-DATA-008 Supplier Questionnaire Processor
- GL-EUDR-APP Supplier Management module
- Email notification service

**Estimated Effort:** 3 weeks (1 backend engineer, 1 frontend engineer)

---

#### Feature 9: Regulatory Export and DDS Integration

**User Story:**
```
As a compliance officer,
I want to export supply chain data in the exact format required for my Due Diligence Statement,
So that I can submit a complete, valid DDS to the EU Information System.
```

**Acceptance Criteria:**
- [ ] Exports supply chain data in EU Information System DDS format (JSON/XML)
- [ ] Includes all Article 4(2) required fields: operator info, product details, geolocation references, supply chain nodes
- [ ] Generates supply chain summary section for DDS with node counts, tier depth, and traceability score
- [ ] Validates export against EU DDS schema before submission
- [ ] Links to existing DDS generation workflow in GL-EUDR-APP
- [ ] Supports batch export for multiple products/shipments
- [ ] Generates audit-ready supply chain documentation (PDF report)
- [ ] Includes provenance hashes (SHA-256) for data integrity verification

**Non-Functional Requirements:**
- Compliance: 100% schema validation pass rate against EU DDS spec
- Completeness: All required fields populated or flagged as missing

**Dependencies:**
- AGENT-DATA-005 EUSystemConnector
- GL-EUDR-APP DDS Reporting Engine
- EU EUDR Information System DDS schema specification

**Estimated Effort:** 2 weeks (1 backend engineer)

---

### 6.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 10: Supply Chain Simulation and What-If Analysis
- Model impact of losing a supplier (which products are affected?)
- Model impact of a country risk reclassification
- Simulate batch re-routing through alternative supply chains
- Estimate cost and time impact of supply chain changes

#### Feature 11: AI-Assisted Anomaly Detection
- Detect unusual patterns in custody transfers (volume spikes, new routes)
- Detect potential circumvention (re-routing through low-risk countries)
- Detect data quality anomalies (inconsistent quantities, impossible timelines)

#### Feature 12: Multi-Commodity Blended Product Mapping
- Model products containing multiple EUDR commodities (e.g., chocolate bar with cocoa + soya lecithin + palm oil)
- Track individual commodity traceability within blended products
- Calculate aggregate risk across commodity components

---

### 6.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- Carbon footprint calculation per supply chain path (defer to GL-GHG-APP integration)
- Blockchain-based immutable ledger (SHA-256 provenance hashes provide sufficient integrity)
- Real-time IoT sensor integration for live shipment tracking
- Predictive analytics for future supply chain disruptions
- Mobile native application (web responsive design only for v1.0)
- Automated supplier payment management

---

## 7. Technical Requirements

### 7.1 Architecture Overview

```
                                    +---------------------------+
                                    |     GL-EUDR-APP v1.0      |
                                    |   Frontend (React/TS)     |
                                    +-------------+-------------+
                                                  |
                                    +-------------v-------------+
                                    |     Unified API Layer      |
                                    |       (FastAPI)            |
                                    +-------------+-------------+
                                                  |
            +-------------------------------------+-------------------------------------+
            |                                     |                                     |
+-----------v-----------+           +-------------v-------------+           +-----------v-----------+
| AGENT-EUDR-001        |           | AGENT-DATA-005            |           | AGENT-DATA-007        |
| Supply Chain Mapping  |<--------->| EUDR Traceability         |<--------->| Deforestation         |
| Master                |           | Connector                 |           | Satellite Connector   |
|                       |           |                           |           |                       |
| - Graph Engine        |           | - PlotRegistryEngine      |           | - Sentinel-2 Client   |
| - Multi-Tier Mapper   |           | - ChainOfCustodyEngine    |           | - Landsat Client      |
| - Risk Propagation    |           | - DueDiligenceEngine      |           | - GFW Client          |
| - Gap Analysis        |           | - RiskAssessmentEngine    |           | - NDVI Calculator     |
| - Visualization API   |           | - CommodityClassifier     |           | - Forest Change Det.  |
+-----------+-----------+           | - ComplianceVerifier      |           +-----------------------+
            |                       | - EUSystemConnector       |
            |                       +---------------------------+
            |
+-----------v-----------+           +---------------------------+           +---------------------------+
| AGENT-DATA-006        |           | AGENT-DATA-003            |           | AGENT-DATA-008            |
| GIS/Mapping Connector |           | ERP/Finance Connector     |           | Supplier Questionnaire    |
|                       |           |                           |           | Processor                 |
| - PostGIS Queries     |           | - SAP Integration         |           | - Sub-tier Discovery      |
| - Spatial Indexing    |           | - Oracle Integration      |           | - Declaration Processing  |
| - Protected Areas     |           | - Procurement Records     |           | - Validation              |
+-----------------------+           +---------------------------+           +---------------------------+
```

### 7.2 Module Structure

```
greenlang/eudr_supply_chain_mapper/
    __init__.py                          # Public API exports
    config.py                            # SupplyChainMapperConfig with GL_EUDR_SCM_ env prefix (dataclass, thread-safe singleton)
    models.py                            # Pydantic v2 models for graph nodes, edges, queries
    graph_engine.py                      # SupplyChainGraphEngine: core graph operations
    multi_tier_mapper.py                 # MultiTierMapper: recursive supply chain discovery
    geolocation_linker.py                # GeolocationLinker: plot-to-node linkage
    batch_traceability.py                # BatchTraceabilityEngine: split/merge/trace
    risk_propagation.py                  # RiskPropagationEngine: graph risk traversal
    gap_analyzer.py                      # GapAnalyzer: compliance gap detection
    visualization_engine.py              # VisualizationEngine: graph layout and export
    regulatory_exporter.py               # RegulatoryExporter: DDS format export
    supplier_onboarding.py               # SupplierOnboardingEngine: onboarding workflow
    provenance.py                        # ProvenanceTracker: SHA-256 hash chains
    metrics.py                           # 15 Prometheus self-monitoring metrics
    setup.py                             # SupplyChainMapperService facade
    api/
        __init__.py
        router.py                        # FastAPI router (25+ endpoints)
        graph_routes.py                  # Graph CRUD and query endpoints
        mapping_routes.py                # Multi-tier mapping endpoints
        traceability_routes.py           # Batch traceability endpoints
        risk_routes.py                   # Risk propagation endpoints
        gap_routes.py                    # Gap analysis endpoints
        visualization_routes.py          # Visualization and export endpoints
        onboarding_routes.py             # Supplier onboarding endpoints
```

### 7.3 Data Models (Key Entities)

```python
# Supply Chain Node Types
class NodeType(str, Enum):
    PRODUCER = "producer"        # Farm, plantation, forest concession
    COLLECTOR = "collector"      # Cooperative, aggregation point, silo
    PROCESSOR = "processor"      # Mill, refinery, slaughterhouse, sawmill
    TRADER = "trader"            # Trading company, intermediary
    IMPORTER = "importer"        # EU-based operator placing on market
    CERTIFIER = "certifier"      # Certification body
    WAREHOUSE = "warehouse"      # Storage and logistics
    PORT = "port"                # Port of loading / unloading

# Supply Chain Node
class SupplyChainNode(BaseModel):
    node_id: str                 # Unique identifier
    node_type: NodeType          # Actor role in supply chain
    operator_id: str             # Reference to operator/company ID
    operator_name: str           # Human-readable name
    country_code: str            # ISO 3166-1 alpha-2
    region: Optional[str]        # Sub-national region
    coordinates: Optional[Tuple[float, float]]  # (lat, lon)
    commodities: List[EUDRCommodity]  # Commodities handled
    tier_depth: int              # Distance from importer (0 = importer)
    risk_score: float            # Composite risk score (0-100)
    risk_level: RiskLevel        # LOW/STANDARD/HIGH
    compliance_status: ComplianceStatus
    certifications: List[str]    # FSC, RSPO, etc.
    plot_ids: List[str]          # Linked production plot IDs (producers only)
    metadata: Dict[str, Any]

# Supply Chain Edge (Custody Transfer)
class SupplyChainEdge(BaseModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    commodity: EUDRCommodity
    product_description: str
    quantity: Decimal
    unit: str
    batch_number: Optional[str]
    custody_model: CustodyModel  # IP/Segregated/MassBalance
    transfer_date: datetime
    cn_code: Optional[str]
    hs_code: Optional[str]
    transport_mode: Optional[str]
    provenance_hash: str         # SHA-256

# Supply Chain Graph
class SupplyChainGraph(BaseModel):
    graph_id: str
    operator_id: str             # Owner of this supply chain view
    commodity: EUDRCommodity     # Primary commodity
    nodes: Dict[str, SupplyChainNode]
    edges: Dict[str, SupplyChainEdge]
    total_nodes: int
    total_edges: int
    max_tier_depth: int
    traceability_score: float    # 0-100
    compliance_readiness: float  # 0-100
    risk_summary: Dict[str, int] # Count by risk level
    gaps: List[SupplyChainGap]
    created_at: datetime
    updated_at: datetime
    version: int                 # Incremented on every mutation
```

### 7.4 Database Schema (New Migration: V089)

```sql
CREATE SCHEMA IF NOT EXISTS eudr_supply_chain_mapper;

-- Supply chain graphs (one per operator per commodity)
CREATE TABLE eudr_supply_chain_mapper.supply_chain_graphs (
    graph_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id UUID NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    graph_name VARCHAR(500),
    total_nodes INTEGER DEFAULT 0,
    total_edges INTEGER DEFAULT 0,
    max_tier_depth INTEGER DEFAULT 0,
    traceability_score NUMERIC(5,2) DEFAULT 0.0,
    compliance_readiness NUMERIC(5,2) DEFAULT 0.0,
    risk_summary JSONB DEFAULT '{}',
    version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_operator FOREIGN KEY (operator_id) REFERENCES auth.users(id)
);

-- Supply chain nodes (actors in the supply chain)
CREATE TABLE eudr_supply_chain_mapper.supply_chain_nodes (
    node_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id UUID NOT NULL REFERENCES eudr_supply_chain_mapper.supply_chain_graphs(graph_id),
    node_type VARCHAR(50) NOT NULL,
    operator_id VARCHAR(100),
    operator_name VARCHAR(500) NOT NULL,
    country_code CHAR(2) NOT NULL,
    region VARCHAR(200),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    commodities JSONB DEFAULT '[]',
    tier_depth INTEGER DEFAULT 0,
    risk_score NUMERIC(5,2) DEFAULT 0.0,
    risk_level VARCHAR(20) DEFAULT 'standard',
    compliance_status VARCHAR(50) DEFAULT 'pending_verification',
    certifications JSONB DEFAULT '[]',
    plot_ids JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Supply chain edges (custody transfers between nodes)
CREATE TABLE eudr_supply_chain_mapper.supply_chain_edges (
    edge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id UUID NOT NULL REFERENCES eudr_supply_chain_mapper.supply_chain_graphs(graph_id),
    source_node_id UUID NOT NULL REFERENCES eudr_supply_chain_mapper.supply_chain_nodes(node_id),
    target_node_id UUID NOT NULL REFERENCES eudr_supply_chain_mapper.supply_chain_nodes(node_id),
    commodity VARCHAR(50) NOT NULL,
    product_description VARCHAR(1000),
    quantity NUMERIC(18,4) NOT NULL,
    unit VARCHAR(20) DEFAULT 'kg',
    batch_number VARCHAR(100),
    custody_model VARCHAR(30) DEFAULT 'segregated',
    transfer_date TIMESTAMPTZ,
    cn_code VARCHAR(20),
    hs_code VARCHAR(20),
    transport_mode VARCHAR(50),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Supply chain gap analysis results (hypertable)
CREATE TABLE eudr_supply_chain_mapper.gap_analysis_results (
    analysis_id UUID DEFAULT gen_random_uuid(),
    graph_id UUID NOT NULL,
    gap_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    affected_node_id UUID,
    affected_edge_id UUID,
    description TEXT NOT NULL,
    remediation TEXT,
    eudr_article VARCHAR(20),
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_supply_chain_mapper.gap_analysis_results', 'detected_at');

-- Risk propagation audit log (hypertable)
CREATE TABLE eudr_supply_chain_mapper.risk_propagation_log (
    log_id UUID DEFAULT gen_random_uuid(),
    graph_id UUID NOT NULL,
    node_id UUID NOT NULL,
    previous_risk_score NUMERIC(5,2),
    new_risk_score NUMERIC(5,2),
    previous_risk_level VARCHAR(20),
    new_risk_level VARCHAR(20),
    propagation_source VARCHAR(50),
    risk_factors JSONB DEFAULT '[]',
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_supply_chain_mapper.risk_propagation_log', 'calculated_at');

-- Graph version snapshots for audit trail
CREATE TABLE eudr_supply_chain_mapper.graph_snapshots (
    snapshot_id UUID DEFAULT gen_random_uuid(),
    graph_id UUID NOT NULL,
    version INTEGER NOT NULL,
    snapshot_data JSONB NOT NULL,
    provenance_hash VARCHAR(64) NOT NULL,
    created_by VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_supply_chain_mapper.graph_snapshots', 'created_at');

-- Indexes
CREATE INDEX idx_nodes_graph_id ON eudr_supply_chain_mapper.supply_chain_nodes(graph_id);
CREATE INDEX idx_nodes_country ON eudr_supply_chain_mapper.supply_chain_nodes(country_code);
CREATE INDEX idx_nodes_type ON eudr_supply_chain_mapper.supply_chain_nodes(node_type);
CREATE INDEX idx_nodes_risk ON eudr_supply_chain_mapper.supply_chain_nodes(risk_level);
CREATE INDEX idx_edges_graph_id ON eudr_supply_chain_mapper.supply_chain_edges(graph_id);
CREATE INDEX idx_edges_source ON eudr_supply_chain_mapper.supply_chain_edges(source_node_id);
CREATE INDEX idx_edges_target ON eudr_supply_chain_mapper.supply_chain_edges(target_node_id);
CREATE INDEX idx_edges_commodity ON eudr_supply_chain_mapper.supply_chain_edges(commodity);
CREATE INDEX idx_gaps_graph_id ON eudr_supply_chain_mapper.gap_analysis_results(graph_id);
CREATE INDEX idx_snapshots_graph_id ON eudr_supply_chain_mapper.graph_snapshots(graph_id);
```

### 7.5 API Endpoints (25+)

| Method | Path | Description |
|--------|------|-------------|
| **Graph Management** | | |
| POST | `/v1/graphs` | Create a new supply chain graph for an operator/commodity |
| GET | `/v1/graphs` | List supply chain graphs (with filters) |
| GET | `/v1/graphs/{graph_id}` | Get graph details with summary statistics |
| DELETE | `/v1/graphs/{graph_id}` | Archive a supply chain graph |
| GET | `/v1/graphs/{graph_id}/export` | Export graph in GraphML/JSON-LD/GeoJSON format |
| **Node Operations** | | |
| POST | `/v1/graphs/{graph_id}/nodes` | Add a supply chain node |
| GET | `/v1/graphs/{graph_id}/nodes` | List nodes (with filters: type, country, risk) |
| GET | `/v1/graphs/{graph_id}/nodes/{node_id}` | Get node details with connections |
| PUT | `/v1/graphs/{graph_id}/nodes/{node_id}` | Update node attributes |
| DELETE | `/v1/graphs/{graph_id}/nodes/{node_id}` | Remove node (and its edges) |
| **Edge Operations** | | |
| POST | `/v1/graphs/{graph_id}/edges` | Add a custody transfer edge |
| GET | `/v1/graphs/{graph_id}/edges` | List edges (with filters: commodity, date range) |
| DELETE | `/v1/graphs/{graph_id}/edges/{edge_id}` | Remove an edge |
| **Multi-Tier Mapping** | | |
| POST | `/v1/graphs/{graph_id}/discover` | Trigger multi-tier discovery from Tier 1 |
| GET | `/v1/graphs/{graph_id}/tiers` | Get tier-depth distribution |
| **Traceability** | | |
| GET | `/v1/graphs/{graph_id}/trace/forward/{node_id}` | Forward trace from node |
| GET | `/v1/graphs/{graph_id}/trace/backward/{node_id}` | Backward trace to origins |
| GET | `/v1/graphs/{graph_id}/trace/batch/{batch_id}` | Trace batch to origin plots |
| **Risk** | | |
| POST | `/v1/graphs/{graph_id}/risk/propagate` | Run risk propagation |
| GET | `/v1/graphs/{graph_id}/risk/summary` | Get risk summary per graph |
| GET | `/v1/graphs/{graph_id}/risk/heatmap` | Get risk heatmap data |
| **Gap Analysis** | | |
| POST | `/v1/graphs/{graph_id}/gaps/analyze` | Run gap analysis |
| GET | `/v1/graphs/{graph_id}/gaps` | List detected gaps |
| PUT | `/v1/graphs/{graph_id}/gaps/{gap_id}/resolve` | Mark gap as resolved |
| **Visualization** | | |
| GET | `/v1/graphs/{graph_id}/layout` | Get graph layout for visualization |
| GET | `/v1/graphs/{graph_id}/sankey` | Get Sankey diagram data |
| **Regulatory** | | |
| GET | `/v1/graphs/{graph_id}/dds-export` | Export supply chain data for DDS |
| **Health** | | |
| GET | `/health` | Service health check |

### 7.6 Prometheus Self-Monitoring Metrics (15)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_scm_graphs_created_total` | Counter | Supply chain graphs created |
| 2 | `gl_eudr_scm_nodes_added_total` | Counter | Nodes added to graphs by type |
| 3 | `gl_eudr_scm_edges_added_total` | Counter | Edges added to graphs |
| 4 | `gl_eudr_scm_tier_discovery_total` | Counter | Multi-tier discovery operations |
| 5 | `gl_eudr_scm_trace_operations_total` | Counter | Forward/backward trace operations |
| 6 | `gl_eudr_scm_risk_propagations_total` | Counter | Risk propagation runs |
| 7 | `gl_eudr_scm_gaps_detected_total` | Counter | Gaps detected by type and severity |
| 8 | `gl_eudr_scm_gaps_resolved_total` | Counter | Gaps resolved |
| 9 | `gl_eudr_scm_dds_exports_total` | Counter | DDS export operations |
| 10 | `gl_eudr_scm_processing_duration_seconds` | Histogram | Processing operation latency by operation type |
| 11 | `gl_eudr_scm_graph_query_duration_seconds` | Histogram | Graph query latency |
| 12 | `gl_eudr_scm_errors_total` | Counter | Errors by operation type |
| 13 | `gl_eudr_scm_active_graphs` | Gauge | Currently active supply chain graphs |
| 14 | `gl_eudr_scm_total_nodes` | Gauge | Total nodes across all graphs |
| 15 | `gl_eudr_scm_compliance_readiness_avg` | Gauge | Average compliance readiness score |

### 7.7 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| Graph Library | NetworkX (in-memory) + PostgreSQL (persistence) | NetworkX for complex graph algorithms; PG for durable storage |
| Spatial | PostGIS + Shapely + GeoJSON | Spatial queries, polygon operations, GeoJSON export |
| Database | PostgreSQL + TimescaleDB | Persistent storage + time-series hypertables |
| Cache | Redis | Graph query caching, node lookup caching |
| Object Storage | S3 | Graph snapshots, GeoJSON exports, visualization assets |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based graph access control |
| Monitoring | Prometheus + Grafana | 15 metrics + dedicated dashboard |
| Tracing | OpenTelemetry | Distributed tracing across agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment |

### 7.8 RBAC Permissions (SEC-002 Integration)

The following permissions will be registered in the GreenLang PERMISSION_MAP for RBAC enforcement:

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-scm:graphs:read` | View supply chain graphs | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-scm:graphs:write` | Create, update, archive graphs | Analyst, Compliance Officer, Admin |
| `eudr-scm:nodes:read` | View supply chain nodes | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-scm:nodes:write` | Add, update, remove nodes | Analyst, Compliance Officer, Admin |
| `eudr-scm:edges:read` | View custody transfer edges | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-scm:edges:write` | Add, remove edges | Analyst, Compliance Officer, Admin |
| `eudr-scm:mapping:execute` | Trigger multi-tier discovery | Analyst, Compliance Officer, Admin |
| `eudr-scm:traceability:read` | Execute forward/backward traces | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-scm:risk:read` | View risk scores and heatmaps | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-scm:risk:execute` | Trigger risk propagation runs | Analyst, Compliance Officer, Admin |
| `eudr-scm:gaps:read` | View gap analysis results | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-scm:gaps:execute` | Trigger gap analysis runs | Analyst, Compliance Officer, Admin |
| `eudr-scm:gaps:resolve` | Mark gaps as resolved | Compliance Officer, Admin |
| `eudr-scm:visualization:read` | Access graph layouts and Sankey data | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-scm:export:dds` | Export supply chain data for DDS | Compliance Officer, Admin |
| `eudr-scm:export:graph` | Export graphs (GraphML/GeoJSON/JSON-LD) | Analyst, Compliance Officer, Admin |
| `eudr-scm:onboarding:manage` | Manage supplier onboarding workflows | Procurement Manager, Compliance Officer, Admin |
| `eudr-scm:audit:read` | View graph snapshots and audit trail | Auditor (read-only), Compliance Officer, Admin |

### 7.9 Integration Points

#### Upstream Dependencies (Data Sources)

| Agent | Integration | Data Flow |
|-------|-------------|-----------|
| AGENT-DATA-005 EUDR Traceability | PlotRegistryEngine, ChainOfCustodyEngine | Plot data, custody transfers, batch records -> graph nodes/edges |
| AGENT-DATA-003 ERP/Finance Connector | Procurement records | Tier 1 supplier data, purchase orders -> graph nodes |
| AGENT-DATA-008 Supplier Questionnaire | Sub-tier declarations | Tier 2+ supplier data -> graph nodes |
| AGENT-DATA-001 PDF Invoice Extractor | Custody documents | Transfer records from scanned documents -> graph edges |
| AGENT-DATA-002 Excel/CSV Normalizer | Bulk imports | Supplier lists, plot registries -> graph nodes |
| AGENT-DATA-006 GIS/Mapping Connector | Spatial operations | Protected area intersections, distance calculations |
| AGENT-DATA-007 Deforestation Satellite | Deforestation risk | Satellite-derived deforestation alerts -> risk scores on nodes |
| AGENT-FOUND-005 Citations | Regulatory references | EUDR article citations for compliance checks |
| AGENT-FOUND-008 Reproducibility | Determinism verification | Bit-perfect verification of risk propagation |

#### Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| GL-EUDR-APP v1.0 | API integration | Graph data -> frontend visualization, DDS generation |
| AGENT-DATA-005 DueDiligenceEngine | DDS supply chain section | Supply chain summary for DDS export |
| AGENT-DATA-005 EUSystemConnector | EU submission | Formatted supply chain data for EU Information System |
| GL-CSDDD-APP (future) | Supply chain due diligence | Reuse supply chain graph for CSDDD compliance |
| External Auditors | Read-only API + exports | Graph exports for third-party verification |

---

## 8. User Experience

### 8.1 User Flows

#### Flow 1: Initial Supply Chain Mapping (Compliance Officer)

```
1. Compliance officer logs in to GL-EUDR-APP
2. Navigates to "Supply Chain Mapping" module
3. Clicks "New Supply Chain Map" -> selects commodity (e.g., Cocoa)
4. System imports Tier 1 suppliers from ERP data (AGENT-DATA-003)
   -> Graph shows importer node + Tier 1 supplier nodes
5. System sends supplier questionnaires to Tier 1 suppliers (AGENT-DATA-008)
6. As suppliers respond, Tier 2+ nodes are automatically added
7. System links producer nodes to registered plots (AGENT-DATA-005)
8. System runs risk propagation -> nodes color-coded by risk level
9. System runs gap analysis -> gaps highlighted with remediation actions
10. Compliance officer reviews, addresses gaps, reaches target readiness
11. Exports supply chain data for DDS generation
```

#### Flow 2: Batch Backward Trace (Supply Chain Analyst)

```
1. Analyst receives a batch of palm oil at EU port (Batch BATCH-2026-0342)
2. Opens "Traceability" view in Supply Chain Mapping module
3. Enters batch number -> system queries ChainOfCustodyEngine
4. System displays backward trace graph:
   Batch-0342 <- Refinery (MY) <- Mill-A (MY) <- Plantation-1 (MY)
                                <- Mill-B (MY) <- Plantation-2 (MY)
                                                <- Plantation-3 (ID)
5. Each plantation node shows GPS coordinates, polygon, deforestation status
6. Plantation-3 (ID) is flagged HIGH RISK (country = ID, no RSPO cert)
7. Analyst clicks node -> sees risk factors and mitigation recommendations
8. Analyst initiates enhanced due diligence workflow for Plantation-3
```

#### Flow 3: Gap Closure (Procurement Manager)

```
1. Gap analysis report shows: "15 producers missing GPS coordinates"
2. Procurement manager opens gap detail view
3. System shows list of 15 producers with missing data
4. Manager clicks "Send Onboarding Request" for each producer
5. Producers receive email with link to self-service GPS capture form
6. As producers submit coordinates, gaps auto-resolve
7. System updates graph nodes with new geolocation data
8. Compliance readiness score increases from 72% to 89%
```

### 8.2 Key Screen Descriptions

**Supply Chain Graph View:**
- Full-screen interactive graph with force-directed layout
- Left sidebar: filter panel (commodity, country, risk level, tier depth)
- Right sidebar: node/edge detail panel (appears on selection)
- Top bar: graph statistics (nodes, edges, depth, readiness score, gaps)
- Bottom bar: time slider for historical views
- Map toggle: switch between abstract graph and geographic overlay

**Gap Analysis Dashboard:**
- Summary cards: Critical gaps, High gaps, Medium gaps, Resolved gaps
- Grouped gap list: organized by gap type with severity badges
- Trend chart: gap count over time (closing trend)
- Remediation action queue: prioritized by risk impact

**Traceability View:**
- Search bar: enter batch number, product ID, or plot ID
- Trace result: animated path highlighting through the supply chain graph
- Origin panel: list of all contributing plots with map pins
- Mass balance panel: input/output quantity comparison with balance status

---

## 9. Success Criteria

### 9.1 Launch Criteria (Go/No-Go)

- [ ] All 9 P0 features (Features 1-9) implemented and tested
  - [ ] Feature 1: Supply Chain Graph Engine -- node/edge CRUD, DAG operations, serialization
  - [ ] Feature 2: Multi-Tier Recursive Mapping -- automated Tier 1 through Tier N discovery
  - [ ] Feature 3: Plot-Level Geolocation Integration -- Article 9 compliance for all plots
  - [ ] Feature 4: Many-to-Many Batch Traceability -- split/merge/trace with mass balance
  - [ ] Feature 5: Risk Propagation Engine -- deterministic graph-wide risk scoring
  - [ ] Feature 6: Supply Chain Gap Analysis -- detection, classification, remediation
  - [ ] Feature 7: Interactive Supply Chain Visualization -- graph rendering, geographic overlay
  - [ ] Feature 8: Supplier Onboarding and Discovery Workflow -- self-service portal
  - [ ] Feature 9: Regulatory Export and DDS Integration -- EU Information System format
- [ ] >= 85% test coverage achieved
- [ ] Security audit passed (JWT + RBAC integrated)
- [ ] Performance targets met (< 500ms graph query p99 for 10,000 nodes)
- [ ] All 7 commodity supply chain archetypes tested with golden test fixtures
- [ ] Risk propagation verified deterministic (bit-perfect reproducibility)
- [ ] Gap analysis validated against manually audited supply chains (>= 95% detection)
- [ ] API documentation complete (OpenAPI spec)
- [ ] Database migration V089 tested and validated
- [ ] Integration with AGENT-DATA-005, AGENT-DATA-006, AGENT-DATA-007 verified
- [ ] 5 beta customers successfully mapped their supply chains
- [ ] No critical or high-severity bugs in backlog

### 9.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 50+ supply chain graphs created by customers
- Average tier depth mapped >= 3
- Average compliance readiness >= 60%
- < 5 support tickets per customer
- p99 query latency < 500ms in production

**60 Days:**
- 200+ supply chain graphs active
- Average tier depth mapped >= 4
- Average compliance readiness >= 75%
- Gap detection precision >= 95% (validated by audit sample)
- 3+ commodities mapped per average customer

**90 Days:**
- 500+ supply chain graphs active
- 1,000+ multi-tier mappings completed
- Average compliance readiness >= 85%
- Zero EUDR penalties attributed to supply chain traceability gaps for active customers
- NPS > 50 from compliance officer persona

---

## 10. Timeline and Milestones

### Phase 1: Core Engine (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Graph Engine (Feature 1): node/edge CRUD, DAG operations, serialization | Backend Engineer |
| 2-3 | Batch Traceability (Feature 4): split/merge/trace with mass balance | Backend Engineer |
| 3-4 | Multi-Tier Mapper (Feature 2): recursive discovery, tier tracking | Data Engineer |
| 4-5 | Geolocation Linker (Feature 3): plot-to-node linkage, spatial validation | GIS Specialist |
| 5-6 | Risk Propagation (Feature 5): graph traversal risk scoring | Backend Engineer |

**Milestone: Core engine operational with all P0 graph operations (Week 6)**

### Phase 2: Analysis, API, and Regulatory Export (Weeks 7-10)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Gap Analysis (Feature 6): detection, severity classification, remediation | Backend Engineer |
| 8-9 | REST API Layer: 25+ endpoints, authentication, rate limiting | Backend Engineer |
| 9-10 | Regulatory Export (Feature 9, P0): DDS format, schema validation, EU Information System integration | Backend Engineer |

**Milestone: Full API operational with gap analysis and DDS export (Week 10)**

### Phase 3: Visualization, Onboarding, and Integration (Weeks 11-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 11-12 | Interactive Visualization (Feature 7, P0): graph rendering, geographic overlay, Sankey diagrams | Frontend Engineer |
| 12-13 | Supplier Onboarding (Feature 8, P0): wizard, self-service portal, GPS capture | Frontend + Backend |
| 13-14 | Visualization export (PNG/SVG/PDF), RBAC integration, end-to-end integration testing | Frontend Engineer + Backend Engineer |

**Milestone: All 9 P0 features implemented with full UI (Week 14)**

### Phase 4: Testing and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Complete test suite: 800+ tests, golden tests for all 7 commodities | Test Engineer |
| 16-17 | Performance testing, security audit, load testing | DevOps + Security |
| 17 | Database migration V089 finalized and tested | DevOps |
| 17-18 | Beta customer onboarding (5 customers) | Product + Engineering |
| 18 | Launch readiness review (all 9 P0 features verified) and go-live | All |

**Milestone: Production launch with all 9 P0 features (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- What-if simulation (Feature 10)
- AI anomaly detection (Feature 11)
- Multi-commodity blended product support (Feature 12)
- Performance optimization for 100K+ node graphs
- Additional commodity-specific supply chain templates

---

## 11. Dependencies

### 11.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| AGENT-DATA-005 EUDR Traceability Connector | BUILT (100%) | Low | Stable, production-ready |
| AGENT-DATA-006 GIS/Mapping Connector | BUILT (100%) | Low | Stable, production-ready |
| AGENT-DATA-007 Deforestation Satellite Connector | BUILT (100%) | Low | Stable, production-ready |
| AGENT-DATA-003 ERP/Finance Connector | BUILT (100%) | Low | Stable, production-ready |
| AGENT-DATA-008 Supplier Questionnaire Processor | BUILT (100%) | Low | Stable, production-ready |
| GL-EUDR-APP v1.0 Platform | BUILT (100%) | Low | Integration points defined |
| PostgreSQL + TimescaleDB + PostGIS | Production Ready | Low | Standard infrastructure |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration |

### 11.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EU Information System DDS schema | Published (v1.x) | Medium | Adapter pattern for schema version changes |
| EC country benchmarking list | Published; updated periodically | Medium | Database-driven, hot-reloadable country risk |
| EU EUDR implementing regulations | Evolving | Medium | Configuration-driven compliance rules |
| Satellite data providers (Sentinel-2, Landsat) | Available | Low | Multi-provider fallback in AGENT-DATA-007 |
| Global Forest Watch API | Available | Low | Cached data, fallback to offline datasets |

---

## 12. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | EU Information System API specification changes before or after launch | Medium | High | Adapter pattern isolates EU API layer; can update connector without touching graph engine |
| R2 | Suppliers unwilling to disclose sub-tier supply chain data | High | High | Graduated disclosure model; risk score increases for opaque segments; regulatory pressure from enforcement |
| R3 | Graph performance degrades for very large supply chains (> 100K nodes) | Low | Medium | Lazy loading, graph partitioning, PostgreSQL-backed pagination; benchmark continuously |
| R4 | Country benchmarking list updated by EC (new high-risk countries) | Medium | Medium | Database-driven country risk; hot-reload from EC publication feed; automated re-propagation |
| R5 | Complex batch mixing destroys traceability (mass balance model) | Medium | High | Implement all 3 custody models (IP/Segregated/Mass Balance); flag mass balance chains as partial traceability |
| R6 | EUDR regulation amended or delayed | Low | Medium | Modular design allows quick adaptation; core graph engine is regulation-agnostic |
| R7 | Integration complexity with multiple AGENT-DATA connectors | Medium | Medium | Well-defined interfaces; mock adapters for testing; circuit breaker pattern |
| R8 | Data quality issues in supplier-provided geolocation data | High | Medium | Validation engine rejects invalid coordinates; confidence scoring; satellite cross-verification |
| R9 | Low customer adoption of self-service supplier onboarding | Medium | Medium | Multi-language support; mobile-optimized GPS capture; reminder campaigns |
| R10 | Competitive tools launch before GreenLang reaches market | Medium | Medium | Deeper integration with GreenLang ecosystem; superior graph analytics; faster iteration |

---

## 13. Test Strategy

### 13.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Graph Engine Unit Tests | 150+ | Node/edge CRUD, DAG operations, serialization, cycle detection |
| Multi-Tier Mapping Tests | 80+ | Recursive discovery, tier tracking, incremental mapping |
| Batch Traceability Tests | 100+ | Split, merge, trace, mass balance, origin preservation |
| Risk Propagation Tests | 80+ | Deterministic scoring, inheritance, highest-risk-wins, edge cases |
| Gap Analysis Tests | 70+ | All gap types, severity classification, remediation suggestions |
| Geolocation Integration Tests | 60+ | Coordinate validation, polygon enforcement, spatial queries |
| API Tests | 80+ | All 25+ endpoints, auth, error handling, pagination |
| Visualization Tests | 40+ | Layout algorithms, Sankey data, export formats |
| Golden Tests | 50+ | All 7 commodities, complete/partial/broken chains (extend existing fixtures) |
| Integration Tests | 30+ | Cross-agent integration with DATA-005/006/007 |
| Performance Tests | 20+ | 10K/50K/100K node graphs, concurrent queries, risk propagation timing |
| **Total** | **800+** | |

### 13.2 Golden Test Commodities

Each of the 7 commodities will have a dedicated golden test supply chain with:
1. Complete chain (producer to importer) -- expect 100% traceability
2. Partial chain (missing intermediary) -- expect partial traceability with gaps detected
3. Broken chain (no producer) -- expect broken status with critical gaps
4. Many-to-many chain (multiple producers -> multiple processors -> importer) -- expect correct origin tracking
5. Batch split/merge chain -- expect mass balance verification and origin preservation
6. High-risk chain (high-risk country origin) -- expect risk propagation to all downstream nodes
7. Multi-tier chain (6+ tiers) -- expect correct tier depth and full traversal

Total: 7 commodities x 7 scenarios = 49 golden test scenarios

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **CN Code** | Combined Nomenclature -- EU product classification code |
| **HS Code** | Harmonized System -- international product classification code |
| **DAG** | Directed Acyclic Graph -- graph structure with directed edges and no cycles |
| **Tier** | Supply chain depth level (Tier 0 = importer, Tier 1 = direct supplier, etc.) |
| **Mass Balance** | Chain of custody model where compliant and non-compliant material may be mixed but quantities tracked |
| **Identity Preserved** | Chain of custody model where compliant material is physically separated throughout |
| **Segregated** | Chain of custody model where compliant material is kept separate from non-compliant but may be mixed with other compliant material |
| **Plot** | Specific parcel of land where a commodity is produced, with GPS/polygon geolocation |
| **Operator** | Any entity placing EUDR-regulated products on the EU market |

### Appendix B: EUDR Article 9 Geolocation Requirements

Per Article 9(1), the following geolocation information is required:
- (a) For plots of land: GPS coordinates of a single point (latitude/longitude)
- (b) For plots exceeding 4 hectares used for production of commodities other than cattle: a polygon with GPS points of all vertices
- (c) For plots exceeding 4 hectares used for cattle production: a polygon
- (d) For cattle: GPS coordinates of all establishments where cattle were kept

### Appendix C: Country Risk Classifications (Article 29 Reference)

The European Commission classifies countries as Low, Standard, or High risk based on:
- Rate of deforestation and forest degradation
- Rate of expansion of agriculture for relevant commodities
- Production trends of relevant commodities
- Information from indigenous peoples, local communities, and civil society

The agent implements a configurable country risk database that can be hot-reloaded when the EC publishes updates.

### Appendix D: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023
2. EU Deforestation Regulation Guidance Document (European Commission)
3. EUDR Technical Specifications for the Information System
4. GHG Protocol Corporate Value Chain (Scope 3) Accounting and Reporting Standard
5. ISO 22095:2020 -- Chain of Custody -- General Terminology and Models
6. FSC Chain of Custody Standard (FSC-STD-40-004)
7. RSPO Supply Chain Certification Standard
8. Global Forest Watch Technical Documentation

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-06 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________  |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-06 | GL-ProductManager | Initial draft created |
| 1.0.0 | 2026-03-06 | GL-ProductManager | Finalized: all 9 P0 features confirmed, regulatory coverage verified (Articles 2/3/4/9/10/11/12/29/31), module path aligned with GreenLang conventions, CBAM reference corrected, approval granted |
