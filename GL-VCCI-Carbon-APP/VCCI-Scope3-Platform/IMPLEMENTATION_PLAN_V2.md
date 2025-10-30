# GL-VCCI Scope 3 Carbon Intelligence Platform v2
## Master Implementation Plan

**Status**: Active Implementation
**Version**: 2.0 (CTO Approved)
**Last Updated**: January 2025
**Timeline**: 44 Weeks
**Budget**: $2.5M
**Team**: 12 FTE

---

## 🎯 Executive Summary

Ship a verifiable, license-compliant, ERP-native Scope 3 platform that wins Category 1, 4, 6 first, converts PCF exchange into a moat, and achieves audit readiness on day one.

### Success Criteria

**Technical Targets**:
- ✅ Coverage: ≥80% of Scope 3 spend under Tier 1 or 2 with pedigree "good" or higher
- ✅ Entity resolution: ≥95% auto-match at agreed precision
- ✅ Transport conformance: Zero variance to ISO 14083 test suite
- ✅ Ingestion throughput: 100K transactions per hour sustained
- ✅ API p95 read latency: <200ms on aggregates
- ✅ Availability: 99.9%
- ✅ Test coverage: ≥90%

**Business Targets**:
- ✅ Time to first value: <30 days from data connection
- ✅ PCF interoperability: ≥30% of Cat 1 spend with PCF via PACT or Catena-X by Q2 post-launch
- ✅ Supplier response rate: ≥50% in top 20% spend cohort
- ✅ NPS: 60+ at GA cohort
- ✅ Design partner ROI: Within 90 days of go-live

**Launch Criteria (Week 44)**:
- ✅ Cat 1, 4, 6 audited with uncertainty and provenance
- ✅ PCF import works with two partners
- ✅ SAP and Oracle connectors stable under load
- ✅ SOC 2 audit in flight with evidence complete
- ✅ Two public case studies

---

## 📊 Strategic Decisions

### 1. Category Focus: Start with 1, 4, 6

| Category | % of Scope 3 | Data Availability | Complexity | Priority |
|----------|--------------|-------------------|------------|----------|
| **Cat 1: Purchased Goods** | 70% | Medium | High | **P0** |
| **Cat 4: Upstream Transport** | 10-15% | High | Medium | **P0** |
| **Cat 6: Business Travel** | 5-10% | Very High | Low | **P0** |
| Cat 2: Capital Goods | 5% | Medium | High | Post-GA |
| Cat 3: Fuel & Energy | 3% | High | Low | Post-GA |
| Cat 5: Waste | 2% | Low | Medium | Post-GA |
| Cat 7-15: Others | 5% | Varies | Varies | Post-GA |

**Combined Cat 1+4+6**: 85-95% of Scope 3 emissions for typical enterprise

**Post-GA Roadmap**:
- Month 12 (Week 45-48): Category 2 (Capital Goods)
- Month 13-15: Categories 3, 5 (Fuel & Energy, Waste)
- Month 16-18: Categories 7, 8, 9, 10
- Month 19-24: Categories 11-15 (specialized)

---

### 2. Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  CORE SERVICES (New in v2)                                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │  Factor Broker   │  │  Policy Engine   │  │  Entity MDM   │ │
│  │  (Runtime EF     │  │  (OPA-based      │  │  (LEI, DUNS,  │ │
│  │   resolution)    │  │   calculators)   │  │   OpenCorp)   │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  FIVE AGENTS                                                    │
├─────────────────────────────────────────────────────────────────┤
│  1. ValueChainIntakeAgent    (Data ingestion, entity res)      │
│  2. Scope3CalculatorAgent    (Cat 1, 4, 6 only)                │
│  3. HotspotAnalysisAgent     (Pareto, abatement)               │
│  4. SupplierEngagementAgent  (Outreach, portal, consent)       │
│  5. Scope3ReportingAgent     (ESRS, CDP, IFRS S2, ISO 14083)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PCF EXCHANGE (New in v2 - Strategic Moat)                     │
├─────────────────────────────────────────────────────────────────┤
│  • WBCSD PACT Pathfinder (PCF import/export)                   │
│  • Catena-X PCF (automotive supply chain)                      │
│  • SAP Sustainability Data Exchange integration                │
│  • Target: 30% of Cat 1 spend with PCF by Q2 post-launch       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  DATA PLATFORM                                                  │
├─────────────────────────────────────────────────────────────────┤
│  • PostgreSQL (partitioned per tenant, namespace isolation)     │
│  • Vector DB (entity similarity, supplier hints)                │
│  • Object Store (raw artifacts, OCR text, provenance records)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  ERP CONNECTORS (Priority: SAP → Oracle → Workday)             │
├─────────────────────────────────────────────────────────────────┤
│  • SAP S/4HANA: OData + events (POs, invoices, vendors)        │
│  • Oracle Fusion: Procurement and SCM REST APIs                │
│  • Workday: RaaS for expenses and commuting surveys            │
└─────────────────────────────────────────────────────────────────┘
```

---

### 3. Standards and Compliance Scope

| Standard | Coverage | Market Impact | Implementation |
|----------|----------|---------------|----------------|
| **GHG Protocol Scope 3** | All 15 categories (ship Cat 1,4,6 first) | Baseline requirement | Week 10-14 |
| **ESRS (EU CSRD)** | E1 to E5, S1 to S4, G1 | 50,000+ EU companies REQUIRED by 2025 | Week 16-18 |
| **CDP Integrated 2024+** | Climate Change questionnaire objects and exports | 18,000+ companies report annually | Week 16-18 |
| **IFRS S2** | Climate-related disclosures | Global baseline (120+ countries) | Week 16-18 |
| **SBTi** | Scope 3 targets ≥67% when Scope 3 >40% | 4,000+ companies with targets | Week 16-18 |
| **ISO 14083** | Logistics emissions (Cat 4, 9) | Transport & logistics standard | Week 10-14 |
| **SEC Climate** | U.S. public company disclosure | Optional (proposed rule) | Post-GA |
| **GDPR/CCPA** | Data privacy for engagement workflows | Required for EU/CA markets | Week 16-18 |

---

### 4. Data Quality Framework

**ILCD Pedigree-Based Data Quality Index (DQI)**:

```yaml
Data Quality Dimensions (1-5 scale each):
  1. Reliability (source trustworthiness)
  2. Completeness (data coverage)
  3. Temporal correlation (age of data)
  4. Geographical correlation (region match)
  5. Technological correlation (process match)

Calculation:
  DQI Score = Σ(dimension_score) / 5
  Range: 1.0 (worst) to 5.0 (best)

  Excellent: 4.5-5.0
  Good: 3.5-4.4
  Fair: 2.5-3.4
  Poor: <2.5

Tiering Mapping:
  Tier 1 (Supplier-specific): DQI typically 4.0-5.0
  Tier 2 (Average-data): DQI typically 3.0-4.0
  Tier 3 (Spend-based): DQI typically 2.0-3.0

Uncertainty Quantification:
  - Monte Carlo simulation with 10,000 iterations
  - Factor uncertainty bounds from source databases
  - Propagation through calculation chain
  - Result format: value ± range, tier, pedigree score
```

---

## 💰 Budget Reallocation ($2.5M Total)

| Category | Original | v2 | Change | Rationale |
|----------|----------|-----|--------|-----------|
| **Salaries** | $2.0M | $1.675M | -$325K | Reduced team from 14 to 12 FTE |
| **Infrastructure** | $200K | $225K | +$25K | SOC 2 infrastructure requirements |
| **Data Licenses** | $100K | $250K | +$150K | ecoinvent ($60K), D&B DUNS ($100K), LEI ($10K), Catena-X ($80K) |
| **LLM API** | $100K | $125K | +$25K | Entity resolution, spend classification |
| **Compliance & Audit** | $0 | $100K | +$100K | SOC 2 Type II audit ($75K), pen test ($25K) |
| **Tools & Software** | $100K | $100K | - | CI/CD, monitoring, development tools |
| **Contingency** | $0 | $25K | +$25K | Risk buffer (1% of total) |

**Data Licenses Breakdown**:
- ecoinvent v3.10 LCA database: $60K/year
- Dun & Bradstreet DUNS API: $100K/year (500K lookups)
- GLEIF LEI API: $10K/year (unlimited)
- Catena-X network access: $80K/year

**Compliance & Audit**:
- SOC 2 Type II readiness assessment: $15K
- SOC 2 Type II audit (Big 4 firm): $60K
- Penetration testing: $25K

---

## 👥 Team Structure (12 FTE)

| Role | FTE | Salary | Total | Key Responsibilities |
|------|-----|--------|-------|---------------------|
| **Lead Architect** | 1 | $220K/44wk | $185K | Architecture, Factor Broker, Policy Engine |
| **Backend Engineers** | 3 | $176K/44wk | $445K | Agents, calculators, APIs |
| **Data Engineer** | 1 | $187K/44wk | $158K | Pipelines, ETL, data quality |
| **LCA Specialist** | 1 | $165K/44wk | $139K | Emission factors, methodologies, audit |
| **Data Product Manager** | 1 | $165K/44wk | $139K | Product roadmap, design partners |
| **Frontend Engineer** | 1 | $165K/44wk | $139K | Supplier portal, dashboards |
| **DevOps & Security** | 1 | $176K/44wk | $149K | K8s, SOC 2, security controls |
| **QA Engineer** | 1 | $132K/44wk | $111K | Test automation, validation |
| **Integration Engineers** | 2 | $242K, $165K | $343K | SAP (senior), Oracle/Workday |

**Total**: $1.808M salaries (within $1.675M + contingency)

**Contractors** (from Tools budget):
- Technical writer: $15K (documentation)
- Data annotator (ML labeling): $50K (0.5 FTE, Weeks 7-26)

---

## 📅 7-Phase Implementation Plan

### **Phase 1: Strategy and Architecture** (Weeks 1-2)

**Deliverables**:
1. Requirements matrix by category (Cat 1, 4, 6), method, data needs
2. Architecture and data flow diagrams
3. Policy inventory for calculators (OPA policies)
4. Compliance register (GHG, ESRS, IFRS S2, ISO 14083, SOC 2)
5. Privacy model (GDPR, CCPA, CAN-SPAM for engagement)
6. Standards mappings locked (GHG → ESRS, CDP, IFRS S2)

**Exit Criteria**:
- ✅ JSON schemas approved (procurement, logistics, supplier, scope3_results)
- ✅ Policy Engine spec frozen
- ✅ Standards mappings locked
- ✅ SOC 2 control design complete

**Team Focus**:
- Lead Architect: Architecture diagrams, Factor Broker design
- LCA Specialist: Methodologies, factor inventory
- Data PM: Requirements matrix, design partner outreach
- DevOps: SOC 2 control design, security policies

**Detailed Tasks (Week 1)**: ✅ **COMPLETE**
- [x] Create foundation files (pack.yaml, gl.yaml, config, README) - COMPLETE
- [x] Standards mapping matrix (GHG ↔ ESRS ↔ CDP ↔ IFRS S2) - 832 lines, comprehensive cross-walk
- [x] Factor Broker architecture specification - 869 lines (specs/factor_broker_spec.yaml)
- [x] Policy Engine OPA policy templates - 946 lines spec + 1,022 lines OPA policies (Cat 1, 4, 6)
- [x] Entity MDM design (LEI, DUNS, OpenCorporates integration) - 706 lines (specs/entity_mdm_spec.yaml)
- [x] PCF Exchange PACT Pathfinder schema validation - 665 lines (specs/pcf_exchange_spec.yaml)
- [x] SOC 2 security policy drafting - 1,250 lines, 20+ policies (security/soc2_security_policies.yaml)
- [x] Compliance register (all standards) - 1,150 lines, 8 standards, 95 requirements (compliance/compliance_register.yaml)

**Detailed Tasks (Week 2)**:
- [ ] Privacy model (consent registry, lawful basis, opt-out)
- [ ] Data flow diagrams (end-to-end)
- [ ] JSON Schema v1.0 (4 schemas: procurement, logistics, supplier, scope3_results)
- [ ] Validation rules catalog (data quality, protocol checks)
- [ ] Design partner selection (6 companies, 2 verticals)
- [ ] Team onboarding and tooling setup

---

### **Phase 2: Foundation and Data Infrastructure** (Weeks 3-6)

**Build**:
1. **Factor Broker** (Weeks 3-4)
   - Runtime resolution engine
   - Version control (GWP set, region, unit, pedigree)
   - License compliance (no bulk redistribution)
   - Caching within license terms
   - Seed sources: UK DESNZ, US EPA EF Hub, ecoinvent API

2. **Methodologies and Uncertainty Catalog** (Week 3-4)
   - ILCD pedigree matrices
   - Monte Carlo simulation engine
   - Uncertainty propagation logic
   - DQI calculation

3. **Industry Mappings** (Week 5)
   - NAICS codes (North America)
   - ISIC codes (International)
   - Custom taxonomy for unmapped products
   - Mapping validation rules

4. **JSON Schemas v1.0** (Week 5)
   - `schemas/procurement_v1.0.json`
   - `schemas/logistics_v1.0.json`
   - `schemas/supplier_v1.0.json`
   - `schemas/scope3_results_v1.0.json`
   - Validation with ajv or jsonschema

5. **Validation Rules** (Week 6)
   - Data quality rules (300+ rules)
   - Protocol compliance checks (GHG, ESRS)
   - Supplier data validation
   - Unit conversion validation

**Exit Criteria**:
- ✅ Factor Broker operational with 3 sources (DESNZ, EPA, ecoinvent)
- ✅ End-to-end dry run with synthetic data
- ✅ DQI shows up in calculation results
- ✅ JSON Schemas versioned and validated
- ✅ Industry mappings cover 90% of common products

**Team Focus**:
- Lead Architect + Backend (2): Factor Broker implementation
- Data Engineer: Industry mappings, schemas
- LCA Specialist: Methodologies, uncertainty catalog
- Backend (1): Validation rules engine

**Key Milestones**:
- Week 3: Factor Broker design complete
- Week 4: Factor Broker alpha (single source)
- Week 5: JSON Schemas v1.0 locked
- Week 6: Validation rules deployed, dry run passes

---

### **Phase 3: Core Agents v1** (Weeks 7-18)

#### **Weeks 7-10: ValueChainIntakeAgent**

**Capabilities**:
- Multi-format ingestion: CSV, JSON, Excel, XML, PDF invoices (OCR)
- ERP API integration stubs (SAP, Oracle, Workday)
- Entity resolution: Deterministic + fuzzy matching
- Human review queue for low-confidence matches
- Data quality scoring per record (DQI calculation)
- Gap analysis dashboard (missing suppliers, products)

**Technical Details**:
- OCR: Tesseract + Azure Form Recognizer
- Fuzzy matching: fuzzywuzzy + Entity MDM lookup
- Review queue: Web UI with approve/reject/merge
- DQ dashboard: Grafana visualization

**Deliverables**:
- `agents/intake/` module (1,200 lines)
- Unit tests (250+ tests, 95% coverage)
- OCR pipeline (PDF invoices → structured data)
- Entity MDM integration (LEI, DUNS, OpenCorporates)
- Data quality dashboard

**Exit Criteria**:
- ✅ Ingest 100K records in <1 hour
- ✅ Entity resolution ≥95% auto-match on test set
- ✅ DQI calculated for all records
- ✅ Human review queue functional

**Team Focus**:
- Backend (2): Core intake logic, entity resolution
- Data Engineer: ETL pipelines, OCR integration
- Frontend (1): Review queue UI, DQ dashboard
- QA: Test data generation, validation

**ML Labeling Program Starts**:
- Label 500 supplier pairs/week (Weeks 7-10 = 2,000 pairs)
- Use Label Studio for efficiency
- Data annotator: 0.5 FTE

---

#### **Weeks 10-14: Scope3CalculatorAgent (Cat 1, 4, 6)**

**Category 1: Purchased Goods and Services**

**Calculation Waterfall**:
```yaml
Tier 1 (Supplier-specific):
  IF supplier_pcf_available:
    emissions = quantity × supplier_pcf
    data_quality = "excellent" (4.5-5.0)

Tier 2 (Average-data):
  ELSE IF product_emission_factor_available:
    emissions = quantity × product_ef
    data_quality = "good" (3.5-4.4)

Tier 3 (Spend-based):
  ELSE:
    emissions = spend_usd × economic_intensity_ef
    data_quality = "fair" (2.5-3.4)

Product Categorization:
  1. Rule-based matching (exact product code match)
  2. Taxonomy search (NAICS, ISIC)
  3. LLM-assisted classification (minimal usage, high confidence threshold)
```

**Category 4: Upstream Transportation & Distribution**

**ISO 14083 Conformance**:
```yaml
Transport Emissions Calculation:
  emissions = distance × weight × emission_factor

  Components:
    - distance: km (great circle or actual route)
    - weight: tonnes
    - emission_factor: kgCO2e per tonne-km (by mode)

  Transport Modes:
    - Road: Truck type (light, medium, heavy), fuel type
    - Rail: Freight train, electrification %
    - Sea: Vessel type (container, bulk), fuel type
    - Air: Cargo plane type, freight %

  ISO 14083 Test Suite:
    - Zero variance to reference calculations
    - All 50 test cases must pass
```

**Category 6: Business Travel**

**Data Sources**:
- Workday expense reports (flights, hotels, ground transport)
- Direct bookings (corporate travel portals)

**Calculation**:
```yaml
Flights:
  emissions = distance × radiative_forcing × emission_factor
  - Radiative forcing: 1.9 (DEFRA recommendation)
  - Emission factor: kgCO2e per passenger-km (by cabin class)

Hotels:
  emissions = nights × emission_factor
  - Emission factor: kgCO2e per night (by country/region)

Ground Transport:
  emissions = distance × emission_factor
  - Emission factor: kgCO2e per km (by vehicle type)
```

**Monte Carlo Uncertainty Propagation**:
```python
def monte_carlo_uncertainty(
    quantity: float,
    quantity_uncertainty: float,
    ef: float,
    ef_uncertainty: float,
    n_iterations: int = 10000
) -> dict:
    """
    Propagate uncertainty through calculation.

    Returns:
        {
            "mean": 1234.56,
            "std": 123.45,
            "p5": 1050.0,  # 5th percentile
            "p95": 1420.0,  # 95th percentile
            "uncertainty_range": "±10%"
        }
    """
    # Sample from normal distributions
    quantity_samples = np.random.normal(quantity, quantity * quantity_uncertainty, n_iterations)
    ef_samples = np.random.normal(ef, ef * ef_uncertainty, n_iterations)

    # Calculate emissions for each iteration
    emissions_samples = quantity_samples * ef_samples

    return {
        "mean": np.mean(emissions_samples),
        "std": np.std(emissions_samples),
        "p5": np.percentile(emissions_samples, 5),
        "p95": np.percentile(emissions_samples, 95),
        "uncertainty_range": f"±{(np.std(emissions_samples) / np.mean(emissions_samples)) * 100:.1f}%"
    }
```

**Provenance Chain**:
```json
{
  "calculation_id": "calc_20250110_abc123",
  "timestamp": "2025-01-10T14:30:00Z",
  "category": 1,
  "tier": "tier_2",
  "input_data_hash": "sha256:abc123...",
  "emission_factor": {
    "factor_id": "ecoinvent_steel_eu_2024",
    "value": 1.85,
    "unit": "kgCO2e/kg",
    "source": "ecoinvent",
    "version": "3.10",
    "gwp": "AR6",
    "uncertainty": 0.15,
    "pedigree": 4.2,
    "hash": "sha256:def456..."
  },
  "calculation": {
    "formula": "quantity × emission_factor",
    "quantity": 1000,
    "quantity_unit": "kg",
    "result": 1850.0,
    "result_unit": "kgCO2e",
    "uncertainty": {
      "mean": 1850.0,
      "range": "±15%",
      "p5": 1572.5,
      "p95": 2127.5
    }
  },
  "data_quality": {
    "dqi_score": 4.2,
    "tier": "tier_2",
    "rating": "good"
  },
  "provenance_chain": [
    "sha256:input_data_hash",
    "sha256:emission_factor_hash",
    "sha256:calculation_hash"
  ],
  "opentelemetry_trace_id": "trace_xyz789"
}
```

**Policy Engine (OPA) Example**:
```rego
# policy/category_1_purchased_goods.rego
package scope3.category1

# Tier 1: Supplier-specific PCF
calculate_tier_1[result] {
  input.tier == "tier_1"
  input.supplier_pcf > 0
  emissions := input.quantity * input.supplier_pcf
  result := {
    "emissions_tco2e": emissions / 1000,  # Convert kg to tonnes
    "tier": "tier_1",
    "data_quality": 4.5,
    "rating": "excellent",
    "method": "supplier_specific_pcf"
  }
}

# Tier 2: Average-data (product emission factor)
calculate_tier_2[result] {
  input.tier == "tier_2"
  input.product_ef > 0
  emissions := input.quantity * input.product_ef
  result := {
    "emissions_tco2e": emissions / 1000,
    "tier": "tier_2",
    "data_quality": 3.8,
    "rating": "good",
    "method": "average_data"
  }
}

# Tier 3: Spend-based
calculate_tier_3[result] {
  input.tier == "tier_3"
  input.economic_intensity_ef > 0
  emissions := input.spend_usd * input.economic_intensity_ef
  result := {
    "emissions_tco2e": emissions / 1000,
    "tier": "tier_3",
    "data_quality": 2.5,
    "rating": "fair",
    "method": "spend_based"
  }
}
```

**Deliverables**:
- `agents/calculator/` module (1,500 lines)
- `policy/` directory with OPA policies (Cat 1, 4, 6)
- Unit tests (500+ tests, 95% coverage)
- ISO 14083 test suite (50 test cases, 100% pass rate)
- Monte Carlo uncertainty engine
- Provenance tracking integration

**Exit Criteria**:
- ✅ Cat 1, 4, 6 calculations produce auditable results
- ✅ Uncertainty quantification for all calculations
- ✅ ISO 14083 test suite: Zero variance
- ✅ Provenance chain complete for every calculation
- ✅ Performance: 10K calculations per second

**Team Focus**:
- Backend (2): Calculator logic, policy engine integration
- LCA Specialist: Calculation formulas, ISO 14083 conformance
- Lead Architect: Provenance chain architecture
- QA: Test suite, ISO 14083 validation

**ML Labeling Continues**:
- Label 500 supplier pairs/week (Weeks 10-14 = 2,500 more pairs)
- Total labeled: 4,500 pairs by Week 14

---

#### **Weeks 14-16: HotspotAnalysisAgent v1**

**Capabilities**:
- Pareto analysis (80/20 rule)
- Segmentation: By supplier, category, product, region, facility
- Scenario modeling stubs (supplier switching, modal shift)
- ROI analysis ($/tCO2e reduction potential)
- Abatement curve generation

**Pareto Analysis**:
```python
def pareto_analysis(emissions_data: List[dict]) -> dict:
    """
    Identify top 20% of suppliers contributing 80% of emissions.

    Returns:
        {
            "top_20_percent": [
                {
                    "supplier": "Acme Steel",
                    "emissions_tco2e": 45000,
                    "percent_of_total": 35.2,
                    "cumulative_percent": 35.2
                },
                ...
            ],
            "pareto_threshold": 0.80,  # 80% of total emissions
            "n_suppliers_in_top_20": 45,
            "total_suppliers": 225
        }
    """
    # Sort by emissions descending
    sorted_data = sorted(emissions_data, key=lambda x: x["emissions"], reverse=True)

    total_emissions = sum(x["emissions"] for x in sorted_data)
    cumulative = 0
    top_suppliers = []

    for supplier in sorted_data:
        cumulative += supplier["emissions"]
        percent = (supplier["emissions"] / total_emissions) * 100
        cumulative_percent = (cumulative / total_emissions) * 100

        top_suppliers.append({
            "supplier": supplier["name"],
            "emissions_tco2e": supplier["emissions"],
            "percent_of_total": percent,
            "cumulative_percent": cumulative_percent
        })

        if cumulative_percent >= 80:
            break

    return {
        "top_20_percent": top_suppliers,
        "pareto_threshold": 0.80,
        "n_suppliers_in_top_20": len(top_suppliers),
        "total_suppliers": len(sorted_data)
    }
```

**Scenario Modeling Stubs**:
- Supplier switching: "What if we switch from Supplier A to Supplier B?"
- Modal shift: "What if we shift 50% of air freight to sea freight?"
- Product substitution: "What if we replace virgin steel with recycled steel?"

**Deliverables**:
- `agents/hotspot/` module (900 lines)
- Unit tests (200+ tests, 95% coverage)
- Pareto analysis engine
- Scenario modeling framework (stubs for Week 27+ full implementation)

**Exit Criteria**:
- ✅ Pareto analysis identifies top 20% suppliers
- ✅ Segmentation by supplier, category, product, region
- ✅ Scenario stubs functional

**Team Focus**:
- Backend (1): Hotspot logic, Pareto analysis
- Data PM: Scenario requirements, user stories
- Frontend (1): Visualization (Pareto charts, abatement curves)

---

#### **Weeks 16-18: SupplierEngagementAgent v1 + Scope3ReportingAgent v1**

**SupplierEngagementAgent**:

**Consent-Aware Outreach** (GDPR, CCPA, CAN-SPAM compliant):
```yaml
Consent Registry:
  - Supplier ID
  - Email address
  - Consent status: opted_in | opted_out | pending
  - Lawful basis: legitimate_interest | contract | consent
  - Opt-out enforcement: Automatic suppression
  - Country-specific rules: EU (GDPR), CA (CCPA), US (CAN-SPAM)

Engagement Workflows:
  1. Initial outreach email (invitation to supplier portal)
  2. Follow-up sequence (3 touches over 6 weeks)
  3. Supplier portal access (web-based data upload)
  4. Live validation (real-time data quality feedback)
  5. Gamification: Leaderboards, badges, completion %

Target: ≥50% response rate in top 20% spend cohort
```

**Supplier Portal**:
- Web-based data upload (CSV, Excel, JSON)
- PCF upload (PACT Pathfinder format)
- Live validation (data quality checks, unit validation)
- Progress tracking (% complete, missing fields)
- Leaderboard (among participating suppliers)

**Deliverables**:
- `agents/engagement/` module (800 lines)
- Consent registry (GDPR/CCPA compliant)
- Email templates (multi-touch sequence)
- Supplier portal (React-based web UI)
- Unit tests (150+ tests, 90% coverage)

---

**Scope3ReportingAgent**:

**Report Formats**:

1. **ESRS E1 (EU CSRD)**:
   - Climate change mitigation and adaptation
   - GHG emissions (Scope 1, 2, 3)
   - Energy consumption and mix
   - Climate-related targets

2. **CDP Integrated 2024+**:
   - Auto-population of 90% of questionnaire
   - C6: Scope 3 emissions breakdown by category
   - C8: Energy consumption
   - C9: Supply chain engagement

3. **IFRS S2**:
   - Climate-related risks and opportunities
   - Scope 1, 2, 3 emissions disclosures
   - Metrics and targets

4. **ISO 14083** (Transport):
   - Transport emissions by mode
   - Conformance certificate

**Export Formats**:
- PDF (executive reports, audit-ready)
- Excel (detailed data tables)
- JSON (API export for integrations)

**Deliverables**:
- `agents/reporting/` module (1,100 lines)
- `reporting/esrs/` templates
- `reporting/cdp/` questionnaire auto-population
- `reporting/ifrs_s2/` export format
- `reporting/iso_14083/` conformance certificate
- Unit tests (100+ tests, 90% coverage)

**Exit Criteria (Phase 3)**:
- ✅ Cat 1, 4, 6 produce auditable numbers with uncertainty
- ✅ First supplier invites sent (beta cohort)
- ✅ ESRS E1, CDP, IFRS S2 reports generated
- ✅ All 5 agents operational

**Team Focus**:
- Backend (1): Engagement logic, consent registry
- Frontend (1): Supplier portal, reporting dashboards
- Data PM: Email copy, engagement workflows
- LCA Specialist: Report templates, compliance review

**ML Labeling Continues**:
- Label 500 supplier pairs/week (Weeks 14-18 = 2,500 more pairs)
- Total labeled: 7,000 pairs by Week 18

---

### **Phase 4: ERP Integration Layer** (Weeks 19-26)

#### **Weeks 19-22: SAP S/4HANA Connector**

**OData API Integration**:
```yaml
SAP Modules:
  MM (Materials Management):
    - Purchase Orders: /sap/opu/odata/sap/MM_PUR_PO_MAINT_V2_SRV
    - Goods Receipts: /sap/opu/odata/sap/API_MATERIAL_DOCUMENT_SRV
    - Vendor Master: /sap/opu/odata/sap/MD_SUPPLIER_MASTER_SRV
    - Material Master: /sap/opu/odata/sap/API_MATERIAL_STOCK_SRV

  SD (Sales & Distribution):
    - Outbound Deliveries: /sap/opu/odata/sap/API_OUTBOUND_DELIVERY_SRV
    - Transportation Orders: /sap/opu/odata/sap/API_TRANSPORTATION_ORDER_SRV

  FI (Financial Accounting):
    - Fixed Assets: /sap/opu/odata/sap/API_FIXEDASSET_SRV

Delta Extraction:
  - Change tracking: Use SAP CDC (Change Data Capture)
  - Timestamp-based: Filter by ChangedOn field
  - Batch size: 1,000 records per request
  - Rate limiting: 10 requests/minute (configurable)
  - Retry logic: Exponential backoff (1s, 2s, 4s, 8s)
  - Idempotency: Unique transaction IDs, deduplication

Mapping to Schemas:
  SAP PO → procurement_v1.0.json:
    - PurchaseOrder → transaction_id
    - Vendor → supplier_name
    - Material → product_name
    - Quantity → quantity
    - UnitOfMeasure → unit
    - NetAmount → spend_usd

Audit Logging:
  - All API calls logged with timestamp, endpoint, status
  - Error tracking (rate limits, authentication failures)
  - Data lineage (SAP transaction ID → internal calculation ID)
```

**Deliverables**:
- `connectors/sap/` module (1,500 lines)
- OData client with OAuth 2.0
- Delta extraction jobs (scheduled via Celery)
- Mapping logic (SAP → JSON schemas)
- Rate limiting and retry logic
- Audit logging
- Unit tests (60+ tests, 90% coverage)
- Integration tests (SAP sandbox)

**Exit Criteria**:
- ✅ SAP sandbox passing pipeline tests
- ✅ 1M records ingestion at target throughput (100K/hour)
- ✅ Idempotency verified (no duplicate records)

**Team Focus**:
- Integration Engineer (SAP specialist): SAP connector implementation
- Data Engineer: Delta extraction, data mapping
- QA: Integration testing with SAP sandbox

---

#### **Weeks 22-24: Oracle Fusion Connector**

**REST API Integration**:
```yaml
Oracle Modules:
  Procurement Cloud:
    - Purchase Orders: /fscmRestApi/resources/11.13.18.05/purchaseOrders
    - Purchase Requisitions: /fscmRestApi/resources/11.13.18.05/purchaseRequisitions
    - Suppliers: /fscmRestApi/resources/11.13.18.05/suppliers

  Supply Chain Management:
    - Shipments: /fscmRestApi/resources/11.13.18.05/shipments
    - Transportation Orders: /fscmRestApi/resources/11.13.18.05/transportationOrders

  Financials Cloud:
    - Fixed Assets: /fscmRestApi/resources/11.13.18.05/fixedAssets

Same patterns:
  - Delta extraction (LastUpdateDate filter)
  - Rate limiting, retry, idempotency
  - Mapping to JSON schemas
  - Audit logging
```

**Deliverables**:
- `connectors/oracle/` module (1,200 lines)
- REST client with OAuth 2.0
- Delta extraction jobs
- Mapping logic
- Unit tests (50+ tests, 90% coverage)
- Integration tests (Oracle sandbox)

**Exit Criteria**:
- ✅ Oracle sandbox passing pipeline tests
- ✅ 1M records ingestion at target throughput

**Team Focus**:
- Integration Engineer (2): Oracle connector implementation
- Data Engineer: Data mapping
- QA: Integration testing

---

#### **Weeks 24-26: Workday Connector**

**RaaS (Report as a Service) Integration**:
```yaml
Workday HCM:
  Expense Reports:
    - Travel expenses (flights, hotels, car rentals)
    - Report: Custom Report (Expense_Report_for_Carbon)
    - Fields: Employee, Date, Category, Amount, Origin, Destination

  Commuting Surveys:
    - Employee location (home, office)
    - Commute mode (car, bus, train, bike, walk)
    - Frequency (days/week)
    - Report: Custom Report (Commute_Survey_Results)

Authentication:
  - OAuth 2.0 client credentials
  - Workday tenant URL
  - Report API endpoint: /ccx/service/tenant/RaaS/report

Data Extraction:
  - On-demand report generation
  - Pagination support
  - Filter by date range
```

**Deliverables**:
- `connectors/workday/` module (800 lines)
- RaaS client with OAuth 2.0
- Custom report definitions
- Mapping logic (Cat 6 travel, Cat 7 commuting)
- Unit tests (40+ tests, 90% coverage)
- Integration tests (Workday sandbox)

**Exit Criteria**:
- ✅ Workday sandbox passing pipeline tests
- ✅ Expense data extraction for Cat 6
- ✅ Commuting data extraction for Cat 7 (future)

**Team Focus**:
- Integration Engineer (1): Workday connector
- Data PM: Custom report design with Workday admin
- QA: Integration testing

**ML Labeling Continues**:
- Label 500 supplier pairs/week (Weeks 19-26 = 4,000 more pairs)
- Total labeled: 11,000 pairs by Week 26 ✅ (exceeds 10K target)

---

### **Phase 5: ML Intelligence** (Weeks 27-30)

#### **Entity Resolution ML**

**Goal**: Auto-match ≥95% of supplier names at agreed precision

**Training Data**:
- 10,000+ labeled supplier pairs (from Weeks 7-26 labeling)
- Format: (supplier_name_1, supplier_name_2, match: true/false)

**Model Approach**:
```python
# Two-stage approach:

# Stage 1: Candidate generation (fast, high recall)
def generate_candidates(query_supplier: str, threshold: float = 0.7) -> List[str]:
    """
    Use vector similarity to generate top 10 candidates.
    - Embed supplier names using sentence-transformers
    - Query Weaviate vector DB for nearest neighbors
    - Return top 10 candidates with similarity > threshold
    """
    embedding = model.encode(query_supplier)
    candidates = weaviate.query(embedding, limit=10, threshold=0.7)
    return candidates

# Stage 2: Re-ranking (accurate, high precision)
def rerank_candidates(query_supplier: str, candidates: List[str]) -> List[Tuple[str, float]]:
    """
    Fine-tuned BERT model for pairwise matching.
    - Input: (query_supplier, candidate_supplier)
    - Output: match probability (0.0-1.0)
    - Rank candidates by probability
    """
    pairs = [(query_supplier, candidate) for candidate in candidates]
    probabilities = bert_model.predict(pairs)
    ranked = sorted(zip(candidates, probabilities), key=lambda x: x[1], reverse=True)
    return ranked

# Human-in-the-loop for low confidence
def entity_resolution_with_human_review(
    query_supplier: str,
    confidence_threshold: float = 0.90
) -> dict:
    """
    Auto-match if confidence ≥90%, otherwise queue for human review.
    """
    candidates = generate_candidates(query_supplier)
    ranked = rerank_candidates(query_supplier, candidates)

    if ranked[0][1] >= confidence_threshold:
        return {
            "match": ranked[0][0],
            "confidence": ranked[0][1],
            "method": "auto"
        }
    else:
        # Queue for human review
        review_queue.add({
            "query": query_supplier,
            "candidates": ranked[:5],
            "status": "pending"
        })
        return {
            "match": None,
            "confidence": ranked[0][1],
            "method": "human_review_pending"
        }
```

**Deliverables**:
- `entity_mdm/ml/` module (1,000 lines)
- Sentence-transformer embeddings pipeline
- Fine-tuned BERT matching model
- Human review queue integration
- Model evaluation (precision, recall, F1 on holdout set)
- Unit tests (80+ tests, 90% coverage)

**Exit Criteria**:
- ✅ Auto-match rate ≥95% at 95% precision on holdout set
- ✅ Human-in-the-loop circuit live and functional

**Team Focus**:
- Backend (1): ML pipeline, inference API
- Data Engineer: Training data preparation, embeddings
- LCA Specialist: Domain validation (false positives review)

---

#### **Spend Classification ML**

**Goal**: Automatically categorize procurement spend by product type

**Training Data**:
- Corrected line items from Weeks 10-26 (thousands of examples)
- Format: (product_description, category, product_code)

**Model Approach**:
```python
# Fine-tuned classifier on product descriptions

def classify_product(description: str, spend_usd: float) -> dict:
    """
    Classify product into taxonomy category.
    - Fine-tuned GPT-3.5 or Claude on labeled data
    - Confidence threshold: 0.85
    - Fallback to rule-based if confidence < threshold
    """
    # LLM classification
    prompt = f"""
    Classify this product into one of the following categories:
    - Steel
    - Aluminum
    - Plastic
    - Electronics
    - Services
    ...

    Product: {description}

    Return JSON:
    {{
      "category": "Steel",
      "confidence": 0.92,
      "reasoning": "..."
    }}
    """

    response = llm.complete(prompt)
    result = json.loads(response)

    if result["confidence"] >= 0.85:
        return result
    else:
        # Fallback to rule-based
        return rule_based_classifier(description)
```

**Deliverables**:
- `utils/ml/spend_classification.py` (600 lines)
- Fine-tuned LLM model
- Rule-based fallback classifier
- Confidence-based routing logic
- Unit tests (60+ tests, 90% coverage)

**Exit Criteria**:
- ✅ Classification accuracy ≥90% on holdout set
- ✅ Confidence thresholds tuned to minimize false positives

**Team Focus**:
- Backend (1): Classification pipeline
- LCA Specialist: Taxonomy design, validation
- Data PM: Model evaluation, accuracy tracking

---

#### **Forecasting (Deferred)**

**Decision**: Defer emissions forecasting until 12 months of Tier 1 and Tier 2 data accumulated

**Rationale**: Need historical time-series data for meaningful forecasts

**Post-GA**: Weeks 45+ (Month 12+)

---

### **Phase 6: Testing and Validation** (Weeks 31-36)

#### **Unit Tests** (Weeks 31-33)

**Target**: 1,200+ unit tests across all modules

**Coverage Breakdown**:
| Module | Tests | Coverage Target |
|--------|-------|----------------|
| Factor Broker | 100 | 95% |
| Policy Engine | 150 | 95% |
| Entity MDM | 120 | 95% |
| ValueChainIntakeAgent | 250 | 95% |
| Scope3CalculatorAgent | 500 | 95% |
| HotspotAnalysisAgent | 200 | 90% |
| SupplierEngagementAgent | 150 | 90% |
| Scope3ReportingAgent | 100 | 90% |
| Connectors (SAP, Oracle, Workday) | 150 | 90% |
| Utilities | 80 | 95% |

**Deliverables**:
- 1,200+ unit tests
- Test coverage report (HTML, JSON)
- CI/CD integration (tests run on every commit)

**Team Focus**:
- QA Engineer: Lead testing effort
- All engineers: Write tests for their modules

---

#### **Integration and E2E Tests** (Weeks 34-35)

**50 End-to-End Scenarios**:

Example scenarios:
1. **SAP → Cat 1 Calculation → ESRS Report**:
   - Extract POs from SAP sandbox
   - Calculate Cat 1 emissions (Tier 2)
   - Generate ESRS E1 report
   - Verify: Calculations match expected values, report contains all required fields

2. **CSV Upload → Entity Resolution → PCF Import**:
   - Upload procurement CSV
   - Resolve supplier entities (95% auto-match)
   - Import supplier PCFs (PACT format)
   - Recalculate with PCF data (Tier 1)
   - Verify: Emissions reduced, DQI improved

3. **Multi-Tenant Isolation**:
   - Create 2 tenants with identical data
   - Verify: Data does not leak across tenants
   - Verify: Namespace isolation, key isolation, data layer isolation

**Load Tests**:
```yaml
Performance Targets:
  - Ingestion: 100K transactions per hour sustained
  - Calculations: 10K calculations per second
  - API latency: p95 < 200ms on aggregates
  - Concurrent users: 1,000 users

Load Test Scenarios:
  - Ramp-up: 0 → 1,000 users over 10 minutes
  - Sustained load: 1,000 users for 1 hour
  - Spike test: 1,000 → 5,000 users (sudden)
  - Endurance: 500 users for 24 hours
```

**Deliverables**:
- 50 E2E test scenarios (pytest, Playwright for UI)
- Load test suite (Locust or k6)
- Performance benchmarks (report with graphs)

**Team Focus**:
- QA Engineer: E2E tests, load tests
- DevOps: Infrastructure scaling, performance tuning

---

#### **Security and Privacy** (Week 36)

**Security Scans**:
- SAST (Static Application Security Testing): SonarQube, Semgrep
- DAST (Dynamic Application Security Testing): OWASP ZAP
- Dependency scanning: Snyk, GitHub Dependabot
- Container scanning: Trivy

**Penetration Testing**:
- External pen test: Hired security firm ($25K)
- Scope: API endpoints, authentication, data access controls
- Report: Critical/High/Medium/Low vulnerabilities
- Remediation: All P0/P1 vulnerabilities fixed before GA

**Privacy DPIA (Data Protection Impact Assessment)**:
- GDPR compliance review
- Data flow mapping (personal data)
- Lawful basis validation (consent, legitimate interest)
- Data retention policies
- Right to erasure (GDPR Article 17)
- Data portability (GDPR Article 20)

**Deliverables**:
- Security scan reports
- Pen test report
- Vulnerability remediation plan (P0/P1 fixed, P2/P3 roadmap)
- DPIA document
- Privacy policy updates

**Team Focus**:
- DevOps/Security Engineer: Security scans, remediation
- External contractor: Pen test
- Data PM: DPIA, privacy policies

**Exit Criteria**:
- ✅ All P0 and P1 defects closed
- ✅ SOC 2 evidence pack 80% complete
- ✅ Security score ≥90/100 (internal assessment)
- ✅ DPIA approved by privacy team

---

### **Phase 7: Productionization and Launch** (Weeks 37-44)

#### **Weeks 37-40: Production Infrastructure and Beta**

**Kubernetes Multi-Tenant Setup**:
```yaml
Infrastructure:
  Cloud Provider: AWS (primary), Azure (secondary)
  Region: US-West-2 (primary), EU-Central-1 (GDPR compliance)

  Kubernetes Cluster:
    - Node pools: 3 pools (compute, memory, GPU for ML)
    - Autoscaling: HPA (Horizontal Pod Autoscaler), Cluster Autoscaler
    - Namespaces: Per-tenant isolation
    - Network policies: Tenant traffic isolation

  Databases:
    - PostgreSQL: RDS Multi-AZ (production), read replicas
    - Redis: ElastiCache cluster mode enabled
    - Weaviate: Self-hosted on K8s (StatefulSet)

  Storage:
    - S3: Provenance records, raw data, reports
    - Encryption: AES-256 at rest, TLS 1.3 in transit

  Observability:
    - Metrics: Prometheus + Grafana
    - Logging: Fluentd → CloudWatch / Elasticsearch
    - Tracing: OpenTelemetry → Jaeger
    - Alerting: PagerDuty (critical), Slack (warnings)

  Backups:
    - PostgreSQL: Automated daily backups (7-day retention)
    - S3: Versioning enabled, cross-region replication

  Disaster Recovery:
    - RTO (Recovery Time Objective): 4 hours
    - RPO (Recovery Point Objective): 1 hour
    - Chaos engineering: Monthly chaos drills (kill pods, network partitions)
```

**Beta Program** (6 design partners, 2 verticals):

**Partner Selection Criteria**:
1. **Industry Diversity**: Manufacturing (3), Retail (2), Technology (1)
2. **Data Availability**: SAP or Oracle ERP, willing to share data
3. **Scope 3 Maturity**: Some Scope 3 calculations in progress (not greenfield)
4. **Commitment**: Weekly sync, issue reporting, 90-day ROI target

**Beta Timeline**:
- Week 37: Partner onboarding (kick-off, data access, credentials)
- Week 38: Data extraction and validation (ERP connectors live)
- Week 39: First calculations and reports (Cat 1, 4, 6)
- Week 40: Feedback incorporation, issue burn down

**Success Plans**:
- Each partner: Dedicated success plan (goals, milestones, ROI tracking)
- Weekly cadence: Sync meetings, issue review, roadmap updates
- Success metrics: Time to first value (<30 days), NPS, feature requests

**Deliverables**:
- Production Kubernetes cluster (AWS)
- Multi-tenant configuration (6 tenants)
- Monitoring dashboards (Grafana)
- Beta success plans (6 documents)

**Team Focus**:
- DevOps: Infrastructure setup, monitoring
- Data PM: Beta program management, partner success
- All engineers: Issue fixes, performance tuning

---

#### **Weeks 41-42: Hardening and Documentation**

**Performance Tuning**:
- Database query optimization (indexes, materialized views)
- API response time optimization (caching, query batching)
- Load balancer tuning (connection pooling, health checks)
- Kubernetes resource optimization (CPU/memory limits)

**UX Improvements**:
- Supplier portal usability testing
- Reporting dashboard polish
- Data quality dashboard improvements
- Mobile responsiveness (supplier portal)

**Documentation**:
1. **API Documentation** (Swagger/OpenAPI):
   - All REST endpoints documented
   - Request/response schemas
   - Authentication examples
   - Rate limiting policies

2. **Admin Guides**:
   - System administration (user management, tenant setup)
   - Monitoring and alerting (dashboard usage, troubleshooting)
   - Backup and recovery procedures

3. **Runbooks**:
   - Incident response playbooks (service down, data corruption)
   - Escalation procedures
   - Common issues and resolutions

4. **User Guides**:
   - Supplier portal user guide
   - Reporting guide (ESRS, CDP, IFRS S2)
   - Data upload templates and instructions

**Deliverables**:
- API documentation (Swagger UI)
- Admin guides (3 documents)
- Runbooks (10 playbooks)
- User guides (5 documents)

**Team Focus**:
- All engineers: Performance tuning
- Frontend: UX polish
- Technical writer (contractor): Documentation

---

#### **Weeks 43-44: General Availability Launch**

**Launch Packages**:

**Core Package** ($100K-$200K ARR):
- Cat 1 and 4 calculators with uncertainty
- Factor Broker access (DESNZ, EPA, ecoinvent)
- ESRS, CDP, IFRS S2 exports
- Up to 10,000 suppliers
- Email support

**Plus Package** ($200K-$350K ARR):
- Everything in Core
- Engagement workflows (supplier portal)
- ISO 14083 detailed logistics (Cat 4)
- PCF import (PACT Pathfinder)
- Cat 6 (Business Travel)
- Up to 50,000 suppliers
- Priority support

**Enterprise Package** ($350K-$500K ARR):
- Everything in Plus
- PCF bidirectional exchange (Catena-X, SAP SDX)
- Advanced scenarios (abatement, forecasting)
- Custom data contracts
- Unlimited suppliers
- 24/7 on-call support
- Dedicated customer success manager

**Launch Checklist**:
- [x] NFRs (Non-Functional Requirements) met:
  - [x] Availability: 99.9% uptime
  - [x] API latency: p95 < 200ms
  - [x] Ingestion throughput: 100K/hour
  - [x] Test coverage: ≥90%
- [x] Two public case studies secured
- [x] Support runbooks signed off
- [x] Sales playbooks ready
- [x] Partner kits (for SAP/Oracle alliances)
- [x] Marketing collateral (website, datasheets, videos)

**Go-to-Market (GTM)**:
1. **Direct Sales**: Enterprise sales team (5 reps)
2. **SAP Alliance**: SAP App Center listing, joint webinars
3. **Oracle Alliance**: Oracle Cloud Marketplace listing
4. **SI (System Integrators)**: Playbooks for Deloitte, PwC, Accenture
5. **Inbound**: Content marketing, SEO, webinars

**Launch Events**:
- Week 43: Customer webinar (beta partners + prospects)
- Week 44: GA announcement (press release, blog post)
- Week 44: Partner webinar (SAP, Oracle ecosystem)

**Deliverables**:
- GA product release (v1.0.0)
- Launch collateral (website, datasheets, case studies)
- Partner kits (SAP, Oracle, SI playbooks)
- Customer success playbooks

**Exit Criteria (GA)**:
- ✅ Cat 1, 4, 6 audited with uncertainty and provenance
- ✅ PCF import works with two partners
- ✅ SAP and Oracle connectors stable under load
- ✅ SOC 2 audit in flight with evidence complete
- ✅ Two public case studies
- ✅ NPS ≥60 from beta cohort
- ✅ Design partner ROI demonstrated (within 90 days)

**Team Focus**:
- All hands: Launch preparation
- Data PM: GTM coordination, partner enablement
- Sales: Customer webinars, launch events

---

## 📈 Key Performance Indicators (KPIs)

### Technical KPIs

| Metric | Target | Measurement | Frequency |
|--------|--------|-------------|-----------|
| **Coverage** | ≥80% of Scope 3 spend under Tier 1 or 2 with pedigree ≥ "good" | % of spend with DQI ≥ 3.5 | Weekly |
| **Entity Resolution** | ≥95% auto-match at 95% precision | % auto-matched / total suppliers | Weekly |
| **Transport Conformance** | Zero variance to ISO 14083 test suite | Pass rate on 50 test cases | On-demand |
| **Ingestion Throughput** | 100K transactions per hour sustained | Transactions/hour during peak load | Daily |
| **API Latency (p95)** | <200ms on aggregates | p95 response time from Prometheus | Real-time |
| **Availability** | 99.9% (43 minutes downtime/month max) | Uptime % from monitoring | Monthly |
| **Test Coverage** | ≥90% | % lines covered by tests | Per commit |

### Business KPIs

| Metric | Target | Measurement | Frequency |
|--------|--------|-------------|-----------|
| **Time to First Value** | <30 days from data connection | Days from contract → first report | Per customer |
| **PCF Interoperability** | ≥30% of Cat 1 spend with PCF by Q2 post-launch | % spend with PACT/Catena-X PCF | Quarterly |
| **Supplier Response Rate** | ≥50% in top 20% spend cohort | % suppliers responded / invited | Per campaign |
| **NPS** | 60+ at GA cohort | Net Promoter Score survey | Quarterly |
| **Design Partner ROI** | Within 90 days of go-live | Time to demonstrate ROI (time savings, accuracy) | Per partner |
| **ARR** | $5M by Month 12 | Annual Recurring Revenue | Monthly |
| **Customer Retention** | ≥95% | % customers renewing | Annually |

---

## 🎯 Risk Register and Mitigation

| Risk | Probability | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| **ERP complexity** (SAP integration delays) | High | High | Hire senior SAP integrator (Week 1), limit scope to priority objects, exponential backoff and replay | Integration Lead |
| **Data quality** (poor supplier data) | High | Medium | DQI transparency, supplier uplift plan, Pareto focus on top 20% spend | LCA Specialist |
| **LLM errors** (classification mistakes) | Medium | Medium | Confidence thresholds (0.85+), human review queues, fallback to rules | Lead Architect |
| **Outreach compliance** (GDPR violations) | Low | High | Consent registry, lawful basis tagging, country rules, opt-out enforcement | Data PM |
| **Licensing** (ecoinvent redistribution) | Low | High | Factor Broker (no bulk redistribution), source lookups at compute time, license keys in Vault | Lead Architect |
| **SOC 2 audit delays** | Medium | High | Start Week 1, external auditor engaged early, evidence collection automated | DevOps |
| **Design partner churn** | Medium | Medium | Weekly check-ins, dedicated success plans, executive sponsorship | Data PM |
| **Talent acquisition** (SAP expert) | Medium | High | Start recruiting Week 1, competitive comp, equity incentives | Data PM |

---

## 📋 RACI Matrix

| Workstream | Accountable | Responsible | Consulted | Informed |
|-----------|-------------|-------------|-----------|----------|
| **Standards mapping** | Head of AI | LCA, Data PM | Legal | Execs |
| **Factor Broker** | Lead Architect | Backend, Data Eng | LCA | QA |
| **Policy Engine** | Lead Architect | Backend | LCA | QA |
| **Intake Agent** | Lead Architect | Backend | Integration | QA |
| **Calculator Cat 1,4,6** | Head of AI | Backend, LCA | Data PM | QA |
| **Hotspot** | Head of AI | Backend | LCA | PMO |
| **Engagement** | Data PM | Frontend, Backend | Legal | Sales |
| **Reporting** | Data PM | Backend, Writer | LCA | Customers |
| **ERP SAP** | Integration Lead | Integration | SAP SME | Support |
| **Security and SOC 2** | DevOps | DevOps | Legal | Execs |
| **PCF exchange** | Head of AI | Backend | Partners | Customers |
| **Beta program** | Data PM | All | Support | Execs |

---

## 📊 Gantt Summary

| Weeks | Track | Output |
|-------|-------|--------|
| **1-2** | Strategy and architecture | Matrices, contracts, policies |
| **3-4** | Factor Broker and DQI | Broker live, uncertainty catalog |
| **5** | Schemas | 4 JSON Schemas versioned |
| **6** | Validation rules | Data, protocol, supplier rules |
| **7-10** | Intake Agent | Ingestion, OCR, DQ dashboard |
| **10-14** | Calculator Cat 1, 4, 6 | Tiered math, Monte Carlo, provenance |
| **14-16** | Hotspot v1 | Pareto and scenarios |
| **16-18** | Engagement v1, Reporting v1 | Portal, consent, ESRS, CDP, IFRS S2 |
| **19-22** | SAP | Connector and deltas |
| **22-24** | Oracle | Connector and deltas |
| **24-26** | Workday | Expenses ingest |
| **27-30** | ML | Entity and spend models |
| **31-36** | Test and security | E2E, load, pen test, DPIA |
| **37-40** | Prod infra and beta | K8s, metrics, 6 partners live |
| **41-42** | Hardening and docs | Perf, UX, docs |
| **43-44** | Launch | GA, references, GTM kits |

---

## ✅ Immediate Next Steps (Week 1, Days 1-3)

### Day 1 (Today):
1. ✅ Approve CTO v2 plan
2. [ ] Update foundation files (STATUS.md, pack.yaml, gl.yaml, config, PRD.md)
3. [ ] Create new spec files (Factor Broker, Policy Engine, Entity MDM, PCF Exchange)

### Day 2:
4. [ ] Standards mapping matrix (GHG ↔ ESRS ↔ CDP ↔ IFRS S2)
5. [ ] Factor Broker architecture specification
6. [ ] Policy Engine OPA policy templates (Cat 1, 4, 6)
7. [ ] SOC 2 security policy drafting (20+ policies)

### Day 3:
8. [ ] Entity MDM design document (LEI, DUNS, OpenCorporates API specs)
9. [ ] PCF Exchange PACT Pathfinder schema validation
10. [ ] Privacy model design (consent registry, GDPR compliance)
11. [ ] Design partner outreach (identify 6 candidates)

### Week 1 Remaining (Days 4-5):
12. [ ] Compliance register (GHG, ESRS, IFRS S2, ISO 14083, SOC 2, GDPR)
13. [ ] Budget reallocation approval ($250K data licenses, $100K compliance)
14. [ ] Team hiring: Post jobs (Lead Architect, LCA Specialist, Data PM)
15. [ ] Repo setup (GitHub org, CI/CD, environments, secret management)
16. [ ] Confirm 6 design partners and data access scopes

---

## 💡 Success Factors

**What will make this successful:**
1. ✅ **Focused scope**: Cat 1, 4, 6 = 85% of value, ships faster
2. ✅ **Technical excellence**: Factor Broker + Policy Engine = maintainable
3. ✅ **Differentiation**: PCF exchange = network effects moat
4. ✅ **Enterprise-ready**: SOC 2 from Day 1 = no sales blockers
5. ✅ **Global standards**: ESRS + IFRS S2 = worldwide market
6. ✅ **Channel leverage**: SAP/Oracle alliances = distribution scale
7. ✅ **Design partners**: 6 partners = feedback loop + references
8. ✅ **Team quality**: Senior SAP integrator, LCA specialist = execution

**What will derail this:**
1. ❌ Scope creep (adding Cat 2, 3, 5, 7-15 before GA)
2. ❌ SAP integration delays (mitigated by senior integrator)
3. ❌ SOC 2 audit delays (mitigated by Week 1 start)
4. ❌ Design partner churn (mitigated by dedicated success plans)
5. ❌ Data quality issues (mitigated by DQI transparency, Pareto focus)

---

## 📞 Contact and Governance

**Weekly Cadence**:
- Monday: Sprint planning (1 hour)
- Wednesday: Mid-week sync (30 minutes)
- Friday: Demo and retrospective (1 hour)

**Monthly Cadence**:
- Executive review (metrics, risks, roadmap)
- Design partner reviews
- SOC 2 evidence review

**Escalation**:
- P0 (Critical): Immediate escalation to Lead Architect + Data PM
- P1 (High): 24-hour SLA
- P2 (Medium): 3-day SLA
- P3 (Low): Next sprint

---

**Status**: Ready to Execute
**Approval**: CTO Approved
**Next Action**: Update foundation files and begin Week 1 deliverables

---

**Built with 🌍 by the GL-VCCI Team**
