# PRD: PACK-020 Battery Passport Prep Pack

**Pack Name:** PACK-020 Battery Passport Prep Pack
**Category:** EU Compliance Packs (Solution Packs)
**Tier:** Standalone
**Version:** 1.0.0
**Author:** GreenLang Platform Team
**Date:** March 2026
**Status:** Approved

---

## 1. Executive Summary

PACK-020 delivers a comprehensive battery passport readiness and compliance platform for the EU Battery Regulation - Regulation (EU) 2023/1542. This regulation establishes sustainability, safety, labelling, and information requirements for batteries placed on the EU market, including the groundbreaking Digital Product Passport (Battery Passport) requirement.

This pack provides 8 calculation engines, 8 workflows, 8 report templates, 10 integrations, and 6 sector presets enabling battery manufacturers, importers, and economic operators to prepare battery passports, calculate carbon footprints, track recycled content, manage supply chain due diligence, and ensure compliance with all Battery Regulation requirements.

### Key Capabilities
- Battery carbon footprint declaration per Articles 7 and Annex II
- Recycled content calculation and tracking per Article 8
- Battery passport data compilation per Articles 77-78 and Annex XIII
- Supply chain due diligence for raw materials per Article 48 (cobalt, lithium, nickel, natural graphite)
- Performance and durability information per Article 10 and Annex IV
- Labelling and marking compliance per Articles 13-14
- Battery categorisation and scope determination (portable, LMT, SLI, industrial, EV)
- End-of-life management and collection rate tracking per Articles 56-71
- QR code and data carrier requirements per Article 77(3)
- Conformity assessment preparation per Articles 17-22
- Interoperability with ESPR Digital Product Passport framework
- GBA Battery Passport and Catena-X/Battery Pass standard alignment

---

## 2. Regulatory Basis

### 2.1 Primary Regulation
**EU Battery Regulation**
- **Reference:** Regulation (EU) 2023/1542
- **Adopted:** 12 July 2023
- **Published:** OJ L 191, 28 July 2023
- **Entry into force:** 17 August 2023
- **Replaces:** Directive 2006/66/EC (Batteries Directive)

### 2.2 Key Application Dates

| Requirement | Date | Article |
|-------------|------|---------|
| General requirements (labelling, hazardous substances) | 18 August 2024 | Art 6, 13-14 |
| Carbon footprint declaration (EV & industrial >2kWh) | 18 February 2025 | Art 7(1) |
| Carbon footprint performance class | 18 August 2026 | Art 7(2) |
| Carbon footprint maximum threshold | 18 February 2028 | Art 7(3) |
| Recycled content documentation (cobalt, lithium, nickel, lead) | 18 August 2028 | Art 8(1) |
| Recycled content minimum targets | 18 August 2031 | Art 8(4) |
| Performance & durability (portable general) | 18 August 2025 | Art 10 |
| Performance & durability (EV batteries) | 18 August 2025 | Art 10, Annex IV |
| Battery passport (EV, LMT, industrial >2kWh) | 18 February 2027 | Art 77 |
| Supply chain due diligence | 18 August 2025 | Art 48 |
| Collection rates (portable: 63%) | 31 December 2027 | Art 59 |
| Collection rates (portable: 73%) | 31 December 2030 | Art 59 |
| Collection rates (LMT: 51%) | 31 December 2028 | Art 59 |
| Collection rates (LMT: 61%) | 31 December 2031 | Art 59 |
| Material recovery (lithium: 50%) | 31 December 2027 | Art 71 |
| Material recovery (lithium: 80%) | 31 December 2031 | Art 71 |

### 2.3 Battery Categories

| Category | Definition | Examples |
|----------|-----------|----------|
| Portable | Sealed, ≤5 kg, not industrial/EV/SLI/LMT | Consumer electronics, AA/AAA |
| LMT (Light Means of Transport) | Sealed, ≤25 kg, for e-bikes/scooters | E-bike, e-scooter batteries |
| SLI (Starting, Lighting, Ignition) | For vehicle SLI functions | Car starter batteries |
| Industrial | >2 kWh, not EV/LMT/SLI/portable | Stationary storage, UPS |
| EV (Electric Vehicle) | For vehicle traction in hybrid/electric | BEV, PHEV traction batteries |

### 2.4 Battery Passport Data Requirements (Annex XIII)

**A. General Battery Information:**
1. Battery manufacturer identification
2. Manufacturing plant information
3. Battery model and batch/serial number
4. Date of placing on market
5. Battery weight and dimensions
6. Battery category and chemistry

**B. Carbon Footprint Information:**
1. Carbon footprint of manufacturing
2. Carbon footprint performance class
3. Share of carbon footprint per lifecycle stage (raw material, manufacturing, distribution, end-of-life)
4. Reference to carbon footprint calculation methodology
5. Link to EU carbon footprint declaration

**C. Supply Chain Due Diligence:**
1. Due diligence report reference
2. Third-party verification information
3. Supply chain risk mitigation policies
4. Information on cobalt, lithium, nickel, natural graphite sourcing

**D. Material Composition:**
1. Battery chemistry (cathode, anode, electrolyte)
2. Critical raw materials content
3. Hazardous substances (mercury, cadmium, lead)
4. Recycled content (cobalt, lithium, nickel, lead)

**E. Performance & Durability:**
1. Rated capacity (Ah)
2. Minimum/remaining capacity
3. Voltage (nominal, min, max)
4. Original/remaining power capability
5. Expected battery lifetime (cycles, calendar)
6. Temperature range (operating, storage)
7. C-rate capability
8. Round-trip energy efficiency
9. Internal resistance (initial, current)
10. State of Health (SoH) at reporting date
11. State of Charge (SoC)

**F. End-of-Life Information:**
1. Collection and recycling instructions
2. Dismantling and disassembly information
3. Safety information for handling
4. Role of end-users in waste prevention

### 2.5 Secondary Regulations and Standards
- **ESPR (EU) 2024/1781** - Ecodesign for Sustainable Products Regulation (DPP framework)
- **GBA Battery Passport** - Global Battery Alliance passport standard
- **Catena-X / Battery Pass** - Data exchange standards
- **ISO 14040/14044** - Life Cycle Assessment standards
- **IEC 62660** - Lithium-ion battery testing
- **UN 38.3** - Transport safety testing
- **OECD Due Diligence Guidance** - Mineral supply chains
- **PEFCR for batteries** - Product Environmental Footprint Category Rules
- **EU Taxonomy Regulation** - DNSH criteria for battery manufacturing

---

## 3. Architecture

### 3.1 Engines (8)

| # | Engine | Class | Purpose |
|---|--------|-------|---------|
| 1 | Carbon Footprint Engine | `CarbonFootprintEngine` | Calculate battery carbon footprint per Art 7, lifecycle stage breakdown (raw material extraction, manufacturing, distribution, end-of-life), performance class assignment, threshold compliance |
| 2 | Recycled Content Engine | `RecycledContentEngine` | Track and calculate recycled content percentages for cobalt, lithium, nickel, lead per Art 8, verify against minimum targets (2031 thresholds) |
| 3 | Battery Passport Engine | `BatteryPassportEngine` | Compile and validate all battery passport data fields per Annex XIII, generate QR code payload, validate completeness and data quality |
| 4 | Performance Durability Engine | `PerformanceDurabilityEngine` | Calculate and validate battery performance metrics per Art 10 and Annex IV - capacity, voltage, power, cycle life, efficiency, SoH, SoC |
| 5 | Supply Chain DD Engine | `SupplyChainDDEngine` | Assess supply chain due diligence for critical raw materials (cobalt, lithium, nickel, natural graphite) per Art 48, OECD alignment |
| 6 | Labelling Compliance Engine | `LabellingComplianceEngine` | Validate labelling and marking requirements per Art 13-14, CE marking, QR code, collection symbol, hazardous substance markings |
| 7 | End of Life Engine | `EndOfLifeEngine` | Track collection rates, recycling efficiency, material recovery rates per Art 56-71, calculate compliance against targets |
| 8 | Conformity Assessment Engine | `ConformityAssessmentEngine` | Assess readiness for conformity assessment per Art 17-22, EU declaration of conformity, technical documentation, notified body requirements |

### 3.2 Workflows (8)

| # | Workflow | Class | Phases |
|---|----------|-------|--------|
| 1 | Carbon Footprint Assessment | `CarbonFootprintWorkflow` | 4-phase: Data Collection -> LCA Calculation -> Performance Class -> Declaration Generation |
| 2 | Recycled Content Tracking | `RecycledContentWorkflow` | 4-phase: Material Inventory -> Content Calculation -> Target Comparison -> Documentation |
| 3 | Passport Compilation | `PassportCompilationWorkflow` | 5-phase: Data Gathering -> Validation -> Passport Assembly -> QR Generation -> Registry Submission |
| 4 | Performance Testing | `PerformanceTestingWorkflow` | 4-phase: Test Data Collection -> Metric Calculation -> Threshold Validation -> Report Generation |
| 5 | Due Diligence Assessment | `DueDiligenceAssessmentWorkflow` | 4-phase: Supplier Mapping -> Risk Assessment -> Mitigation Planning -> Audit Verification |
| 6 | Labelling Verification | `LabellingVerificationWorkflow` | 4-phase: Requirement Mapping -> Label Review -> Compliance Check -> Corrective Actions |
| 7 | End of Life Planning | `EndOfLifePlanningWorkflow` | 4-phase: Collection Assessment -> Recycling Targets -> Recovery Calculation -> Compliance Reporting |
| 8 | Regulatory Submission | `RegulatorySubmissionWorkflow` | 4-phase: Documentation Assembly -> Conformity Check -> Submission Package -> Registry Upload |

### 3.3 Templates (8)

| # | Template | Class | Purpose |
|---|----------|-------|---------|
| 1 | Carbon Footprint Declaration | `CarbonFootprintDeclarationTemplate` | Art 7 carbon footprint declaration document with lifecycle breakdown |
| 2 | Recycled Content Report | `RecycledContentReportTemplate` | Art 8 recycled content documentation and target tracking |
| 3 | Battery Passport Report | `BatteryPassportReportTemplate` | Complete battery passport data compilation per Annex XIII |
| 4 | Performance Report | `PerformanceReportTemplate` | Battery performance and durability data per Annex IV |
| 5 | Due Diligence Report | `DueDiligenceReportTemplate` | Art 48 supply chain due diligence findings |
| 6 | Labelling Compliance Report | `LabellingComplianceReportTemplate` | Art 13-14 labelling requirement compliance status |
| 7 | End of Life Report | `EndOfLifeReportTemplate` | Collection rates, recycling efficiency, material recovery tracking |
| 8 | Battery Regulation Scorecard | `BatteryRegulationScorecardTemplate` | Executive dashboard with article-by-article compliance status |

### 3.4 Integrations (10)

| # | Integration | Class | Purpose |
|---|-------------|-------|---------|
| 1 | Pack Orchestrator | `BatteryPassportOrchestrator` | Master pipeline orchestrating all battery passport assessment phases |
| 2 | MRV Bridge | `MRVBridge` | Routes MRV emission data for carbon footprint calculation |
| 3 | CSRD Pack Bridge | `CSRDPackBridge` | Maps ESRS E1/E2/E5 disclosures to battery regulation requirements |
| 4 | Supply Chain Bridge | `SupplyChainBridge` | Links supply chain agents for mineral sourcing due diligence |
| 5 | EUDR Bridge | `EUDRBridge` | Connects EUDR due diligence for deforestation-free supply chains |
| 6 | Taxonomy Bridge | `TaxonomyBridge` | Validates DNSH criteria for battery manufacturing activities |
| 7 | CSDDD Bridge | `CSDDDBridge` | Links CSDDD due diligence requirements for battery supply chains |
| 8 | Data Bridge | `DataBridge` | Routes data intake agents for battery test data, BOM data |
| 9 | Health Check | `BatteryPassportHealthCheck` | System verification across all engines and bridges |
| 10 | Setup Wizard | `BatteryPassportSetupWizard` | Guided configuration for battery type, chemistry, supply chain |

### 3.5 Presets (6)

| # | Preset | Sector | Key Focus |
|---|--------|--------|-----------|
| 1 | EV Battery | EV_BATTERY | Traction batteries for BEV/PHEV, full passport required |
| 2 | Industrial Storage | INDUSTRIAL_STORAGE | Stationary storage >2kWh, carbon footprint + passport |
| 3 | LMT Battery | LMT_BATTERY | E-bike/e-scooter batteries, passport required |
| 4 | Portable Battery | PORTABLE_BATTERY | Consumer electronics, labelling + collection focus |
| 5 | SLI Battery | SLI_BATTERY | Automotive starter batteries, performance + recycled content |
| 6 | Cell Manufacturer | CELL_MANUFACTURER | Battery cell production, carbon footprint + due diligence |

---

## 4. Data Models

### 4.1 Key Enums
- `BatteryCategory` - EV, INDUSTRIAL, LMT, PORTABLE, SLI
- `BatteryChemistry` - NMC, NCA, LFP, NMC811, NMC622, NMC532, LMO, LTO, LEAD_ACID, NIMH, ALKALINE, ZINC_AIR, SOLID_STATE, SODIUM_ION
- `LifecycleStage` - RAW_MATERIAL_EXTRACTION, MANUFACTURING, DISTRIBUTION, USE_PHASE, END_OF_LIFE, RECYCLING
- `CriticalRawMaterial` - COBALT, LITHIUM, NICKEL, NATURAL_GRAPHITE, MANGANESE
- `CarbonFootprintClass` - CLASS_A, CLASS_B, CLASS_C, CLASS_D, CLASS_E (best to worst)
- `ComplianceStatus` - COMPLIANT, PARTIALLY_COMPLIANT, NON_COMPLIANT, NOT_APPLICABLE, PENDING
- `PassportField` - All Annex XIII data fields
- `DueDiligenceRisk` - VERY_HIGH, HIGH, MEDIUM, LOW, NEGLIGIBLE
- `LabelElement` - CE_MARKING, QR_CODE, COLLECTION_SYMBOL, CAPACITY_LABEL, HAZARDOUS_SUBSTANCE, BATTERY_CHEMISTRY, CARBON_FOOTPRINT, SEPARATE_COLLECTION
- `RecoveryMaterial` - COBALT, LITHIUM, NICKEL, COPPER, LEAD
- `ConformityModule` - MODULE_A, MODULE_B, MODULE_C, MODULE_D, MODULE_E, MODULE_G, MODULE_H

### 4.2 Key Models (Pydantic BaseModel)
- `BatteryProfile` - battery_id, category, chemistry, manufacturer, model, weight_kg, capacity_ah, voltage_nominal, energy_kwh, production_date, placing_on_market_date
- `CarbonFootprintResult` - total_co2e_kg, per_kwh_co2e, lifecycle_breakdown, performance_class, threshold_compliant, methodology_reference
- `RecycledContentData` - cobalt_pct, lithium_pct, nickel_pct, lead_pct, target_year, targets_met
- `PassportData` - general_info, carbon_footprint, supply_chain_dd, material_composition, performance_durability, end_of_life
- `PerformanceMetrics` - rated_capacity_ah, min_capacity_ah, remaining_capacity_pct, voltage_nominal, voltage_min, voltage_max, power_capability_w, cycle_life, calendar_life_years, efficiency_pct, internal_resistance_mohm, soh_pct, soc_pct, c_rate, temperature_range
- `DueDiligenceAssessment` - materials_assessed, risk_level, oecd_step_compliance, third_party_verified, mitigation_measures
- `LabelRequirement` - element, required, present, compliant, notes
- `EndOfLifeMetrics` - collection_rate_pct, recycling_efficiency_pct, material_recovery (per material), dismantling_info_available

---

## 5. Carbon Footprint Calculation Methodology

### 5.1 Lifecycle Stages (per PEFCR)
1. **Raw Material Extraction** - Mining, refining of cathode/anode materials
2. **Cell Manufacturing** - Electrode preparation, cell assembly, formation
3. **Battery Pack Assembly** - Module assembly, BMS, cooling system
4. **Distribution** - Transport from factory to market
5. **Use Phase** - Electricity losses during charging/discharging (excluded per Art 7)
6. **End of Life** - Collection, dismantling, recycling, disposal

### 5.2 Performance Classes
- **Class A**: ≤ 20th percentile of batteries on market
- **Class B**: > 20th and ≤ 40th percentile
- **Class C**: > 40th and ≤ 60th percentile
- **Class D**: > 60th and ≤ 80th percentile
- **Class E**: > 80th percentile (worst performing)

### 5.3 Recycled Content Targets

| Material | 2028 (Documentation) | 2031 (Minimum) | 2036 (Increased) |
|----------|----------------------|----------------|-------------------|
| Cobalt | Report actual | 16% | 26% |
| Lithium | Report actual | 6% | 12% |
| Nickel | Report actual | 6% | 15% |
| Lead | Report actual | 85% | 85% |

---

## 6. Agent Dependencies

### 6.1 Total Agent Count: 54
- **MRV Agents:** 30 (Scope 1-3 emissions for carbon footprint LCA)
- **Data Agents:** 14 (BOM extraction, test data, supply chain data)
- **Foundation Agents:** 10 (Orchestrator, schema, units, citations)

### 6.2 Key Agent Reuse
- **MRV Scope 1**: Stationary/Mobile combustion for manufacturing emissions
- **MRV Scope 2**: Location/Market-based for electricity in manufacturing
- **MRV Scope 3 Cat 1**: Purchased goods (raw materials)
- **MRV Scope 3 Cat 4**: Upstream transportation
- **MRV Scope 3 Cat 12**: End-of-life treatment
- **AGENT-DATA-008**: Supplier Questionnaire Processor (due diligence)
- **AGENT-DATA-009**: Spend Data Categorizer (material cost tracking)
- **AGENT-EUDR supply chain**: For mineral sourcing traceability

---

## 7. Testing Strategy

### 7.1 Test Structure
- `conftest.py` - Dynamic module loading with `pack020_test.*` namespace
- 8 engine test files (~50 tests each)
- `test_templates.py` - All 8 templates (~45 tests)
- `test_workflows.py` - All 8 workflows (~40 tests)
- `test_integrations.py` - All 10 integrations (~35 tests)
- `test_config.py` - PackConfig and presets (~30 tests)
- `test_manifest.py` - pack.yaml validation (~25 tests)
- `test_demo.py` - Demo config (~15 tests)
- `test_agent_integration.py` - Agent bridges (~30 tests)
- `test_e2e.py` - End-to-end pipeline (~20 tests)

### 7.2 Test Targets
- Total tests: 700+
- Pass rate: 100%
- Coverage: All engines, workflows, templates, integrations, config

---

## 8. Performance Targets

| Component | Metric | Target |
|-----------|--------|--------|
| Carbon Footprint Calculation | Max batteries assessed | 10,000 |
| Passport Compilation | Max passports generated | 50,000 |
| Recycled Content Tracking | Max material lots | 100,000 |
| Performance Validation | Max test records | 500,000 |
| Supply Chain DD | Max suppliers assessed | 5,000 |
| Cache hit ratio | Overall | 65% |
| Memory ceiling | Peak | 8,192 MB |

---

## 9. Security & Access Control

- **Authentication:** JWT (RS256)
- **Authorization:** RBAC with battery-level access control
- **Encryption at rest:** AES-256-GCM
- **Encryption in transit:** TLS 1.3
- **Audit logging:** All passport activities logged
- **PII redaction:** Supply chain partner data protected
- **Required roles:** battery_manager, passport_administrator, carbon_footprint_analyst, recycled_content_specialist, due_diligence_officer, quality_engineer, labelling_specialist, compliance_officer, external_auditor, admin

---

## 10. Compliance Timeline

| Phase | Date | Requirement | Impact |
|-------|------|-------------|--------|
| Phase 1 | 18 Feb 2025 | Carbon footprint declaration | EV + industrial >2kWh |
| Phase 2 | 18 Aug 2025 | Performance & durability labels | All categories |
| Phase 3 | 18 Aug 2025 | Supply chain due diligence | All batteries |
| Phase 4 | 18 Aug 2026 | Carbon footprint performance class | EV + industrial >2kWh |
| Phase 5 | 18 Feb 2027 | Battery passport mandatory | EV, LMT, industrial >2kWh |
| Phase 6 | 18 Feb 2028 | Carbon footprint max threshold | EV + industrial >2kWh |
| Phase 7 | 18 Aug 2028 | Recycled content documentation | All rechargeable |
| Phase 8 | 31 Dec 2030 | Collection rate 73% (portable) | Producers |
| Phase 9 | 18 Aug 2031 | Recycled content minimum targets | All rechargeable |

---

*End of PRD*
