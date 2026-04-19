# PRD: APP-001 -- GL-CSRD-APP v1.1 Enhancement

## Document Info
| Field | Value |
|-------|-------|
| PRD ID | PRD-APP-001 |
| Application | GL-CSRD-APP v1.1 |
| Priority | P0 (Critical) |
| Version | 1.1.0 |
| Status | Approved |
| Author | GL-ProductManager + GL-RegulatoryIntelligence |
| Date | 2026-02-28 |
| Base | GL-CSRD-APP v1.0 (applications/GL-CSRD-APP/CSRD-Reporting-Platform/) |

---

## 1. Overview

### 1.1 Purpose
Enhance GL-CSRD-APP from v1.0 (95% scaffolded but with critical gaps) to v1.1 (production-ready) by:
1. **Completing all 12 ESRS standards** with full data point definitions (50 -> 1,082+)
2. **Full XBRL/iXBRL taxonomy** mapping and validation (22 -> 1,082+ mappings)
3. **Multi-language support** for DE (German), FR (French), ES (Spanish)
4. **Fixing placeholder implementations** (PDF generation, RAG indexing, audit rules)
5. **Expanding calculation formulas** (42 -> 520+) and compliance rules (25 -> 215+)

### 1.2 Current State (v1.0 Gap Analysis)

| Component | v1.0 State | v1.1 Target |
|-----------|-----------|-------------|
| ESRS data points | 50 defined (4.6%) | 1,082+ (100%) |
| ESRS-1 General Requirements | 0 data points | 200 data points |
| ESRS-2 General Disclosures | 0 data points | 142 data points |
| XBRL taxonomy mappings | 22 hardcoded | 1,082+ from EFRAG taxonomy |
| XBRL validation | 5 mandatory elements | Full taxonomy validation |
| PDF generation | Placeholder (text file) | Real PDF with WeasyPrint |
| Multi-language | English-only templates | EN/DE/FR/ES full support |
| i18n infrastructure | None | gettext/babel framework |
| Calculation formulas | 42 defined | 520+ formulas |
| Compliance rules | ~25 rules | 215+ rules |
| Framework mappings | 28 mappings | 350+ mappings |
| RAG indexing | Placeholder (pass) | Functional implementation |
| Audit rule patterns | 4 types | 10+ pattern types |
| NarrativeGeneratorAI | Standalone, not integrated | Integrated into pipeline |
| Security defaults | Disabled | Enabled for production |

### 1.3 Regulatory Context
- **CSRD** (Corporate Sustainability Reporting Directive) 2022/2464
- **ESRS** (European Sustainability Reporting Standards) -- adopted July 2023
- **ESEF** (European Single Electronic Format) -- iXBRL required
- **First reporting**: FY2024 for large public-interest entities
- **Languages**: Reports must be in official EU language of filing jurisdiction

---

## 2. Enhancement Areas

### 2.1 ESRS Data Point Expansion

#### 2.1.1 Cross-Cutting Standards (NEW - currently 0%)
**ESRS-1: General Requirements** (~200 data points)
- General basis for preparation (BP-1, BP-2)
- Qualities of information
- Double materiality assessment documentation
- Value chain information
- Time horizons (short/medium/long-term)
- Preparation and presentation of sustainability statements

**ESRS-2: General Disclosures** (~142 data points)
- GOV-1 to GOV-5: Governance (role of admin/management bodies, sustainability expertise, incentives, due diligence, risk management)
- SBM-1 to SBM-3: Strategy & business model (market position, stakeholder interests, material impacts)
- IRO-1 to IRO-2: Impact, risk, opportunity identification and assessment
- MDR-P, MDR-A, MDR-M, MDR-T: Minimum disclosure requirements for policies/actions/metrics/targets

#### 2.1.2 Environmental Standards (E1-E5) Expansion
Each standard needs full data point coverage per EFRAG implementation guidance:

**E1 Climate Change** (50 -> complete with sub-disclosures):
- E1-1: Transition plan for climate change mitigation
- E1-2: Policies (IRO-level and topical)
- E1-3: Actions and resources
- E1-4: Targets (absolute, intensity, SBTi)
- E1-5: Energy consumption and mix (total, renewable %, by source)
- E1-6: Gross GHG Scope 1/2/3 (by category, with biogenic)
- E1-7: GHG removals and carbon credits
- E1-8: Internal carbon pricing
- E1-9: Financial effects (physical and transition risks, opportunities)

**E2 Pollution** (45 -> complete):
- E2-1 to E2-6: Policies, actions, targets, pollution of air/water/soil, substances of concern, microplastics

**E3 Water** (40 -> complete):
- E3-1 to E3-5: Policies, actions, targets, water consumption, water in areas of stress

**E4 Biodiversity** (55 -> complete):
- E4-1 to E4-6: Transition plan, policies, actions, targets, impact metrics, financial effects

**E5 Circular Economy** (60 -> complete):
- E5-1 to E5-6: Policies, actions, targets, resource inflows/outflows, waste

#### 2.1.3 Social Standards (S1-S4) Expansion
**S1 Own Workforce** (180 -> complete, largest standard):
- S1-1 to S1-17: Policies, engagement, remediation, characteristics (headcount, diversity, wages), health & safety (incidents, ill health), work-life balance, training, adequate wages, social protection, persons with disabilities, freedom of association, collective bargaining

**S2 Value Chain Workers** (80 -> complete):
- S2-1 to S2-5: Policies, engagement, remediation, material impacts/risks, actions

**S3 Affected Communities** (65 -> complete):
- S3-1 to S3-5: Policies, engagement, remediation, material impacts, actions

**S4 Consumers** (70 -> complete):
- S4-1 to S4-5: Policies, engagement, remediation, material impacts, actions

#### 2.1.4 Governance Standard (G1) Expansion
**G1 Business Conduct** (95 -> complete):
- G1-1 to G1-6: Policies, corporate culture, anti-corruption, political engagement, payment practices, supplier relationships

### 2.2 XBRL/iXBRL Enhancement

#### 2.2.1 Full EFRAG Taxonomy Mapping
- Map all 1,082+ data points to EFRAG ESRS XBRL taxonomy elements
- Include both mandatory and voluntary disclosure elements
- Support typed dimensions (e.g., by country, by GHG scope, by energy source)
- Support filing indicators per standard

#### 2.2.2 iXBRL Document Generation
- Generate valid iXBRL (inline XBRL in HTML/XHTML)
- ESEF Reporting Manual compliance
- Proper context creation (reporting entity, period, dimensions)
- Unit references (ISO 4217 currencies, custom units for emissions)
- Calculation linkbase validation

#### 2.2.3 XBRL Validation Engine
- Taxonomy-level element name validation
- Calculation linkbase consistency checks
- Presentation linkbase ordering
- Dimension/hypercube validation
- Filing indicator validation
- ESEF RTS (Regulatory Technical Standards) compliance

### 2.3 Multi-Language Support (DE/FR/ES)

#### 2.3.1 i18n Infrastructure
- Build translation framework using Python gettext/babel patterns
- JSON-based locale files for each language
- Locale-aware number formatting (1,000.00 vs 1.000,00)
- Locale-aware date formatting (MM/DD/YYYY vs DD.MM.YYYY)
- Currency formatting per locale

#### 2.3.2 Translation Files
Create translation catalogs for:
- **ESRS standard names and descriptions** (all 12 standards)
- **Data point labels** (1,082+ entries)
- **Compliance rule messages** (215+ entries)
- **Report narrative templates** (section headers, boilerplate)
- **UI labels and error messages**
- **Regulatory terminology glossary** (CSRD-specific terms)

#### 2.3.3 Language-Specific Features
- **German (DE)**: Official ESRS translations from EFRAG, formal "Sie" form, German number format (1.000,00), Datum format (DD.MM.JJJJ)
- **French (FR)**: Official ESRS translations, formal register, French number format (1 000,00), date format (DD/MM/AAAA)
- **Spanish (ES)**: Official ESRS translations, formal register, Spanish number format (1.000,00), fecha format (DD/MM/AAAA)

### 2.4 Additional Enhancements

#### 2.4.1 PDF Report Generation
- Replace placeholder with WeasyPrint-based PDF generation
- CSRD report template with proper formatting
- Cover page, table of contents, section navigation
- Charts and tables for metrics
- Multi-language PDF generation

#### 2.4.2 Expanded Formulas and Rules
- Expand from 42 to 520+ deterministic formulas
- Expand compliance rules from ~25 to 215+
- Expand framework mappings from 28 to 350+

#### 2.4.3 Security Defaults
- Enable encryption, PII detection, data anonymization
- Production-ready security configuration

---

## 3. File Structure (New/Modified Files)

### 3.1 New Files

```
applications/GL-CSRD-APP/CSRD-Reporting-Platform/
    i18n/
        __init__.py                              (~200 lines)
        locale_manager.py                        (~800 lines)
        number_formatter.py                      (~400 lines)
        date_formatter.py                        (~300 lines)
        locales/
            en/
                esrs_labels.json                 (~2,500 lines)
                report_templates.json            (~800 lines)
                messages.json                    (~400 lines)
                glossary.json                    (~300 lines)
            de/
                esrs_labels.json                 (~2,500 lines)
                report_templates.json            (~800 lines)
                messages.json                    (~400 lines)
                glossary.json                    (~300 lines)
            fr/
                esrs_labels.json                 (~2,500 lines)
                report_templates.json            (~800 lines)
                messages.json                    (~400 lines)
                glossary.json                    (~300 lines)
            es/
                esrs_labels.json                 (~2,500 lines)
                report_templates.json            (~800 lines)
                messages.json                    (~400 lines)
                glossary.json                    (~300 lines)
    xbrl/
        __init__.py                              (~100 lines)
        taxonomy_mapper.py                       (~2,000 lines)
        ixbrl_generator.py                       (~1,800 lines)
        xbrl_validator.py                        (~1,500 lines)
        taxonomy_data/
            efrag_esrs_taxonomy.json             (~3,000 lines)
            filing_indicators.json               (~200 lines)
            calculation_linkbase.json             (~500 lines)
    pdf/
        __init__.py                              (~50 lines)
        pdf_generator.py                         (~1,200 lines)
        report_template.html                     (~500 lines)
        styles.css                               (~300 lines)
```

### 3.2 Modified Files

```
    data/
        esrs_data_points.json                    (50 -> 1,082+ entries)
        esrs_formulas.yaml                       (42 -> 520+ formulas)
        framework_mappings.json                  (28 -> 350+ mappings)
    rules/
        esrs_compliance_rules.yaml               (25 -> 215+ rules)
        xbrl_validation_rules.yaml               (enhanced)
    agents/
        reporting_agent.py                       (integrate XBRL/PDF/i18n modules)
        materiality_agent.py                     (implement RAG indexing)
        audit_agent.py                           (expand rule patterns)
    config/
        csrd_config.yaml                         (security defaults, i18n config)
```

---

## 4. Development Tasks (Parallel Build Plan)

### Task Group A: ESRS Data Expansion (Agent 1)
- A1: Build complete ESRS-1 data points (200 entries)
- A2: Build complete ESRS-2 data points (142 entries)
- A3: Expand E1-E5 data points to full coverage
- A4: Expand S1-S4 data points to full coverage
- A5: Expand G1 data points to full coverage
- A6: Update esrs_formulas.yaml (42 -> 520+)
- A7: Update esrs_compliance_rules.yaml (25 -> 215+)
- A8: Update framework_mappings.json (28 -> 350+)

### Task Group B: XBRL/iXBRL Engine (Agent 2)
- B1: Build EFRAG ESRS taxonomy data (efrag_esrs_taxonomy.json)
- B2: Build taxonomy_mapper.py (data point -> XBRL element mapping)
- B3: Build ixbrl_generator.py (iXBRL document generation)
- B4: Build xbrl_validator.py (comprehensive validation)
- B5: Build filing_indicators.json and calculation_linkbase.json

### Task Group C: Multi-Language i18n (Agent 3)
- C1: Build i18n infrastructure (locale_manager.py, formatters)
- C2: Build EN locale files (base translations)
- C3: Build DE locale files (German translations)
- C4: Build FR locale files (French translations)
- C5: Build ES locale files (Spanish translations)

### Task Group D: PDF & Agent Fixes (Agent 4)
- D1: Build PDF generator (WeasyPrint-based)
- D2: Fix MaterialityAgent RAG indexing
- D3: Expand AuditAgent rule patterns
- D4: Integrate NarrativeGeneratorAI into pipeline
- D5: Update config with security defaults and i18n settings

---

## 5. Acceptance Criteria

1. All 12 ESRS standards with 1,082+ data points defined
2. Full XBRL/iXBRL taxonomy mapping for all data points
3. Valid iXBRL document generation passing ESEF validation
4. Multi-language support: EN, DE, FR, ES with proper locale formatting
5. PDF report generation with professional formatting
6. 520+ deterministic calculation formulas
7. 215+ compliance rules with expanded pattern matching
8. 350+ framework mappings (TCFD, GRI, SASB -> ESRS)
9. RAG indexing functional in MaterialityAgent
10. Production security defaults enabled
11. All new code with comprehensive test coverage
