# FuelAgentAI v2 - Data Acquisition Plan

**Version:** 1.0
**Date:** 2025-10-24
**Owner:** GreenLang Framework Team
**Status:** Planning Phase

---

## Executive Summary

This document outlines the strategy for acquiring emission factor data to support FuelAgentAI v2's enhanced capabilities including multi-gas reporting (CO2, CH4, N2O), provenance tracking, data quality scoring, and regulatory compliance (CSRD, CDP, GHG Protocol).

**Target Coverage:** 50+ fuel types × 20+ countries × 3 scopes = ~3,000 emission factors
**Estimated Annual Cost:** $8,000 - $15,000 (licenses + maintenance)
**Timeline:** 12 weeks (coordinated with development phases)

---

## 1. Current State Assessment

### Existing Data (v1)
- **Source:** `greenlang/data/emission_factors.py`
- **Coverage:** 3 regions (US, EU, UK) × ~20 fuel types × ~3 units = ~180 factors
- **Format:** Single CO2e scalar values (no CH4/N2O breakdown)
- **Provenance:** None (unknown sources, no versions, no dates)
- **Quality:** Unknown confidence, uncertainty, or methodology

### Critical Gaps
1. ❌ No source attribution (can't defend in audit)
2. ❌ No multi-gas breakdown (can't meet CSRD Annex II)
3. ❌ No versioning (can't track updates)
4. ❌ No uncertainty ranges (can't quantify risk)
5. ❌ Limited regional coverage (US/EU/UK only)
6. ❌ No upstream (WTT) factors (Scope 3 incomplete)

---

## 2. Data Requirements (v2)

### A. Factor Metadata (per factor)
```json
{
  "factor_id": "EF:US:diesel:2025:v1",
  "fuel_type": "diesel",
  "unit": "gallons",
  "geography": "US",
  "geography_level": "country",

  "co2_kg_per_unit": 10.18,
  "ch4_kg_per_unit": 0.00082,
  "n2o_kg_per_unit": 0.000164,

  "gwp_set": "IPCC_AR6_100",
  "co2e_kg_per_unit": 10.21,

  "scope": "1",
  "boundary": "combustion",

  "source_org": "EPA",
  "source_publication": "Emission Factors Hub 2024",
  "source_year": 2024,
  "methodology": "IPCC_Tier_1",

  "valid_from": "2024-01-01",
  "valid_to": "2024-12-31",
  "version": "v1",

  "uncertainty_95ci": 0.05,
  "dqs_temporal": 5,
  "dqs_geographical": 4,
  "dqs_technological": 4,
  "dqs_representativeness": 4,
  "dqs_methodological": 5,

  "license": "CC0-1.0",
  "citation": "EPA (2024). Emission Factors for Greenhouse Gas Inventories...",
  "redistribution_allowed": true
}
```

### B. Coverage Targets

| Category | Target | Priority | Notes |
|----------|--------|----------|-------|
| **Fuel Types** | 50+ | P0 | Fossil, renewable, bio, synthetic |
| **Geographies** | 25+ countries | P0 | Focus: US, EU27, UK, IN, CN, JP, AU |
| **Units** | 30+ | P0 | Energy, volume, mass, count |
| **Scopes** | Scope 1, 2, 3 | P0 | Full coverage for Scope 1, partial for 3 |
| **Boundaries** | Combustion, WTT, WTW | P1 | Lifecycle analysis |
| **Time Coverage** | 2020-2025 | P1 | Historical + current |
| **GWP Variants** | AR6 100yr, AR6 20yr, AR5 100yr | P0 | Regulatory flexibility |

---

## 3. Data Sources

### Tier 1: Free, Open-License Sources (P0)

#### 3.1 EPA Emission Factors Hub (US)
- **URL:** https://www.epa.gov/climateleadership/ghg-emission-factors-hub
- **Coverage:** US fuels, electricity (eGRID), mobile sources
- **License:** Public Domain (US Government Work)
- **Cost:** FREE
- **Update Frequency:** Annual (Q2)
- **Multi-Gas:** ✅ Yes (CO2, CH4, N2O by fuel)
- **Quality:** High (IPCC Tier 1-3)
- **Redistribution:** ✅ Allowed
- **Formats:** Excel, CSV, JSON (via API)

**Key Datasets:**
- Stationary Combustion (natural gas, coal, fuel oil, propane, etc.)
- Mobile Combustion (gasoline, diesel, jet fuel, marine fuel)
- eGRID (electricity by subregion, ~20 grid zones)
- Fugitive Emissions (natural gas transmission, venting)

**Action Items:**
- [ ] Download 2024 dataset (Q3 release)
- [ ] Parse Excel → Python dict structure
- [ ] Map EPA fuel codes → GreenLang fuel_type taxonomy
- [ ] Extract CH4/N2O vectors per fuel
- [ ] Document methodology references

---

#### 3.2 IPCC Emission Factor Database (Global)
- **URL:** https://www.ipcc-nggip.iges.or.jp/EFDB/
- **Coverage:** Global default factors by fuel and technology
- **License:** Open (IPCC Guidelines Annex)
- **Cost:** FREE
- **Update Frequency:** Irregular (2019 Refinement current)
- **Multi-Gas:** ✅ Yes (CO2, CH4, N2O, HFCs, etc.)
- **Quality:** Moderate-High (Tier 1 defaults)
- **Redistribution:** ✅ Allowed (with citation)
- **Formats:** Web interface, Excel downloads

**Key Datasets:**
- Energy (stationary combustion by fuel and technology)
- Transport (by vehicle type and fuel)
- Fugitive Emissions (coal mining, oil/gas systems)
- Default vs. Country-Specific factors

**Action Items:**
- [ ] Extract Tier 1 default factors (fallback when country-specific unavailable)
- [ ] Map IPCC fuel categories → GreenLang taxonomy
- [ ] Document uncertainty ranges from IPCC Guidelines Vol 2
- [ ] Create precedence rule: Country-specific > IPCC defaults

---

#### 3.3 IEA CO2 Emissions from Fuel Combustion (Global)
- **URL:** https://www.iea.org/data-and-statistics/data-product/emissions-factors-2024
- **Coverage:** 150+ countries, electricity and fuels
- **License:** CC BY 4.0 (Free tier: country aggregates)
- **Cost:** FREE (country-level) / $500/year (detailed)
- **Update Frequency:** Annual (Q3)
- **Multi-Gas:** ⚠️ Partial (CO2 primarily, some CH4 for coal)
- **Quality:** High (official energy statistics)
- **Redistribution:** ✅ Allowed (with attribution)
- **Formats:** Excel, CSV

**Key Datasets:**
- Electricity emission factors by country (kg CO2/kWh)
- Fuel combustion defaults by country
- Grid emission factors (location-based)

**Action Items:**
- [ ] Download free tier (country aggregates)
- [ ] Evaluate $500/year tier for sub-national grid factors
- [ ] Integrate with EPA/IPCC for multi-gas (IEA CO2 + IPCC CH4/N2O ratios)
- [ ] Document IEA methodology (energy balance approach)

---

#### 3.4 UK BEIS / DESNZ GHG Conversion Factors (UK)
- **URL:** https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting
- **Coverage:** UK fuels, electricity, transport, waste
- **License:** Open Government License v3.0
- **Cost:** FREE
- **Update Frequency:** Annual (June)
- **Multi-Gas:** ✅ Yes (CO2, CH4, N2O breakdown)
- **Quality:** Very High (UK regulatory standard)
- **Redistribution:** ✅ Allowed
- **Formats:** Excel (comprehensive workbook)

**Key Datasets:**
- Fuels (including WTT upstream)
- UK Electricity (by region and time period)
- Transport (by mode and fuel)

**Action Items:**
- [ ] Download 2024 factors (June release)
- [ ] Parse WTT factors for Scope 3 support
- [ ] Use as UK-specific override for IPCC defaults
- [ ] Document UK-specific methodologies (Defra standards)

---

### Tier 2: Paid, High-Quality Sources (P1)

#### 3.5 Ecoinvent Database (Global LCA)
- **URL:** https://ecoinvent.org/
- **Coverage:** 18,000+ processes, full lifecycle
- **License:** Proprietary (strict terms)
- **Cost:** €3,500 - €10,000/year (organization size-based)
- **Update Frequency:** 3-year cycle (v3.10 current)
- **Multi-Gas:** ✅ Full lifecycle (all GHGs)
- **Quality:** Very High (peer-reviewed)
- **Redistribution:** ❌ NOT allowed (usage only)
- **Formats:** XML, JSON, CSV exports

**Key Datasets:**
- Electricity mixes (by country, technology, time)
- Fuel production chains (WTT, upstream emissions)
- Refined factors (not just combustion)

**Limitations:**
- Cannot redistribute factors to customers
- Must be used internally only
- Expensive for startups

**Decision:** **DEFERRED to Phase 4** (once revenue justifies cost)

---

#### 3.6 GaBi Databases (Global LCA)
- **URL:** https://gabi.sphera.com/
- **Coverage:** Similar to Ecoinvent (industry-focused)
- **License:** Proprietary
- **Cost:** €8,000 - €15,000/year
- **Multi-Gas:** ✅ Full lifecycle
- **Quality:** Very High

**Decision:** **DEFERRED to Phase 4** (Ecoinvent preferred if choosing one)

---

#### 3.7 GREET Model (US DOE Argonne)
- **URL:** https://greet.es.anl.gov/
- **Coverage:** Transportation fuels, electricity, hydrogen
- **License:** FREE (US DOE)
- **Cost:** FREE
- **Update Frequency:** Annual
- **Multi-Gas:** ✅ Yes (full lifecycle)
- **Quality:** Very High (Argonne National Lab)
- **Redistribution:** ✅ Allowed
- **Formats:** Excel model (complex), published defaults

**Key Datasets:**
- Vehicle fuel pathways (gasoline, diesel, electric, H2)
- Upstream (WTT) emissions by pathway
- Electricity generation by technology

**Action Items:**
- [ ] Extract GREET defaults for US transport fuels
- [ ] Use for WTT factors (free alternative to Ecoinvent)
- [ ] Document pathways (conventional, renewable, etc.)

---

### Tier 3: Regional/Specialized Sources (P1)

#### 3.8 European Environment Agency (EEA)
- **URL:** https://www.eea.europa.eu/data-and-maps
- **Coverage:** EU27 countries, sectoral
- **License:** Open (with attribution)
- **Cost:** FREE
- **Formats:** Excel, CSV

**Action Items:**
- [ ] Extract EU electricity grid factors
- [ ] Supplement IEA with EEA for EU-specific
- [ ] Use for residual mix (market-based Scope 2)

---

#### 3.9 India - Bureau of Energy Efficiency (BEE)
- **URL:** https://beeindia.gov.in/
- **Coverage:** India electricity, fuels
- **License:** Government Open Data
- **Cost:** FREE
- **Formats:** PDF reports (requires extraction)

**Action Items:**
- [ ] Extract India grid emission factors (state-wise if available)
- [ ] Supplement with IEA/IPCC for fuels
- [ ] Document methodology (CEA grid database)

---

#### 3.10 Australia - National Greenhouse Accounts (NGA) Factors
- **URL:** https://www.industry.gov.au/data-and-publications/national-greenhouse-accounts-factors
- **Coverage:** Australia fuels, electricity (state-level)
- **License:** CC BY 4.0
- **Cost:** FREE
- **Multi-Gas:** ✅ Yes

**Action Items:**
- [ ] Download annual factors
- [ ] Use for Australia-specific calculations
- [ ] Document NGA methodology

---

## 4. Data Precedence & Quality Rules

### Source Precedence (Highest to Lowest)
When multiple sources provide factors for the same fuel:

1. **Country-specific regulatory factors** (EPA for US, BEIS for UK, etc.)
2. **Regional regulatory factors** (EEA for EU)
3. **IEA country-specific** (if available)
4. **IPCC country-specific** (if in database)
5. **IPCC Tier 1 defaults** (global fallback)

**Tie-breaking:**
- Most recent publication year
- Higher data quality score (DQS)
- More transparent methodology
- Official government > NGO > academic

---

### Quality Thresholds

| Criterion | Accept | Review | Reject |
|-----------|--------|--------|--------|
| **Source Year** | ≤ 3 years old | 3-5 years | > 5 years |
| **Uncertainty** | ≤ 10% | 10-20% | > 20% |
| **Methodology** | IPCC Tier 1+ | Documented | Undocumented |
| **Peer Review** | Yes | Industry std | No |
| **Geographic Match** | Country/state | Regional | Global only |

---

## 5. Licensing & Redistribution Matrix

| Source | Use Internally | Redistribute to Customers | Attribution Required | Commercial Use |
|--------|----------------|---------------------------|----------------------|----------------|
| EPA | ✅ Yes | ✅ Yes | ⚠️ Recommended | ✅ Yes |
| IPCC | ✅ Yes | ✅ Yes | ✅ Required | ✅ Yes |
| IEA (free) | ✅ Yes | ✅ Yes | ✅ Required | ✅ Yes |
| UK BEIS | ✅ Yes | ✅ Yes | ✅ Required | ✅ Yes |
| GREET | ✅ Yes | ✅ Yes | ✅ Required | ✅ Yes |
| Ecoinvent | ✅ Yes | ❌ NO | ✅ Required | ⚠️ License-dependent |
| GaBi | ✅ Yes | ❌ NO | ✅ Required | ⚠️ License-dependent |

**Legal Review:** Consult legal counsel before:
- Redistributing any paid database factors
- Using factors in customer-facing reports
- Publishing factors in documentation/examples

---

## 6. Implementation Phases

### Phase 1: Foundation (Weeks 1-3)
**Goal:** Establish baseline with free, high-quality sources

**Data Deliverables:**
- [ ] EPA factors (US) - ~200 factors
- [ ] IPCC Tier 1 defaults (Global) - ~150 factors
- [ ] IEA free tier (25 countries electricity) - ~25 factors
- [ ] UK BEIS factors - ~100 factors
- [ ] Factor database schema implemented
- [ ] Ingestion scripts (Excel → Python dict)
- [ ] Validation tests (schema compliance)

**Coverage Target:** US, UK, EU (aggregated), 30 fuel types, Scope 1+2
**Total Factors:** ~500

---

### Phase 2: Multi-Gas Enrichment (Weeks 4-6)
**Goal:** Add CH4/N2O vectors from IPCC ratios

**Data Deliverables:**
- [ ] IPCC CH4/N2O ratios by fuel type
- [ ] Merge EPA CO2 + IPCC CH4/N2O
- [ ] Calculate CO2e using AR6 GWP (100yr)
- [ ] Calculate CO2e using AR6 GWP (20yr)
- [ ] Uncertainty ranges from IPCC Guidelines
- [ ] DQS scoring per factor

**Coverage Target:** All Phase 1 factors enriched with multi-gas
**Total Factors:** ~500 (enhanced)

---

### Phase 3: Regional Expansion (Weeks 7-9)
**Goal:** Add key markets (India, Australia, Japan, Canada)

**Data Deliverables:**
- [ ] India BEE electricity factors
- [ ] Australia NGA factors
- [ ] Japan electricity mix (IEA + METI)
- [ ] Canada NIR factors
- [ ] Regional precedence rules implemented
- [ ] Regional testing (compare vs. national calculators)

**Coverage Target:** 10 additional countries
**Total Factors:** ~750

---

### Phase 4: Advanced (Weeks 10-12)
**Goal:** WTT upstream, hourly electricity, specialized fuels

**Data Deliverables:**
- [ ] GREET WTT factors (US transport)
- [ ] UK BEIS WTT factors (all fuels)
- [ ] Hydrogen pathways (GREET, IEA)
- [ ] Biofuels and blends (IPCC + GREET)
- [ ] Evaluate Ecoinvent (if budget approved)
- [ ] Hourly electricity (if customer demand)

**Coverage Target:** Scope 3 Category 3 (upstream fuels)
**Total Factors:** ~1,000

---

## 7. Data Governance

### Update Frequency
- **Quarterly Review:** Check for EPA, IEA, BEIS updates
- **Annual Refresh:** Full factor database update (Q4)
- **Ad-Hoc:** Customer-reported discrepancies or corrections

### Approval Process
1. **Data Analyst** ingests and validates new factors
2. **Technical Lead** reviews schema compliance and DQS
3. **Compliance Officer** verifies licensing and redistribution rights
4. **CTO** approves factors that change reported emissions > 5%
5. **Version Control** commit with changelog and citation

### Change Management
- **Patch updates** (v1.1): Add new fuels, same methodology
- **Minor updates** (v1.2): Update factors within ±5%
- **Major updates** (v2.0): Methodology changes, require customer notice

---

## 8. Cost Estimate

### Year 1 (Setup)
| Item | Cost | Frequency | Annual Total |
|------|------|-----------|--------------|
| EPA, IPCC, IEA (free tier) | $0 | - | $0 |
| UK BEIS | $0 | - | $0 |
| GREET | $0 | - | $0 |
| IEA detailed ($500) | $500 | One-time | $500 |
| Data analyst (20% FTE) | $25,000 | Annual | $25,000 |
| Legal review (licensing) | $2,000 | One-time | $2,000 |
| **Subtotal** | | | **$27,500** |

### Year 2+ (Maintenance)
| Item | Cost | Annual Total |
|------|------|--------------|
| Free sources (maintenance) | $0 | $0 |
| IEA subscription | $500 | $500 |
| Data analyst (10% FTE) | $12,500 | $12,500 |
| Ecoinvent (if approved) | $5,000 | $5,000 |
| **Subtotal** | | **$18,000** |

**Note:** Ecoinvent deferred until revenue > $500K/year or specific customer request.

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Source changes license terms | Medium | High | Monitor annually, maintain IPCC fallbacks |
| EPA/IEA delays annual update | Low | Medium | Use previous year with flag |
| Customer requires Ecoinvent | Medium | High | Budget approval process, pass-through cost |
| Data quality insufficient for audit | Low | Critical | Prioritize official govt sources, DQS scoring |
| CH4/N2O ratios inaccurate | Medium | Medium | Validate vs. EPA, use uncertainty bounds |
| Multi-source conflicts | High | Medium | Clear precedence rules, document rationale |

---

## 10. Success Metrics

### Coverage
- ✅ **P0:** 500+ factors covering US, EU, UK, 30 fuel types (Week 6)
- ✅ **P1:** 750+ factors covering 10+ countries (Week 9)
- ✅ **P2:** 1,000+ factors with WTT upstream (Week 12)

### Quality
- ✅ **P0:** 100% of factors have source attribution and license
- ✅ **P0:** 95% of factors have uncertainty ≤ 15%
- ✅ **P1:** 80% of factors have DQS ≥ 4.0
- ✅ **P1:** 100% compliance tests pass vs EPA/GHGP calculators

### Compliance
- ✅ **P0:** All redistributed factors have permissive licenses
- ✅ **P0:** Zero legal violations (unauthorized redistribution)
- ✅ **P1:** CSRD Annex II compliant (multi-gas breakdown)
- ✅ **P1:** GHG Protocol Corporate Standard compliant

---

## 11. Action Items & Owners

### Immediate (Week 1)
- [ ] **CTO:** Approve budget ($27,500 Year 1)
- [ ] **Legal:** Review licensing matrix, flag any concerns
- [ ] **Data Lead:** Download EPA 2024 factors
- [ ] **Data Lead:** Download IPCC EFDB extracts
- [ ] **Data Lead:** Download UK BEIS 2024 factors
- [ ] **Eng Lead:** Implement EmissionFactorRecord schema
- [ ] **Eng Lead:** Create ingestion pipeline (Excel → dict)

### Week 2-3
- [ ] **Data Lead:** Parse EPA factors into schema
- [ ] **Data Lead:** Parse IPCC defaults into schema
- [ ] **Data Lead:** Parse UK BEIS into schema
- [ ] **Data Lead:** Download IEA free tier
- [ ] **QA:** Validate schema compliance (100% pass)
- [ ] **QA:** Spot-check factors vs. source PDFs
- [ ] **Eng:** Implement factor precedence logic

### Week 4-6
- [ ] **Data Lead:** Enrich factors with IPCC CH4/N2O ratios
- [ ] **Data Lead:** Calculate AR6 GWP values (100yr, 20yr)
- [ ] **Data Lead:** Assign DQS scores (5 dimensions)
- [ ] **Data Lead:** Add uncertainty ranges
- [ ] **QA:** Compliance tests vs EPA calculator
- [ ] **QA:** Compliance tests vs GHGP tools
- [ ] **Eng:** Integrate multi-gas into API v2

### Week 7-12
- [ ] **Data Lead:** Acquire regional factors (IN, AU, JP, CA)
- [ ] **Data Lead:** Download GREET WTT defaults
- [ ] **Data Lead:** Extract hydrogen pathways
- [ ] **QA:** Regional validation tests
- [ ] **Product:** Evaluate Ecoinvent (customer demand)
- [ ] **Eng:** Implement WTT boundary support

---

## 12. Conclusion

This Data Acquisition Plan provides a **phased, cost-effective approach** to building a world-class emission factor database for FuelAgentAI v2:

- **Phase 1-2 (Weeks 1-6):** Free sources (EPA, IPCC, BEIS, IEA) cover 80% of needs
- **Phase 3 (Weeks 7-9):** Regional expansion adds key markets
- **Phase 4 (Weeks 10-12):** Advanced features (WTT, specialized)
- **Cost:** $27,500 Year 1, $18,000/year ongoing (without Ecoinvent)
- **Coverage:** 500 → 1,000+ factors over 12 weeks
- **Quality:** 100% sourced, licensed, versioned, multi-gas capable
- **Compliance:** CSRD, CDP, GHG Protocol ready

**Next Steps:**
1. Secure budget approval
2. Begin EPA/IPCC/BEIS downloads (Week 1)
3. Implement schema (parallel with downloads)
4. Validate quality (Week 3)
5. Ship Phase 1 (Week 6)

---

**Document Owner:** Data Lead
**Approvers:** CTO, Legal, Compliance Officer
**Next Review:** 2025-11-01 (after Phase 1 completion)
