# Research: AGENT-MRV-013 -- Scope 2 Dual Reporting Reconciliation Agent

## Executive Summary

This document provides the comprehensive technical research foundation for
AGENT-MRV-013 (GL-MRV-X-024), the Dual Reporting Reconciliation Agent. This
agent is responsible for reconciling, cross-validating, and quality-scoring
the outputs of MRV-009 (Location-Based), MRV-010 (Market-Based), MRV-011
(Steam/Heat Purchase), and MRV-012 (Cooling Purchase) to produce compliant
dual-reporting tables required by GHG Protocol, CSRD, CDP, SBTi, and GRI.

**Key finding**: The GHG Protocol Scope 2 Guidance (2015), Chapter 7, mandates
dual reporting of both location-based and market-based Scope 2 emissions for
all organizations operating in markets with contractual instruments. Despite
this mandate, research shows that as of 2019 only 23% of reporting companies
published both figures, and 32% did not even label which method they used. A
dedicated reconciliation agent is essential to enforce completeness, identify
discrepancies, quality-score data across methods, generate multi-framework
reporting tables, and track year-over-year trends. The proposed 2025-2027
Scope 2 revisions add hourly matching and deliverability requirements that
make automated reconciliation even more critical.

---

## Table of Contents

1. [GHG Protocol Scope 2 Guidance -- Dual Reporting Requirements](#1-ghg-protocol-scope-2-guidance----dual-reporting-requirements)
2. [Regulatory Dual Reporting Requirements](#2-regulatory-dual-reporting-requirements)
3. [Common Discrepancy Types Between Location and Market Methods](#3-common-discrepancy-types-between-location-and-market-methods)
4. [Quality Criteria for Dual Reporting](#4-quality-criteria-for-dual-reporting)
5. [Reconciliation Methodology](#5-reconciliation-methodology)
6. [Emission Factor Hierarchy for Market-Based Method](#6-emission-factor-hierarchy-for-market-based-method)
7. [RE100 and Green Tariff Treatment in Dual Reporting](#7-re100-and-green-tariff-treatment-in-dual-reporting)
8. [Key Formulas for Reconciliation Metrics](#8-key-formulas-for-reconciliation-metrics)
9. [Suggested 7-Engine Architecture](#9-suggested-7-engine-architecture)
10. [Suggested Enums, Models, and Constants](#10-suggested-enums-models-and-constants)

---

## 1. GHG Protocol Scope 2 Guidance -- Dual Reporting Requirements

### 1.1 Chapter 7 Overview

The GHG Protocol Scope 2 Guidance (2015) Chapter 7 ("Reporting") establishes
that organizations must report Scope 2 GHG emissions using two complementary
methods -- location-based and market-based -- whenever they operate in markets
that provide contractual instruments (RECs, GOs, supplier-specific factors,
residual mix data). This "dual reporting" requirement is the cornerstone of
Scope 2 transparency.

**Mandatory dual reporting applies when:**
- The organization operates in a market with product or supplier-specific data
- Contractual instruments (RECs, GOs, PPAs, I-RECs, etc.) are available
- The organization procures any form of renewable or specified energy

**Location-only reporting is acceptable when:**
- The organization operates exclusively in markets without contractual instruments
- No product or supplier-specific emission factor data exists in the market

A decision tree in Section 6.1 of the Guidance determines whether an
organization shall dual-report. In practice, virtually all organizations in
North America, Europe, Australia, Japan, and large parts of Asia and Latin
America must dual-report.

### 1.2 What Chapter 7 Requires

Per GHG Protocol Scope 2 Guidance Chapter 7, reporters must disclose:

1. **Location-based Scope 2 result** (tCO2e) for the reporting year
2. **Market-based Scope 2 result** (tCO2e) for the reporting year
3. **Activity data**: Annual purchased electricity (kWh/MWh) by facility
4. **Emission factors used**: Grid-average EFs and contractual instrument EFs
5. **Contractual instrument categories**: Types of instruments used
   (REC, GO, PPA, supplier-specific, residual mix)
6. **Coverage percentage**: Percentage of total consumption covered by
   contractual instruments vs. residual mix/grid-average fallback
7. **Quality criteria compliance**: Assertion that instruments meet
   the 8 Scope 2 Quality Criteria
8. **Gases included**: CO2, CH4, N2O (and whether CH4/N2O were available)
9. **GWP source**: AR4, AR5, or AR6 values used
10. **Base year**: Identification of which method the base year uses
11. **Biogenic CO2**: Separate disclosure if applicable
12. **Exclusions**: Any exclusions with justification

### 1.3 Dual Reporting Format (Chapter 7 Table)

The Guidance provides an illustrative format:

```
+----------------------------------+------------------+-------------------+
| Scope 2 Emissions                | Location-Based   | Market-Based      |
+----------------------------------+------------------+-------------------+
| Purchased Electricity            | XXX tCO2e        | YYY tCO2e         |
| Purchased Steam                  | XXX tCO2e        | YYY tCO2e         |
| Purchased Heating                | XXX tCO2e        | YYY tCO2e         |
| Purchased Cooling                | XXX tCO2e        | YYY tCO2e         |
+----------------------------------+------------------+-------------------+
| Total Scope 2                    | XXX tCO2e        | YYY tCO2e         |
+----------------------------------+------------------+-------------------+
| % Covered by instruments         | N/A              | ZZ%               |
| % Using residual mix             | N/A              | ZZ%               |
| % Using grid-average fallback    | N/A              | ZZ%               |
+----------------------------------+------------------+-------------------+
```

**Critical**: The two method results are NEVER summed. They represent two
different views of the same physical energy consumption. The difference
between them reflects the impact of an organization's energy procurement
decisions.

### 1.4 Base Year Treatment

- Organizations must choose whether their base year inventory uses
  location-based, market-based, or both methods
- SBTi requires the base year to use the same method as the target
- Recalculation triggers apply independently to each method
- Structural changes (acquisitions, divestitures) require recalculation
  of BOTH methods to maintain comparability

### 1.5 Proposed Scope 2 Revisions (2025-2027)

The GHG Protocol opened public consultation in October 2025 on major Scope 2
revisions. The final standard is expected in 2027 with phased implementation.
Key proposed changes relevant to dual reporting reconciliation:

**Location-Based Method Changes:**
- New emission factor hierarchy prioritizing spatial then temporal granularity
- Consumption-based factors preferred over production-based
- "Accessible" factors: publicly available, free, from credible sources
- Load profiles allowed to translate annual data into hourly estimates

**Market-Based Method Changes:**
- Hourly matching required for all contractual instruments (with exemptions)
- Deliverability requirement: EACs must be from electrically connected grids
- New Standard Supply Service (SSS) rules limiting claims to pro-rata shares
- Residual mix: fossil-only rates required when residual mixes unavailable

**Dual Reporting Continuity:**
- Dual reporting requirement remains mandatory under the revised framework
- Both location-based and market-based methods continue to be required
- No changes to the fundamental two-method disclosure structure

**Exemption Thresholds:**
- Smaller organizations may be exempt from hourly matching
- Based on electricity consumption volume and/or company size
- Most CDP reporters (by count) would qualify for exemptions
- But the majority of grid load (by volume) remains subject

**Implementation Timeline:**
- Public consultation: October-December 2025
- Second consultation: 2026
- Final standard publication: Late 2027
- Phased effective dates: Multi-year transition period

---

## 2. Regulatory Dual Reporting Requirements

### 2.1 CSRD / ESRS E1 (Climate Change)

The EU Corporate Sustainability Reporting Directive (CSRD) through ESRS E1
mandates the most comprehensive Scope 2 dual reporting requirements globally.

**ESRS E1-6: Gross Scopes 1, 2, 3 and Total GHG Emissions**

| Datapoint | Requirement |
|-----------|------------|
| E1-6 para 49(a) | Gross location-based Scope 2 GHG emissions (tCO2e) |
| E1-6 para 49(b) | Gross market-based Scope 2 GHG emissions (tCO2e) |
| E1-6 para 52 | Total GHG emissions disaggregated by location-based and market-based |
| E1-6_18 | Percentage of contractual instruments in Scope 2 |
| E1-6_19 | Types of contractual instruments used |
| E1-6_20 | Percentage of market-based Scope 2 linked to purchased electricity bundled with instruments |
| E1-6_21 | Percentage bundled with generation attributes |
| E1-6_22 | Percentage of unbundled energy attribute claims |
| E1-6_24 | Biogenic CO2 emissions reported separately |
| AR 45(d) | Apply location-based AND market-based methods with contractual instrument documentation |
| AR 47 | Formulas for both location-based and market-based calculations |

**Two Total GHG Figures:**
ESRS E1 para 52 requires TWO total GHG emissions figures:
- Total derived from location-based Scope 2
- Total derived from market-based Scope 2

This means the reconciliation agent must produce two complete views of
total enterprise GHG emissions (Scope 1 + Scope 2_loc + Scope 3) and
(Scope 1 + Scope 2_mkt + Scope 3).

**Amended ESRS (2026):**
The amended ESRS effective for 2026 reporting maintains all dual reporting
requirements while introducing some simplifications for smaller entities.
GHG Protocol mandatory hourly matching requirements (formalized October 2025)
may affect how organizations approach their CSRD Scope 2 disclosures.

### 2.2 CDP Climate Change Questionnaire

CDP is the dominant voluntary disclosure platform requiring dual reporting.

**C6.2 -- Scope 2 Reporting Decision:**
- "Are you reporting a Scope 2, location-based figure?"
- "Are you reporting a Scope 2, market-based figure?"
- This determines which route C6.3 follows

**C6.3 -- Scope 2 Emissions Data (Three Routes):**

| Route | Condition | Max Score |
|-------|-----------|-----------|
| Route A | Both location + market reported | 4/4 points |
| Route B | Location-based only | 2/2 points |
| Route C | Market-based only | 1/2 points |

Route A (both methods) is required for maximum CDP score. Organizations
must provide gross global Scope 2 data for current and historical years.

**C6.4 -- Market-Based Breakdown:**
Appears only when market-based is reported AND C8.2 confirms consumption
of purchased electricity/heat/steam/cooling. Requires breakdown by:
- Instrument type (bundled/unbundled RECs, GOs, PPAs, supplier-specific)
- Quantity covered by each instrument type
- Residual/grid-average used for uncovered portion

**C8.2 -- Energy Consumption:**
- Purchased electricity (MWh)
- Purchased heat/steam/cooling (MWh or GJ)
- Self-generated electricity/heat
- Total energy consumption

**C8.2d -- Market-Based Emission Factor Details:**
- For each low-carbon energy purchase: technology, instrument, country,
  MWh, emission factor, tracking standard (e.g., Green-e, AIB/EECS)

### 2.3 SBTi (Science Based Targets initiative)

SBTi has specific requirements for how Scope 2 dual reporting interacts
with target-setting and progress tracking.

**SBTi Corporate Near-Term Criteria v5.3 (September 2025):**

| Requirement | Detail |
|-------------|--------|
| Target scope | Must cover company-wide Scope 1 + Scope 2 |
| Target method | Companies must disclose whether target uses location or market-based |
| Base year | Must use same method as target (location or market) |
| Ambition | 1.5C aligned (at minimum for near-term) |
| RE100 pathway | 80% renewable by 2025, 100% by 2030 for RE100 targets |
| Dual targets | Under SBTi Net-Zero Standard V2 (draft): BOTH location-based AND procurement-based targets required |
| Reporting | Both methods must be reported even if target uses only one |

**SBTi Net-Zero Standard V2 (Draft, 2025):**
The draft standard requires companies to set:
1. A location-based Scope 2 reduction target
2. A procurement-based target (either market-based or zero-carbon electricity)

This means the reconciliation agent must track progress against both
target types simultaneously.

**SBTi Renewable Electricity Targets:**
- Alternative to traditional market-based Scope 2 reduction
- RE100 aligned: 80% by 2025, 100% by 2030
- Coverage = RE procurement / total electricity consumption
- Requires separate tracking from emission reduction targets

### 2.4 GRI 305-2 / 305-3

The Global Reporting Initiative (GRI) 305 standard specifies Scope 2
disclosure requirements that align with GHG Protocol dual reporting.

**GRI 305-2: Energy Indirect (Scope 2) GHG Emissions**

| Disclosure | Requirement |
|-----------|-------------|
| 305-2-a | Gross location-based Scope 2 emissions (tCO2e) |
| 305-2-b | Gross market-based Scope 2 emissions (if operating in markets with instruments) |
| 305-2-c | Gases included (CO2, CH4, N2O) |
| 305-2-d | Base year for calculation |
| 305-2-e | Source of emission factors and GWP values |
| 305-2-f | Consolidation approach (equity share, financial control, operational control) |
| 305-2-g | Standards, methodologies, assumptions, calculation tools used |

**When Dual Reporting Is Required Under GRI:**
- Location-based: ALWAYS required
- Market-based: Required if the organization operates in markets providing
  product or supplier-specific data in the form of contractual instruments
- In practice: Organizations in US, EU, UK, Australia, Japan, etc. must
  always report both

**GRI 305-4: GHG Emissions Intensity**
- Emissions intensity ratio for the organization
- Can be calculated using either location-based or market-based
- The reconciliation agent should produce intensity ratios for both methods

### 2.5 ISO 14064-1:2018

ISO 14064-1 provides the international standard for quantification and
reporting of GHG emissions.

| Requirement | Detail |
|------------|--------|
| Category 2 | Indirect emissions from imported energy |
| Clause 5.2.3 | Quantify indirect energy emissions |
| Clause 7 | Reporting principles (relevance, completeness, consistency, transparency, accuracy) |
| Annex C | Guidance on allocation of emissions from purchased energy |

ISO 14064 does not mandate dual reporting per se, but the GHG Protocol
Scope 2 Guidance (which builds on ISO 14064) does. Most ISO 14064 reporters
voluntarily dual-report for GHG Protocol alignment.

### 2.6 Summary: Framework Dual Reporting Matrix

| Framework | Location-Based | Market-Based | Dual Mandatory | Reconciliation |
|-----------|---------------|--------------|----------------|----------------|
| GHG Protocol | Required | Required (if instruments exist) | Yes (Chapter 7) | Side-by-side table |
| CSRD/ESRS E1 | Required (E1-6 49a) | Required (E1-6 49b) | Yes | Two total GHG figures |
| CDP C6.3 | Required (Route A) | Required (Route A, max score) | Yes (for max score) | C6.4 breakdown |
| SBTi v5.3 | Target method choice | Target method choice | Both reported | Dual targets (V2 draft) |
| GRI 305-2 | Always required | Required if instruments exist | Yes (in instrument markets) | Separate disclosure |
| ISO 14064 | Category 2 | Optional | No (but recommended) | N/A |

---

## 3. Common Discrepancy Types Between Location and Market Methods

### 3.1 Fundamental Cause of Discrepancies

The location-based and market-based methods measure fundamentally different
things:

- **Location-based**: Physical grid reality -- the average emission intensity
  of the electricity grid serving the organization's facilities
- **Market-based**: Contractual arrangements -- the emission attributes
  associated with the energy the organization has contractually chosen

Because these are different measurement lenses on the same physical energy
consumption, discrepancies are not errors -- they are expected and
informative. The magnitude and direction of discrepancies reveal the
impact of an organization's energy procurement strategy.

### 3.2 Taxonomy of Discrepancy Types

**Type 1: Renewable Energy Certificate (REC/GO) Impact**
- **Cause**: Organization purchases RECs, GOs, or I-RECs for all or part
  of its electricity consumption
- **Direction**: Market < Location (always, when renewable certificates used)
- **Magnitude**: Can be 100% reduction if fully covered by renewables
- **Example**: 10,000 MWh consumption in Germany (grid EF = 0.338 tCO2e/MWh)
  - Location-based: 3,380 tCO2e
  - Market-based (100% wind GOs): 0 tCO2e
  - Discrepancy: -3,380 tCO2e (100% reduction)

**Type 2: Residual Mix Higher Than Grid Average**
- **Cause**: When renewable certificates are removed from the grid mix,
  the remaining "residual mix" has a higher EF than the grid average
- **Direction**: Market > Location (for uncovered consumption)
- **Magnitude**: Typically 10-63% higher than grid average
- **Example**: 5,000 MWh uncovered in France
  - Location-based (grid avg): 0.052 kgCO2e/kWh = 260 tCO2e
  - Market-based (residual mix): 0.085 kgCO2e/kWh = 425 tCO2e
  - Discrepancy: +165 tCO2e (+63% higher)

**Type 3: Supplier-Specific Factor vs. Grid Average**
- **Cause**: Organization's utility has a specific fuel mix different from
  grid average (e.g., nuclear-heavy utility on a coal-heavy grid)
- **Direction**: Can be either direction
- **Magnitude**: Varies widely by utility
- **Example**: Nuclear-heavy utility in coal region
  - Location-based (grid avg): 0.700 tCO2e/MWh = 7,000 tCO2e
  - Market-based (supplier): 0.100 tCO2e/MWh = 1,000 tCO2e
  - Discrepancy: -6,000 tCO2e (-86% reduction)

**Type 4: Geographic Mismatch (Cross-Border)**
- **Cause**: Organization purchases certificates from a different grid
  region than where consumption occurs
- **Direction**: Varies
- **Magnitude**: Can be significant if grid mixes differ substantially
- **Example**: Coal-heavy grid consumption with Nordic hydro GOs
  - Should be flagged as lower quality under Scope 2 Quality Criteria
  - May not satisfy deliverability requirements under proposed revisions

**Type 5: Temporal Mismatch**
- **Cause**: Certificate vintage year does not match consumption year;
  or annual certificates applied to hourly consumption
- **Direction**: Varies
- **Magnitude**: Small under current rules; significant under proposed
  hourly matching requirements
- **Example**: 2023 vintage RECs applied to 2024 consumption
  - Acceptable under current guidance (within vintage window)
  - May be rejected under proposed hourly matching rules

**Type 6: Partial Coverage**
- **Cause**: Organization has instruments for only a portion of consumption;
  the remainder uses residual mix or grid-average fallback
- **Direction**: Net market-based can be above or below location-based
- **Magnitude**: Depends on coverage percentage and factor differences
- **Example**: 60% covered by RECs, 40% by residual mix
  - Location: 100% x grid_avg EF
  - Market: (60% x 0) + (40% x residual_mix_EF)
  - Net effect depends on relative magnitudes

**Type 7: Steam/Heat/Cooling Method Differences**
- **Cause**: Steam, heat, and cooling may use different EF sources under
  location vs. market methods (district heating supplier factor vs. default)
- **Direction**: Varies
- **Magnitude**: Typically smaller than electricity differences
- **Example**: District heating with supplier-specific EF vs. default
  - Location-based: Default DH EF for region
  - Market-based: Supplier-specific DH factor (if contract specifies)

**Type 8: Grid EF Update Timing**
- **Cause**: Grid emission factors are published with 1-2 year lag;
  location-based may use older EFs while market-based uses current contracts
- **Direction**: Varies by grid trajectory (improving or worsening)
- **Magnitude**: Typically 2-10% year-to-year variation

### 3.3 Discrepancy Materiality Thresholds

Based on industry practice and regulatory expectations:

| Threshold | Description | Action Required |
|-----------|-------------|-----------------|
| < 5% | Immaterial | Acceptable, no investigation needed |
| 5-15% | Minor | Explanation recommended in reporting narrative |
| 15-50% | Material | Full explanation required; flag for assurance review |
| 50-100% | Significant | Expected for organizations with major RE procurement; detailed disclosure |
| > 100% | Extreme | Requires investigation for data quality issues or errors |

Note: For organizations with high renewable procurement, a large
market-below-location discrepancy is expected and healthy. The thresholds
above are for UNEXPECTED discrepancies or market-above-location results.

### 3.4 Discrepancy Direction Indicators

| Direction | Code | Meaning |
|-----------|------|---------|
| Market << Location | REC_DOMINANT | Strong renewable procurement impact |
| Market < Location | MIXED_LOWER | Partial renewable + lower-carbon supplier |
| Market ~ Location | ALIGNED | No or minimal contractual impact |
| Market > Location | RESIDUAL_EFFECT | Residual mix higher than grid average |
| Market >> Location | QUALITY_CONCERN | Possible data quality issue; investigate |

---

## 4. Quality Criteria for Dual Reporting

### 4.1 GHG Protocol Scope 2 Quality Criteria (8 Criteria)

These criteria apply specifically to contractual instruments used in the
market-based method. Instruments that fail these criteria should be
downgraded to residual mix treatment.

| # | Criterion | What It Means | Validation Check |
|---|-----------|--------------|-----------------|
| 1 | Convey GHG emission rate | Instrument must carry the GHG emission rate attribute of the underlying generation | EF is explicitly stated or derivable |
| 2 | Unique claim | Only one entity may claim the attribute -- no double counting | Certificate serial number retired in registry |
| 3 | Tracked and retired | Must be redeemed, retired, or cancelled in a recognized system | Registry retirement confirmation |
| 4 | Temporal matching | Issued and redeemed close to the consumption period | Vintage year within reporting period +/- 1 year |
| 5 | Geographic matching | From the same market/interconnected grid as consumption | Same grid region or bidding zone |
| 6 | Supplier-specific calculation | Based on delivered electricity with certificates | Utility fuel mix disclosure documentation |
| 7 | Direct contractual purchase | Verified purchasing rights | Contract documentation |
| 8 | Residual mix availability | Adjusted residual mix characterizing unclaimed energy | Residual mix source documented or absence disclosed |

### 4.2 Dual Reporting Quality Dimensions

Beyond the instrument quality criteria, the reconciliation agent must
assess quality across four dimensions aligned with GHG Protocol principles.

**Dimension 1: Completeness**
- All energy types covered in BOTH methods (electricity, steam, heat, cooling)
- All facilities included in BOTH calculations
- All months/quarters of the reporting period covered
- No energy consumption "orphaned" (present in one method but not the other)
- Completeness score = (covered_items / total_items) for each method

Completeness check matrix:

| Energy Type | Facilities | Location-Based | Market-Based | Gap? |
|-------------|-----------|---------------|-------------|------|
| Electricity | Facility A | Yes | Yes | No |
| Electricity | Facility B | Yes | Missing | YES |
| Steam | Facility C | Yes | Yes | No |
| Cooling | Facility D | Missing | Missing | YES |

**Dimension 2: Consistency**
- Same activity data (MWh, GJ) used for both methods
- Same reporting boundary and consolidation approach
- Same GWP values (AR4/AR5/AR6) for both methods
- Same gas coverage (CO2, CH4, N2O) for both methods
- Same reporting period and base year
- Consistent unit conversions

Consistency checks:
```
Activity_Data_Location == Activity_Data_Market  (must be equal)
GWP_Source_Location == GWP_Source_Market        (must match)
Gases_Location == Gases_Market                   (must match)
Boundary_Location == Boundary_Market             (must match)
Period_Location == Period_Market                  (must match)
```

**Dimension 3: Accuracy**
- Emission factors from authoritative sources
- Activity data from metered readings (not estimates) where possible
- Calculation methodology correctly applied
- Unit conversions verified
- Uncertainty ranges quantified

Accuracy scoring:

| Data Source | Accuracy Score |
|-----------|---------------|
| Calibrated meter data | 1.0 |
| Utility invoice data | 0.9 |
| Estimated from floor area/benchmarks | 0.6 |
| Extrapolated from partial year | 0.7 |
| Default EFs (IPCC/DEFRA) | 0.7 |
| Subregional EFs (eGRID) | 0.9 |
| Supplier-specific EFs (audited) | 0.95 |
| Supplier-specific EFs (unaudited) | 0.8 |

**Dimension 4: Transparency**
- All emission factor sources documented
- Instrument types and quantities disclosed
- Coverage percentage clearly stated
- Residual mix source identified
- Exclusions justified
- Methodology references cited

### 4.3 Composite Quality Score Formula

The reconciliation agent should compute a composite quality score for
the dual reporting package:

```
Quality_Score = (
    w_completeness * Completeness_Score +
    w_consistency  * Consistency_Score +
    w_accuracy     * Accuracy_Score +
    w_transparency * Transparency_Score
)

Default weights:
  w_completeness = 0.30
  w_consistency  = 0.25
  w_accuracy     = 0.25
  w_transparency = 0.20
```

Each sub-score ranges from 0.0 to 1.0. The composite score ranges from
0.0 to 1.0.

Quality grades:

| Score Range | Grade | Meaning |
|-------------|-------|---------|
| 0.90 - 1.00 | A | Excellent -- assurance-ready |
| 0.80 - 0.89 | B | Good -- minor improvements needed |
| 0.70 - 0.79 | C | Acceptable -- several issues to address |
| 0.60 - 0.69 | D | Below standard -- significant gaps |
| 0.00 - 0.59 | F | Failing -- not suitable for external reporting |

---

## 5. Reconciliation Methodology

### 5.1 Reconciliation Process Overview

The reconciliation process is an 8-step procedure that runs after
MRV-009, MRV-010, MRV-011, and MRV-012 have each completed their
calculations for a given reporting period.

```
Step 1: Collect -- Gather results from MRV-009, 010, 011, 012
Step 2: Align   -- Ensure same boundary, period, units, GWP
Step 3: Map     -- Map energy types across location/market
Step 4: Compare -- Calculate discrepancies at each level
Step 5: Score   -- Compute quality scores for each dimension
Step 6: Flag    -- Identify material discrepancies and gaps
Step 7: Format  -- Generate dual-reporting tables per framework
Step 8: Trend   -- Calculate YoY trends for both methods
```

### 5.2 Step 1: Collect Upstream Results

The reconciliation agent consumes output from four upstream agents:

| Upstream Agent | Provides | Key Fields |
|---------------|----------|------------|
| MRV-009 (Location-Based) | Grid-average emissions for electricity by facility/region | total_emissions, per_facility, per_gas, grid_ef_used, td_loss_applied |
| MRV-010 (Market-Based) | Contractual-instrument emissions for electricity | total_emissions, covered_emissions, uncovered_emissions, instruments_used, coverage_pct, residual_mix_ef |
| MRV-011 (Steam/Heat) | Steam and district heating emissions | steam_emissions, heat_emissions, location_ef, market_ef (if supplier-specific) |
| MRV-012 (Cooling) | District cooling and cooling purchase emissions | cooling_emissions, technology_used, cop_values, location_ef, market_ef |

Each upstream agent provides BOTH a location-based and a market-based
result where applicable:
- MRV-009: Location-based electricity (primary output)
- MRV-010: Market-based electricity (primary output) with location reference
- MRV-011: Provides both location-based and market-based steam/heat EFs
- MRV-012: Provides both location-based and market-based cooling EFs

### 5.3 Step 2: Alignment Checks

Before reconciliation, verify alignment:

| Check | Expected | Action if Misaligned |
|-------|----------|---------------------|
| Reporting period | Identical across all 4 agents | Reject; require recalculation |
| Organizational boundary | Identical (operational/financial/equity) | Flag; may indicate boundary error |
| Facility set | Identical list of facilities | Flag missing facilities |
| GWP source | Same (AR4/AR5/AR6) | Convert to common GWP |
| Gas coverage | Same (CO2/CH4/N2O) | Flag differences |
| Unit system | Same (tCO2e, MWh, GJ) | Convert to common units |
| Currency (if EF costs) | Same | Convert to common currency |

### 5.4 Step 3: Energy Type Mapping

Map all energy purchases to their location-based and market-based
treatment across the four agents:

```
Energy Purchase Register:
  For each facility:
    For each energy type (electricity, steam, heat, cooling):
      - Activity data (quantity, units)
      - Location-based source agent (MRV-009, 011, or 012)
      - Location-based EF and emissions result
      - Market-based source agent (MRV-010, 011, or 012)
      - Market-based EF and emissions result
      - Instruments applied (type, quantity, EF)
      - Coverage status (covered, uncovered, partial)
```

### 5.5 Step 4: Discrepancy Calculation

For each energy type at each facility level:

```
Absolute_Discrepancy = Market_Emissions - Location_Emissions
Relative_Discrepancy = (Market_Emissions - Location_Emissions) / Location_Emissions * 100
```

Aggregate at multiple levels:
1. **Facility level**: Per-facility, per-energy-type
2. **Energy type level**: Across all facilities for a given energy type
3. **Method level**: Total location vs. total market
4. **Organization level**: Grand total location vs. grand total market

### 5.6 Step 5: Quality Scoring

Apply the quality assessment framework from Section 4:

```python
# Completeness
completeness_loc = count(loc_results) / count(expected_energy_purchases)
completeness_mkt = count(mkt_results) / count(expected_energy_purchases)
completeness = min(completeness_loc, completeness_mkt)

# Consistency
consistency_checks = [
    activity_data_matches,
    gwp_source_matches,
    gases_match,
    boundary_matches,
    period_matches,
]
consistency = sum(passed) / len(consistency_checks)

# Accuracy (weighted by emissions magnitude)
accuracy = sum(
    emission_i * accuracy_score_i for each source_i
) / total_emissions

# Transparency
transparency_items = [
    ef_sources_documented,
    instrument_types_disclosed,
    coverage_pct_stated,
    residual_mix_source_identified,
    exclusions_justified,
    methodology_cited,
]
transparency = sum(documented) / len(transparency_items)
```

### 5.7 Step 6: Flagging Logic

Generate flags based on discrepancy analysis:

| Flag Type | Condition | Severity |
|-----------|-----------|----------|
| MISSING_LOCATION | Energy purchase has no location-based result | Critical |
| MISSING_MARKET | Energy purchase has no market-based result | Critical |
| ACTIVITY_DATA_MISMATCH | Activity data differs >1% between methods | High |
| GWP_MISMATCH | Different GWP sources used | High |
| LARGE_DISCREPANCY | Relative discrepancy >50% without RE instruments | High |
| MARKET_ABOVE_LOCATION | Market > Location without residual mix explanation | Medium |
| INSTRUMENT_QUALITY_FAIL | Instrument fails Scope 2 Quality Criteria | High |
| TEMPORAL_MISMATCH | Certificate vintage outside reporting period | Medium |
| GEOGRAPHIC_MISMATCH | Certificate from different grid/market | Medium |
| COVERAGE_GAP | Market-based coverage <80% without explanation | Medium |
| MISSING_RESIDUAL_MIX | No residual mix EF available for uncovered | High |
| FACILITY_MISSING | Facility in one method but not the other | Critical |
| BASE_YEAR_INCONSISTENT | Base year method differs from target method | Medium |
| BIOGENIC_MISSING | Biogenic CO2 not separately disclosed | Low |
| EF_OUTDATED | Emission factor older than 3 years | Medium |

### 5.8 Step 7: Framework-Specific Table Generation

**GHG Protocol Format:**
```
| Energy Type      | Location-Based (tCO2e) | Market-Based (tCO2e) | Diff (%) |
|-----------------|----------------------|---------------------|----------|
| Electricity     | X                    | Y                   | Z%       |
| Steam           | X                    | Y                   | Z%       |
| Heating         | X                    | Y                   | Z%       |
| Cooling         | X                    | Y                   | Z%       |
| Total Scope 2   | X                    | Y                   | Z%       |
```

**CSRD/ESRS E1-6 Format:**
```
Scope 2 GHG Emissions (ESRS E1-6):
  Gross location-based: XXXX tCO2e
  Gross market-based:   YYYY tCO2e
  Total GHG (loc-based): Scope1 + Scope2_loc + Scope3 = ZZZZ tCO2e
  Total GHG (mkt-based): Scope1 + Scope2_mkt + Scope3 = ZZZZ tCO2e
  Market-based coverage by instruments: ZZ%
  Bundled instruments: ZZ%
  Unbundled EACs: ZZ%
  Biogenic CO2: XX tCO2 (reported separately)
```

**CDP C6.3 Format:**
```
| Year | Scope 2 Location (tCO2e) | Scope 2 Market (tCO2e) |
|------|--------------------------|------------------------|
| 2024 | XXXX                     | YYYY                   |
| 2023 | XXXX                     | YYYY                   |
| 2022 | XXXX                     | YYYY                   |
```

**SBTi Progress Format:**
```
Target: X% reduction by YYYY from ZZZZ base year
Base year (ZZZZ):
  Location-based: XXXX tCO2e
  Market-based:   YYYY tCO2e
Current year:
  Location-based: XXXX tCO2e (ZZ% reduction)
  Market-based:   YYYY tCO2e (ZZ% reduction)
RE100 progress: ZZ% renewable electricity
```

**GRI 305-2 Format:**
```
305-2-a: Gross location-based Scope 2: XXXX tCO2e
305-2-b: Gross market-based Scope 2:   YYYY tCO2e
305-2-c: Gases included: CO2, CH4, N2O
305-2-d: Base year: ZZZZ
305-2-e: EF sources: [list]; GWP: AR6
305-2-f: Consolidation: Operational control
305-2-g: Methodology: GHG Protocol Scope 2 Guidance (2015)
```

### 5.9 Step 8: Year-over-Year Trend Analysis

Track the following metrics over time for both methods:

| Metric | Formula | Purpose |
|--------|---------|---------|
| Absolute change | Current - Previous | Raw change in tCO2e |
| Percentage change | (Current - Previous) / Previous * 100 | Rate of change |
| CAGR | (Current / Base)^(1/years) - 1 | Compound annual growth rate |
| Intensity (revenue) | Emissions / Revenue | Decoupling from growth |
| Intensity (employee) | Emissions / FTE | Per-employee efficiency |
| Intensity (sqft/m2) | Emissions / Floor area | Building efficiency |
| Discrepancy trend | Market_pct_diff[t] vs Market_pct_diff[t-1] | Procurement strategy impact |
| Coverage trend | Coverage_pct[t] vs Coverage_pct[t-1] | RE procurement progress |
| RE100 progress | RE_MWh / Total_MWh over time | Toward 100% renewable |

---

## 6. Emission Factor Hierarchy for Market-Based Method

### 6.1 GHG Protocol Instrument Hierarchy (Table 6.3)

The GHG Protocol Scope 2 Guidance defines a priority-ordered hierarchy
of contractual instruments for the market-based method:

| Priority | Instrument Type | Description | Typical EF (kgCO2e/kWh) | Quality |
|----------|----------------|-------------|-------------------------|---------|
| 1 | Supplier/utility-specific EF | Utility fuel mix disclosure with certificates sourced and retired on behalf of customers | Varies by supplier | Highest -- reflects actual supply |
| 2 | Supplier/utility-specific EF (no certificates) | Fuel mix disclosure without certificate backing | Varies by supplier | High -- but may include double counting risk |
| 3 | Energy attribute certificates (bundled) | RECs/GOs bundled with energy delivery (e.g., PPA) | 0.000 (renewable) | High -- bundled means closer to consumption |
| 4 | Energy attribute certificates (unbundled) | RECs, GOs, I-RECs purchased separately from energy | 0.000 (renewable) | Medium -- may have geographic/temporal gaps |
| 5 | Residual mix | Grid average minus all tracked renewable claims | Higher than grid avg | Default for uncovered consumption |
| 6 | Location-based (grid average) fallback | Used when no residual mix is available | Grid average EF | Lowest priority for market-based |

### 6.2 Decision Logic for EF Selection

The reconciliation agent must validate that MRV-010 applied the correct
hierarchy:

```
For each energy purchase:
  IF supplier_specific_ef_available AND meets_quality_criteria:
    USE supplier_specific_ef (Priority 1 or 2)
  ELIF bundled_certificates_available AND meets_quality_criteria:
    USE certificate_ef (Priority 3)
  ELIF unbundled_certificates_available AND meets_quality_criteria:
    USE certificate_ef (Priority 4)
  ELIF residual_mix_available:
    USE residual_mix_ef (Priority 5)
  ELSE:
    USE location_based_ef (Priority 6 -- fallback)
    FLAG as "no_market_specific_ef"
```

### 6.3 Residual Mix Factors by Region

When consumption is not covered by contractual instruments, the residual
mix EF applies. Residual mix EFs are HIGHER than grid averages because
they represent the grid mix after removing all claimed renewable
generation.

| Region | Grid Average (kgCO2e/kWh) | Residual Mix (kgCO2e/kWh) | Ratio |
|--------|--------------------------|--------------------------|-------|
| EU Average | 0.296 | 0.380 | 1.28x |
| Germany | 0.350 | 0.520 | 1.49x |
| France | 0.052 | 0.085 | 1.63x |
| UK | 0.207 | 0.285 | 1.38x |
| Spain | 0.187 | 0.290 | 1.55x |
| Italy | 0.315 | 0.420 | 1.33x |
| Netherlands | 0.336 | 0.450 | 1.34x |
| Sweden | 0.013 | 0.045 | 3.46x |
| Norway | 0.008 | 0.360 | 45.0x |
| Poland | 0.635 | 0.720 | 1.13x |
| US Average | 0.386 | 0.425 | 1.10x |
| US-CAMX (California) | 0.225 | 0.285 | 1.27x |
| US-ERCT (Texas) | 0.380 | 0.420 | 1.11x |
| US-MROE (Midwest) | 0.482 | 0.520 | 1.08x |
| US-SRSO (Southeast) | 0.390 | 0.440 | 1.13x |
| Australia | 0.656 | 0.750 | 1.14x |
| Japan | 0.465 | 0.520 | 1.12x |
| Singapore | 0.408 | 0.425 | 1.04x |
| South Korea | 0.459 | 0.510 | 1.11x |
| Global Average | 0.436 | 0.500 | 1.15x |

**Special case -- Norway:**
Norway has an extreme ratio (45x) because its grid is almost entirely
hydro (grid avg = 0.008), but when hydro certificates are exported to
other countries, the residual mix includes fossil imports that push the
residual factor to 0.360. This is a frequently misunderstood data point
in dual reporting.

### 6.4 Proposed Revision Impact on Hierarchy

Under the proposed 2025-2027 Scope 2 revisions:

| Current Rule | Proposed Change |
|-------------|----------------|
| Annual matching for certificates | Hourly matching required (with exemptions) |
| Same market/interconnected grid | Deliverability requirement (electrically connected) |
| Vintage within +/- 1 year | Tighter temporal matching |
| No additionality requirement | Implicit additionality through constraints |
| Residual mix as default | Fossil-only rates when residual mix unavailable |

The reconciliation agent should include a "future-readiness" flag
indicating whether the current instrument portfolio would comply with
the proposed hourly/deliverability requirements.

---

## 7. RE100 and Green Tariff Treatment in Dual Reporting

### 7.1 RE100 Technical Criteria

RE100 is the Climate Group initiative requiring member companies to
commit to 100% renewable electricity by 2050 (with interim targets).
RE100 criteria directly affect market-based Scope 2 accounting.

**RE100 Acceptable Procurement Methods:**

| Method | RE100 Acceptable | Market-Based Treatment | Location-Based Treatment |
|--------|-----------------|----------------------|------------------------|
| On-site generation (owned) | Yes (if additional) | Deducted from purchase | May reduce grid consumption |
| Direct-line PPA (physical) | Yes | 0 kgCO2e/kWh | Grid average EF |
| Virtual/financial PPA | Yes (with EAC retirement) | 0 kgCO2e/kWh | Grid average EF |
| Bundled green tariff | Yes | 0 kgCO2e/kWh | Grid average EF |
| Unbundled RECs/GOs/I-RECs | Yes (with restrictions) | 0 kgCO2e/kWh | Grid average EF |
| Default delivered RE | Conditional (EACs must be retired by utility) | Supplier-specific EF | Grid average EF |
| Standard supply (green mix) | No (unless EACs backing) | Residual mix EF | Grid average EF |

### 7.2 Green Tariff Treatment

Green tariffs are utility programs that provide customers with electricity
from designated renewable generation facilities.

**Key distinctions in dual reporting:**

1. **Bundled green tariff** (electricity + EACs delivered together):
   - Market-based: 0 kgCO2e/kWh (renewable EF)
   - Location-based: Grid average EF (unchanged by tariff)
   - RE100: Acceptable

2. **Unbundled green tariff** (separate EAC purchase):
   - Market-based: 0 kgCO2e/kWh if EAC quality criteria met
   - Location-based: Grid average EF (unchanged)
   - RE100: Acceptable with restrictions

3. **"100% renewable" standard supply** (no dedicated EACs):
   - Market-based: Residual mix EF (NOT zero -- no EACs to claim)
   - Location-based: Grid average EF
   - RE100: NOT acceptable
   - Common source of reporting errors

4. **Default delivered renewable** (utility has RE in mix):
   - Market-based: Supplier-specific EF IF utility retires EACs on behalf
     of customer; otherwise residual mix
   - Location-based: Grid average EF
   - RE100: Conditional

### 7.3 Reconciliation Agent's Role for RE100

The reconciliation agent should:

1. Calculate RE100 coverage percentage: RE_MWh / Total_Electricity_MWh
2. Validate that RE claims meet RE100 technical criteria
3. Cross-check: market-based zero-emission claims must have corresponding
   retired EACs in the instrument register
4. Flag any "claimed as renewable" that lacks EAC backing
5. Track RE100 progress toward interim and final targets
6. Compare RE100 percentage against market-based emission reduction to
   ensure consistency (100% RE should yield ~0 market-based electricity
   emissions, excluding residual steam/heat/cooling)

### 7.4 Additionality Considerations

While current GHG Protocol guidance does not require additionality for
Scope 2 accounting, the reconciliation agent should track and flag:

| Additionality Type | Description | Impact |
|-------------------|-------------|--------|
| Temporal additionality | New project built in response to demand | Highest impact |
| Financial additionality | Project would not exist without buyer commitment | High impact |
| None (existing project) | EAC from existing plant (no new capacity) | Lowest impact |
| Vintage age | How old the generating facility is | Affects SBTi V2 compliance |

The SBTi Net-Zero Standard V2 (draft) is considering limits on facility
age for EAC eligibility, which would make additionality tracking important
for future compliance.

---

## 8. Key Formulas for Reconciliation Metrics

### 8.1 Core Emission Formulas (Reference from Upstream Agents)

**Location-based electricity (MRV-009):**
```
E_loc_elec = Sum_f [ Activity_f * Grid_EF_f * (1 + TD_loss_f) ]
```

**Market-based electricity (MRV-010):**
```
E_mkt_elec = Sum_f [
    Sum_i (Instrument_MWh_i * Instrument_EF_i) +   // covered
    Uncovered_MWh_f * Residual_Mix_EF_f              // uncovered
]
```

**Location-based steam/heat (MRV-011):**
```
E_loc_steam = Sum_f [ Steam_GJ_f * Default_Steam_EF_region ]
E_loc_heat  = Sum_f [ Heat_GJ_f  * Default_DH_EF_region ]
```

**Market-based steam/heat (MRV-011):**
```
E_mkt_steam = Sum_f [ Steam_GJ_f * Supplier_Steam_EF_f ]
              // Falls back to default if no supplier EF
E_mkt_heat  = Sum_f [ Heat_GJ_f  * Supplier_DH_EF_f ]
```

**Location-based cooling (MRV-012):**
```
E_loc_cool = Sum_f [ Cooling_GJ_f * Default_Cool_EF_region ]
```

**Market-based cooling (MRV-012):**
```
E_mkt_cool = Sum_f [ Cooling_GJ_f * Supplier_Cool_EF_f ]
```

### 8.2 Reconciliation Formulas

**Total Scope 2 (Location-based):**
```
Total_S2_Loc = E_loc_elec + E_loc_steam + E_loc_heat + E_loc_cool
```

**Total Scope 2 (Market-based):**
```
Total_S2_Mkt = E_mkt_elec + E_mkt_steam + E_mkt_heat + E_mkt_cool
```

**Absolute Discrepancy:**
```
D_abs = Total_S2_Mkt - Total_S2_Loc
```

**Relative Discrepancy (percentage):**
```
D_rel = D_abs / Total_S2_Loc * 100
```

**Procurement Impact Factor:**
```
PIF = 1 - (Total_S2_Mkt / Total_S2_Loc)
// PIF > 0 means market-based is lower (RE procurement reducing emissions)
// PIF < 0 means market-based is higher (residual mix effect)
// PIF = 0 means methods are equivalent (no contractual impact)
```

**RE Coverage Percentage (electricity only):**
```
RE_Coverage = Sum(RE_Instrument_MWh) / Total_Electricity_MWh * 100
```

**Instrument Coverage Percentage (all instruments):**
```
Instr_Coverage = Sum(All_Instrument_MWh) / Total_Electricity_MWh * 100
```

**Market-Based Emission Factor (effective, electricity):**
```
EF_mkt_effective = Total_S2_Mkt_Elec / Total_Electricity_MWh
```

**Location-Based Emission Factor (effective, electricity):**
```
EF_loc_effective = Total_S2_Loc_Elec / Total_Electricity_MWh
```

**EF Discount (market vs. location):**
```
EF_Discount = (EF_loc_effective - EF_mkt_effective) / EF_loc_effective * 100
```

### 8.3 Year-over-Year Trend Formulas

**Absolute YoY Change:**
```
YoY_abs_loc = S2_Loc[t] - S2_Loc[t-1]
YoY_abs_mkt = S2_Mkt[t] - S2_Mkt[t-1]
```

**Percentage YoY Change:**
```
YoY_pct_loc = (S2_Loc[t] - S2_Loc[t-1]) / S2_Loc[t-1] * 100
YoY_pct_mkt = (S2_Mkt[t] - S2_Mkt[t-1]) / S2_Mkt[t-1] * 100
```

**Compound Annual Growth Rate (CAGR):**
```
CAGR_loc = (S2_Loc[current] / S2_Loc[base])^(1 / years) - 1
CAGR_mkt = (S2_Mkt[current] / S2_Mkt[base])^(1 / years) - 1
```

**Discrepancy Trend:**
```
D_rel_trend = D_rel[t] - D_rel[t-1]
// Positive: discrepancy growing (more RE procurement or rising residual mix)
// Negative: discrepancy shrinking (less RE procurement or falling residual mix)
```

**SBTi Target Progress (linear reduction):**
```
Required_annual_reduction = (Base_emissions - Target_emissions) / Target_years
Current_progress = (Base_emissions - Current_emissions) / Base_emissions * 100
On_track = Current_emissions <= Base_emissions - (Required_annual_reduction * Years_elapsed)
```

### 8.4 Intensity Formulas

**Revenue Intensity:**
```
I_rev_loc = Total_S2_Loc / Revenue
I_rev_mkt = Total_S2_Mkt / Revenue
```

**Employee Intensity:**
```
I_fte_loc = Total_S2_Loc / FTE_count
I_fte_mkt = Total_S2_Mkt / FTE_count
```

**Floor Area Intensity:**
```
I_area_loc = Total_S2_Loc / Total_floor_area_m2
I_area_mkt = Total_S2_Mkt / Total_floor_area_m2
```

**Production Intensity:**
```
I_prod_loc = Total_S2_Loc / Production_units
I_prod_mkt = Total_S2_Mkt / Production_units
```

### 8.5 Quality Score Sub-Formulas

**Completeness Score:**
```
Completeness_Loc = Count(facilities_with_loc_result) / Count(all_facilities)
Completeness_Mkt = Count(facilities_with_mkt_result) / Count(all_facilities)
Energy_Type_Coverage_Loc = Count(energy_types_with_loc) / Count(all_energy_types)
Energy_Type_Coverage_Mkt = Count(energy_types_with_mkt) / Count(all_energy_types)
Completeness = min(Completeness_Loc, Completeness_Mkt,
                   Energy_Type_Coverage_Loc, Energy_Type_Coverage_Mkt)
```

**Consistency Score:**
```
Activity_Data_Match = 1 if abs(AD_loc - AD_mkt) / AD_loc < 0.01 else 0
GWP_Match = 1 if GWP_loc == GWP_mkt else 0
Gas_Match = 1 if gases_loc == gases_mkt else 0
Boundary_Match = 1 if boundary_loc == boundary_mkt else 0
Period_Match = 1 if period_loc == period_mkt else 0
Consistency = (Activity_Data_Match + GWP_Match + Gas_Match +
               Boundary_Match + Period_Match) / 5
```

**Accuracy Score (emission-weighted):**
```
Accuracy = Sum(emission_i * data_quality_score_i) / Sum(emission_i)
  where data_quality_score depends on:
    - activity data source (meter=1.0, invoice=0.9, estimate=0.6)
    - EF source quality (audited supplier=0.95, subregional=0.9, national=0.7)
```

**Transparency Score:**
```
Items = [ef_sources, instrument_types, coverage_pct, residual_mix_source,
         exclusions, methodology, base_year, gwp_stated, gases_stated]
Transparency = Count(documented_items) / Count(all_items)
```

---

## 9. Suggested 7-Engine Architecture

### 9.1 Overview

The Dual Reporting Reconciliation Agent follows the established GreenLang
7-engine pattern used by all MRV agents. Each engine has a focused
responsibility within the reconciliation pipeline.

```
+-------------------------------------------------------------------+
|                   AGENT-MRV-013 Pipeline                          |
|                                                                   |
|  [MRV-009] --+                                                    |
|  [MRV-010] --+--> Engine 1 --> Engine 2 --> Engine 3 --> Engine 4 |
|  [MRV-011] --+      |           |            |            |      |
|  [MRV-012] --+      v           v            v            v      |
|                   Collect     Compare      Quality      Tables    |
|                                                                   |
|              Engine 5 --> Engine 6 --> Engine 7                    |
|                |           |           |                          |
|                v           v           v                          |
|             Trend       Compliance   Pipeline                     |
+-------------------------------------------------------------------+
```

### 9.2 Engine 1: DualResultCollectorEngine (`dual_result_collector.py`)

**Purpose:** Collects, validates, and aligns results from all four upstream
Scope 2 agents into a unified reconciliation workspace.

**Responsibilities:**
- Fetch calculation results from MRV-009, 010, 011, 012 for a given period
- Validate that all four agents have completed for the same reporting period
- Align organizational boundaries (operational/financial/equity control)
- Verify facility sets match across all four agents
- Normalize units (tCO2e, MWh, GJ) across all results
- Verify GWP source consistency (AR4/AR5/AR6)
- Verify gas coverage consistency (CO2, CH4, N2O)
- Build the unified Energy Purchase Register linking each purchase to
  its location-based and market-based results
- Handle missing upstream results gracefully (partial reconciliation)
- Track data provenance (which agent, which calculation run, timestamp)

**Key Data Structures:**
- `UpstreamResult`: Standardized result from any upstream agent
- `ReconciliationWorkspace`: Unified workspace for a reporting period
- `EnergyPurchaseRegister`: Per-facility, per-energy-type mapping

**Public Methods (~20):**
- `collect_upstream_results(period, org_id)` -> `ReconciliationWorkspace`
- `validate_alignment(workspace)` -> `AlignmentReport`
- `build_energy_register(workspace)` -> `EnergyPurchaseRegister`
- `get_missing_results(workspace)` -> `List[MissingResult]`
- `normalize_units(workspace, target_units)` -> `ReconciliationWorkspace`
- `verify_gwp_consistency(workspace)` -> `ConsistencyResult`
- `verify_gas_consistency(workspace)` -> `ConsistencyResult`
- `verify_boundary_consistency(workspace)` -> `ConsistencyResult`
- `verify_period_alignment(workspace)` -> `ConsistencyResult`
- `get_facility_coverage(workspace)` -> `FacilityCoverageMap`
- And 10+ additional utility/query methods

### 9.3 Engine 2: DiscrepancyAnalyzerEngine (`discrepancy_analyzer.py`)

**Purpose:** Calculates and classifies discrepancies between location-based
and market-based results at every aggregation level.

**Responsibilities:**
- Calculate absolute and relative discrepancies at 4 levels:
  (1) facility+energy_type, (2) energy_type, (3) facility, (4) organization
- Classify discrepancy type (from the 8-type taxonomy in Section 3.2)
- Classify discrepancy direction (REC_DOMINANT, MIXED_LOWER, ALIGNED,
  RESIDUAL_EFFECT, QUALITY_CONCERN)
- Assess materiality against configurable thresholds
- Decompose discrepancies into contributing factors:
  - Renewable certificate impact
  - Residual mix uplift
  - Supplier-specific factor difference
  - Geographic mismatch effect
  - Temporal mismatch effect
- Generate per-gas discrepancy breakdown (CO2, CH4, N2O)
- Identify the largest contributors to the overall discrepancy
- Produce a waterfall decomposition of market vs. location difference

**Key Data Structures:**
- `Discrepancy`: Single discrepancy record with absolute, relative, type
- `DiscrepancyReport`: Full report at all aggregation levels
- `WaterfallDecomposition`: Factors contributing to total discrepancy
- `ContributorRanking`: Ranked list of largest discrepancy contributors

**Public Methods (~20):**
- `analyze_all(workspace)` -> `DiscrepancyReport`
- `analyze_facility(workspace, facility_id)` -> `List[Discrepancy]`
- `analyze_energy_type(workspace, energy_type)` -> `Discrepancy`
- `classify_discrepancy(discrepancy)` -> `DiscrepancyType`
- `assess_materiality(discrepancy)` -> `MaterialityLevel`
- `decompose_waterfall(workspace)` -> `WaterfallDecomposition`
- `rank_contributors(report)` -> `ContributorRanking`
- `calculate_procurement_impact(workspace)` -> `ProcurementImpact`
- `per_gas_breakdown(discrepancy)` -> `GasBreakdown`
- And 10+ additional analysis methods

### 9.4 Engine 3: QualityScorerEngine (`quality_scorer.py`)

**Purpose:** Computes comprehensive quality scores for the dual reporting
package across completeness, consistency, accuracy, and transparency.

**Responsibilities:**
- Compute Completeness score per Section 4.2 Dimension 1
  - Facility coverage in both methods
  - Energy type coverage in both methods
  - Temporal coverage (all months/quarters)
  - Missing data identification
- Compute Consistency score per Section 4.2 Dimension 2
  - Activity data match between methods
  - GWP source match
  - Gas coverage match
  - Boundary match
  - Period match
- Compute Accuracy score per Section 4.2 Dimension 3
  - Data source quality (meter vs. estimate)
  - EF source quality (audited vs. default)
  - Emission-weighted accuracy aggregation
- Compute Transparency score per Section 4.2 Dimension 4
  - Documentation completeness checklist
  - EF source citation
  - Instrument type disclosure
  - Coverage percentage disclosure
- Compute Composite Quality Score with configurable weights
- Assign quality grade (A/B/C/D/F)
- Validate Scope 2 Quality Criteria for each instrument (8 criteria)
- Generate quality improvement recommendations

**Key Data Structures:**
- `QualityAssessment`: Complete quality assessment with all 4 dimensions
- `CompletenessDetail`: Detailed completeness check results
- `ConsistencyDetail`: Detailed consistency check results
- `AccuracyDetail`: Emission-weighted accuracy breakdown
- `TransparencyDetail`: Documentation checklist results
- `QualityRecommendation`: Improvement recommendations

**Public Methods (~25):**
- `assess_quality(workspace)` -> `QualityAssessment`
- `score_completeness(workspace)` -> `CompletenessDetail`
- `score_consistency(workspace)` -> `ConsistencyDetail`
- `score_accuracy(workspace)` -> `AccuracyDetail`
- `score_transparency(workspace)` -> `TransparencyDetail`
- `compute_composite(details)` -> `Decimal`
- `assign_grade(composite_score)` -> `QualityGrade`
- `validate_instrument_quality(instrument)` -> `InstrumentQualityResult`
- `validate_all_instruments(workspace)` -> `List[InstrumentQualityResult]`
- `generate_recommendations(assessment)` -> `List[QualityRecommendation]`
- And 15+ additional quality methods

### 9.5 Engine 4: ReportingTableGeneratorEngine (`reporting_table_generator.py`)

**Purpose:** Generates framework-specific dual reporting tables and
disclosure outputs for GHG Protocol, CSRD, CDP, SBTi, and GRI.

**Responsibilities:**
- Generate GHG Protocol Chapter 7 dual reporting table
- Generate CSRD/ESRS E1-6 disclosure package with all required datapoints
  (E1-6 para 49, 52, E1-6_18 through E1-6_24)
- Generate CDP C6.3 multi-year table with both methods
- Generate CDP C6.4 market-based instrument breakdown
- Generate CDP C8.2 energy consumption summary
- Generate SBTi target progress report (both methods)
- Generate GRI 305-2 disclosure with all sub-disclosures
- Generate ISO 14064-1 Category 2 report
- Export in multiple formats: JSON, CSV, Excel, PDF template
- Include narrative explanations for discrepancies
- Include footnotes for methodology and assumptions
- Cross-reference framework requirements to ensure completeness

**Key Data Structures:**
- `DualReportingTable`: Universal table structure adaptable to any framework
- `FrameworkDisclosure`: Framework-specific disclosure package
- `NarrativeExplanation`: Auto-generated explanation of key findings
- `ExportConfig`: Export format and template configuration

**Public Methods (~25):**
- `generate_ghgp_table(workspace)` -> `DualReportingTable`
- `generate_csrd_disclosure(workspace)` -> `FrameworkDisclosure`
- `generate_cdp_c63(workspace)` -> `FrameworkDisclosure`
- `generate_cdp_c64(workspace)` -> `FrameworkDisclosure`
- `generate_cdp_c82(workspace)` -> `FrameworkDisclosure`
- `generate_sbti_progress(workspace, targets)` -> `FrameworkDisclosure`
- `generate_gri_305_2(workspace)` -> `FrameworkDisclosure`
- `generate_iso14064(workspace)` -> `FrameworkDisclosure`
- `generate_all_frameworks(workspace)` -> `Dict[Framework, FrameworkDisclosure]`
- `export_json(disclosure)` -> `str`
- `export_csv(disclosure)` -> `bytes`
- `export_excel(disclosure)` -> `bytes`
- `generate_narrative(workspace, discrepancies)` -> `NarrativeExplanation`
- And 12+ additional formatting/export methods

### 9.6 Engine 5: TrendAnalysisEngine (`trend_analysis.py`)

**Purpose:** Calculates year-over-year trends, multi-year analysis, and
intensity metrics for both methods over time.

**Responsibilities:**
- Calculate absolute and percentage YoY change for both methods
- Calculate CAGR from base year for both methods
- Calculate emission intensity ratios (revenue, FTE, area, production)
  for both methods
- Track discrepancy trend over time (growing/shrinking/stable)
- Track RE coverage trend (progress toward 100%)
- Track instrument coverage trend
- Track effective emission factor trend
- Detect anomalies in year-over-year changes
- Calculate SBTi target trajectory and on/off-track status
- Generate sparkline-compatible trend data for dashboards
- Decomposition of emissions change into drivers:
  - Activity change (more/less energy consumed)
  - Grid change (cleaner/dirtier grid)
  - Procurement change (more/less RE purchased)
  - Structural change (acquisitions/divestitures)

**Key Data Structures:**
- `TrendSeries`: Time series of a single metric
- `TrendReport`: Complete trend analysis for both methods
- `IntensityReport`: Intensity metrics over time
- `TargetTrajectory`: SBTi/custom target progress over time
- `ChangeDecomposition`: Drivers of emission change

**Public Methods (~20):**
- `analyze_trends(org_id, periods)` -> `TrendReport`
- `calculate_yoy(metric, periods)` -> `TrendSeries`
- `calculate_cagr(metric, base_period, current_period)` -> `Decimal`
- `calculate_intensities(workspace, denominators)` -> `IntensityReport`
- `track_discrepancy_trend(org_id, periods)` -> `TrendSeries`
- `track_re_progress(org_id, periods)` -> `TrendSeries`
- `track_target_progress(org_id, target, periods)` -> `TargetTrajectory`
- `detect_anomalies(series, threshold)` -> `List[Anomaly]`
- `decompose_change(period_a, period_b)` -> `ChangeDecomposition`
- And 10+ additional trend methods

### 9.7 Engine 6: ComplianceCheckerEngine (`compliance_checker.py`)

**Purpose:** Validates the dual reporting package against multi-framework
compliance requirements and generates compliance status reports.

**Responsibilities:**
- Validate GHG Protocol Chapter 7 dual reporting completeness
  - Both methods reported
  - All disclosure items present
  - Quality criteria met for all instruments
- Validate CSRD/ESRS E1-6 compliance
  - All required datapoints present (E1-6 para 49, 52, E1-6_18-24)
  - Two total GHG figures produced
  - Biogenic CO2 separately disclosed
- Validate CDP C6.3 Route A compliance
  - Both location and market reported
  - Historical years provided
  - C6.4 instrument breakdown if applicable
- Validate SBTi compliance
  - Target method reported correctly
  - Base year consistent
  - Progress calculated per SBTi methodology
  - RE100 thresholds met (if applicable)
- Validate GRI 305-2 compliance
  - All disclosure items (a through g) present
  - Both methods if operating in instrument markets
- Instrument-level quality criteria validation (8 criteria)
- Geographic matching validation
- Temporal matching validation
- Double-counting detection (same EAC used by multiple facilities)
- Future-readiness assessment (hourly matching, deliverability)
- Generate compliance scorecard per framework

**Key Data Structures:**
- `ComplianceResult`: Per-framework compliance result
- `ComplianceScorecard`: Multi-framework scorecard
- `ComplianceViolation`: Individual violation with severity and remediation
- `FutureReadinessReport`: Assessment against proposed Scope 2 revisions

**Public Methods (~25):**
- `check_all_frameworks(workspace)` -> `ComplianceScorecard`
- `check_ghgp(workspace)` -> `ComplianceResult`
- `check_csrd(workspace)` -> `ComplianceResult`
- `check_cdp(workspace)` -> `ComplianceResult`
- `check_sbti(workspace, targets)` -> `ComplianceResult`
- `check_gri(workspace)` -> `ComplianceResult`
- `check_iso14064(workspace)` -> `ComplianceResult`
- `validate_instrument_criteria(instrument)` -> `List[ComplianceViolation]`
- `detect_double_counting(workspace)` -> `List[DoubleCounting]`
- `assess_future_readiness(workspace)` -> `FutureReadinessReport`
- `generate_remediation_plan(violations)` -> `RemediationPlan`
- And 14+ additional compliance methods

### 9.8 Engine 7: DualReportingPipelineEngine (`dual_reporting_pipeline.py`)

**Purpose:** Orchestrates the full 8-stage reconciliation pipeline,
managing the execution order, error handling, and result assembly.

**Responsibilities:**
- Stage 1: Validate input request (organization, period, configuration)
- Stage 2: Collect upstream results via Engine 1
- Stage 3: Analyze discrepancies via Engine 2
- Stage 4: Score quality via Engine 3
- Stage 5: Generate reporting tables via Engine 4
- Stage 6: Analyze trends via Engine 5 (if historical data available)
- Stage 7: Run compliance checks via Engine 6
- Stage 8: Assemble final reconciliation report with all components
- Handle partial execution (some engines can run even if others fail)
- Configurable pipeline (skip engines, custom thresholds, framework selection)
- Async execution with progress callbacks
- Result caching for performance
- Audit trail generation (who ran, when, what parameters)
- Integration with OBS-001-005 for metrics and tracing

**Key Data Structures:**
- `ReconciliationRequest`: Input request with all configuration
- `ReconciliationReport`: Final comprehensive report
- `PipelineConfig`: Pipeline configuration (engines to run, thresholds)
- `PipelineProgress`: Stage-by-stage progress tracking
- `AuditEntry`: Audit trail record

**Pipeline Stages:**
```
Stage 1: VALIDATE    -> ReconciliationRequest validated
Stage 2: COLLECT     -> ReconciliationWorkspace populated
Stage 3: DISCREPANCY -> DiscrepancyReport generated
Stage 4: QUALITY     -> QualityAssessment computed
Stage 5: REPORTING   -> Framework tables generated
Stage 6: TREND       -> TrendReport computed (optional)
Stage 7: COMPLIANCE  -> ComplianceScorecard produced
Stage 8: ASSEMBLE    -> ReconciliationReport finalized
```

**Public Methods (~20):**
- `run_pipeline(request)` -> `ReconciliationReport`
- `run_stage(stage, workspace)` -> `StageResult`
- `validate_request(request)` -> `ValidationResult`
- `configure_pipeline(config)` -> `PipelineConfig`
- `get_progress(run_id)` -> `PipelineProgress`
- `cancel_pipeline(run_id)` -> `bool`
- `retry_stage(run_id, stage)` -> `StageResult`
- `get_report(run_id)` -> `ReconciliationReport`
- `list_runs(org_id, filters)` -> `List[RunSummary]`
- `delete_run(run_id)` -> `bool`
- And 10+ additional orchestration methods

---

## 10. Suggested Enums, Models, and Constants

### 10.1 Enumerations (22)

```python
class ReconciliationMethod(str, Enum):
    """The two Scope 2 accounting methods."""
    LOCATION_BASED = "location_based"
    MARKET_BASED = "market_based"

class EnergyType(str, Enum):
    """Types of purchased energy in Scope 2."""
    ELECTRICITY = "electricity"
    STEAM = "steam"
    HEATING = "heating"
    COOLING = "cooling"

class DiscrepancyType(str, Enum):
    """Classification of discrepancy causes between methods."""
    REC_IMPACT = "rec_impact"                   # Renewable certificates driving market below location
    RESIDUAL_MIX_UPLIFT = "residual_mix_uplift" # Residual mix higher than grid average
    SUPPLIER_FACTOR = "supplier_factor"          # Supplier-specific EF differs from grid avg
    GEOGRAPHIC_MISMATCH = "geographic_mismatch"  # Certificate from different grid region
    TEMPORAL_MISMATCH = "temporal_mismatch"       # Vintage year vs consumption year mismatch
    PARTIAL_COVERAGE = "partial_coverage"         # Mix of instruments and residual/grid fallback
    STEAM_HEAT_COOL_DIFF = "steam_heat_cool_diff" # Non-electricity EF differences
    GRID_EF_TIMING = "grid_ef_timing"             # Grid EF publication lag vs current contracts

class DiscrepancyDirection(str, Enum):
    """Direction of discrepancy between market and location methods."""
    REC_DOMINANT = "rec_dominant"       # Market << Location (strong RE procurement)
    MIXED_LOWER = "mixed_lower"         # Market < Location (partial RE / lower supplier)
    ALIGNED = "aligned"                 # Market ~ Location (minimal contractual impact)
    RESIDUAL_EFFECT = "residual_effect" # Market > Location (residual mix effect)
    QUALITY_CONCERN = "quality_concern" # Market >> Location (investigate data quality)

class MaterialityLevel(str, Enum):
    """Materiality classification for discrepancies."""
    IMMATERIAL = "immaterial"   # < 5%
    MINOR = "minor"             # 5-15%
    MATERIAL = "material"       # 15-50%
    SIGNIFICANT = "significant" # 50-100%
    EXTREME = "extreme"         # > 100%

class QualityGrade(str, Enum):
    """Overall quality grade for dual reporting package."""
    A = "A"  # 0.90-1.00: Excellent, assurance-ready
    B = "B"  # 0.80-0.89: Good, minor improvements
    C = "C"  # 0.70-0.79: Acceptable, several issues
    D = "D"  # 0.60-0.69: Below standard
    F = "F"  # 0.00-0.59: Failing

class QualityDimension(str, Enum):
    """The four quality dimensions for dual reporting."""
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    TRANSPARENCY = "transparency"

class Scope2QualityCriterion(str, Enum):
    """GHG Protocol 8 Scope 2 Quality Criteria for instruments."""
    CONVEY_GHG_RATE = "convey_ghg_rate"           # Criterion 1
    UNIQUE_CLAIM = "unique_claim"                   # Criterion 2
    TRACKED_RETIRED = "tracked_retired"             # Criterion 3
    TEMPORAL_MATCH = "temporal_match"               # Criterion 4
    GEOGRAPHIC_MATCH = "geographic_match"           # Criterion 5
    SUPPLIER_SPECIFIC = "supplier_specific"         # Criterion 6
    DIRECT_CONTRACT = "direct_contract"             # Criterion 7
    RESIDUAL_MIX_AVAILABLE = "residual_mix_available" # Criterion 8

class ReportingFramework(str, Enum):
    """Regulatory/voluntary frameworks requiring dual reporting."""
    GHGP = "ghgp"             # GHG Protocol Scope 2 Guidance
    CSRD_ESRS_E1 = "csrd_esrs_e1"  # EU CSRD ESRS E1
    CDP = "cdp"               # CDP Climate Change Questionnaire
    SBTI = "sbti"             # Science Based Targets initiative
    GRI_305 = "gri_305"       # GRI 305: Emissions
    ISO_14064 = "iso_14064"   # ISO 14064-1
    RE100 = "re100"           # RE100 (Climate Group)

class ComplianceStatus(str, Enum):
    """Compliance status for a framework check."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_ASSESSED = "not_assessed"

class FlagSeverity(str, Enum):
    """Severity level for reconciliation flags."""
    CRITICAL = "critical"  # Blocks reporting
    HIGH = "high"           # Must fix before external reporting
    MEDIUM = "medium"       # Should fix; explain if not
    LOW = "low"             # Nice to fix; informational
    INFO = "info"           # Informational only

class FlagType(str, Enum):
    """Types of reconciliation flags."""
    MISSING_LOCATION = "missing_location"
    MISSING_MARKET = "missing_market"
    ACTIVITY_DATA_MISMATCH = "activity_data_mismatch"
    GWP_MISMATCH = "gwp_mismatch"
    LARGE_DISCREPANCY = "large_discrepancy"
    MARKET_ABOVE_LOCATION = "market_above_location"
    INSTRUMENT_QUALITY_FAIL = "instrument_quality_fail"
    TEMPORAL_MISMATCH = "temporal_mismatch"
    GEOGRAPHIC_MISMATCH = "geographic_mismatch"
    COVERAGE_GAP = "coverage_gap"
    MISSING_RESIDUAL_MIX = "missing_residual_mix"
    FACILITY_MISSING = "facility_missing"
    BASE_YEAR_INCONSISTENT = "base_year_inconsistent"
    BIOGENIC_MISSING = "biogenic_missing"
    EF_OUTDATED = "ef_outdated"
    DOUBLE_COUNTING = "double_counting"
    FUTURE_NON_COMPLIANT = "future_non_compliant"

class InstrumentType(str, Enum):
    """Types of contractual instruments (reused from MRV-010)."""
    PPA = "ppa"
    REC = "rec"
    GO = "go"
    REGO = "rego"
    I_REC = "i_rec"
    T_REC = "t_rec"
    J_CREDIT = "j_credit"
    LGC = "lgc"
    GREEN_TARIFF = "green_tariff"
    SUPPLIER_SPECIFIC = "supplier_specific"

class EFHierarchyPriority(str, Enum):
    """Market-based EF hierarchy priority levels per GHG Protocol Table 6.3."""
    SUPPLIER_WITH_CERTS = "supplier_with_certs"     # Priority 1
    SUPPLIER_NO_CERTS = "supplier_no_certs"         # Priority 2
    BUNDLED_CERTIFICATES = "bundled_certificates"   # Priority 3
    UNBUNDLED_CERTIFICATES = "unbundled_certificates" # Priority 4
    RESIDUAL_MIX = "residual_mix"                   # Priority 5
    GRID_AVERAGE_FALLBACK = "grid_average_fallback" # Priority 6

class IntensityDenominator(str, Enum):
    """Denominators for emission intensity calculations."""
    REVENUE = "revenue"
    FTE = "fte"
    FLOOR_AREA_M2 = "floor_area_m2"
    FLOOR_AREA_SQFT = "floor_area_sqft"
    PRODUCTION_UNITS = "production_units"
    CUSTOMERS = "customers"

class TrendDirection(str, Enum):
    """Direction of a trend over time."""
    DECREASING_FAST = "decreasing_fast"   # > 10% annual decrease
    DECREASING = "decreasing"             # 2-10% annual decrease
    STABLE = "stable"                     # -2% to +2%
    INCREASING = "increasing"             # 2-10% annual increase
    INCREASING_FAST = "increasing_fast"   # > 10% annual increase

class PipelineStage(str, Enum):
    """Stages of the reconciliation pipeline."""
    VALIDATE = "validate"
    COLLECT = "collect"
    DISCREPANCY = "discrepancy"
    QUALITY = "quality"
    REPORTING = "reporting"
    TREND = "trend"
    COMPLIANCE = "compliance"
    ASSEMBLE = "assemble"

class StageStatus(str, Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class GWPSource(str, Enum):
    """GWP source for CO2e conversion (reused from upstream agents)."""
    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"
    AR6_20YR = "AR6_20YR"

class EmissionGas(str, Enum):
    """GHG gases tracked in Scope 2 (reused from upstream agents)."""
    CO2 = "co2"
    CH4 = "ch4"
    N2O = "n2o"
```

### 10.2 Data Models (25)

```python
@dataclass
class UpstreamResult:
    """Standardized result from an upstream Scope 2 agent."""
    agent_id: str                     # e.g., "MRV-009"
    calculation_id: str               # Upstream calculation UUID
    organization_id: str
    facility_id: str
    energy_type: EnergyType
    method: ReconciliationMethod
    reporting_period_start: date
    reporting_period_end: date
    activity_data_value: Decimal       # MWh or GJ
    activity_data_unit: str
    total_emissions_tco2e: Decimal
    co2_emissions: Decimal
    ch4_emissions_tco2e: Decimal
    n2o_emissions_tco2e: Decimal
    emission_factor_used: Decimal
    emission_factor_source: str
    emission_factor_unit: str
    gwp_source: GWPSource
    instruments_used: Optional[List[str]]  # For market-based
    coverage_percentage: Optional[Decimal]  # For market-based
    data_quality_tier: str
    calculated_at: datetime

@dataclass
class ReconciliationWorkspace:
    """Unified workspace for a single reconciliation run."""
    workspace_id: str
    organization_id: str
    reporting_period_start: date
    reporting_period_end: date
    upstream_results: List[UpstreamResult]
    energy_register: Optional[EnergyPurchaseRegister]
    alignment_status: Optional[AlignmentReport]
    created_at: datetime

@dataclass
class EnergyPurchaseEntry:
    """Single energy purchase mapped to both methods."""
    facility_id: str
    facility_name: str
    energy_type: EnergyType
    activity_data_value: Decimal
    activity_data_unit: str
    location_based_emissions: Optional[Decimal]
    location_based_ef: Optional[Decimal]
    location_based_source_agent: Optional[str]
    market_based_emissions: Optional[Decimal]
    market_based_ef: Optional[Decimal]
    market_based_source_agent: Optional[str]
    instruments: List[str]
    coverage_status: str     # "fully_covered", "partially_covered", "uncovered"
    coverage_percentage: Optional[Decimal]

@dataclass
class EnergyPurchaseRegister:
    """Complete register of all energy purchases for reconciliation."""
    entries: List[EnergyPurchaseEntry]
    total_facilities: int
    total_energy_types: int
    total_activity_data: Decimal

@dataclass
class AlignmentReport:
    """Result of alignment checks between upstream results."""
    period_aligned: bool
    boundary_aligned: bool
    facility_set_aligned: bool
    gwp_aligned: bool
    gas_coverage_aligned: bool
    unit_aligned: bool
    missing_facilities_location: List[str]
    missing_facilities_market: List[str]
    mismatches: List[str]

@dataclass
class Discrepancy:
    """Single discrepancy between location and market methods."""
    level: str                         # "facility", "energy_type", "organization"
    facility_id: Optional[str]
    energy_type: Optional[EnergyType]
    location_based_tco2e: Decimal
    market_based_tco2e: Decimal
    absolute_diff_tco2e: Decimal
    relative_diff_pct: Decimal
    discrepancy_type: DiscrepancyType
    direction: DiscrepancyDirection
    materiality: MaterialityLevel

@dataclass
class DiscrepancyReport:
    """Complete discrepancy analysis at all levels."""
    facility_level: List[Discrepancy]
    energy_type_level: List[Discrepancy]
    organization_level: Discrepancy
    waterfall: WaterfallDecomposition
    top_contributors: List[ContributorRanking]
    total_location_tco2e: Decimal
    total_market_tco2e: Decimal
    procurement_impact_factor: Decimal

@dataclass
class WaterfallDecomposition:
    """Decomposition of total discrepancy into contributing factors."""
    re_certificate_impact: Decimal     # Impact of RECs/GOs
    residual_mix_uplift: Decimal       # Residual mix above grid average
    supplier_factor_impact: Decimal    # Supplier-specific EF difference
    geographic_mismatch_impact: Decimal
    temporal_mismatch_impact: Decimal
    steam_heat_cool_impact: Decimal
    total_discrepancy: Decimal

@dataclass
class ContributorRanking:
    """A ranked contributor to discrepancy."""
    rank: int
    facility_id: str
    energy_type: EnergyType
    absolute_contribution: Decimal
    percentage_of_total: Decimal
    discrepancy_type: DiscrepancyType

@dataclass
class QualityAssessment:
    """Complete quality assessment of dual reporting package."""
    completeness_score: Decimal
    consistency_score: Decimal
    accuracy_score: Decimal
    transparency_score: Decimal
    composite_score: Decimal
    grade: QualityGrade
    completeness_detail: CompletenessDetail
    consistency_detail: ConsistencyDetail
    accuracy_detail: AccuracyDetail
    transparency_detail: TransparencyDetail
    recommendations: List[QualityRecommendation]

@dataclass
class CompletenessDetail:
    """Detailed completeness check results."""
    facility_coverage_location: Decimal
    facility_coverage_market: Decimal
    energy_type_coverage_location: Decimal
    energy_type_coverage_market: Decimal
    temporal_coverage: Decimal
    missing_items: List[str]

@dataclass
class ConsistencyDetail:
    """Detailed consistency check results."""
    activity_data_match: bool
    gwp_source_match: bool
    gas_coverage_match: bool
    boundary_match: bool
    period_match: bool
    mismatches: List[str]

@dataclass
class AccuracyDetail:
    """Emission-weighted accuracy breakdown."""
    overall_accuracy: Decimal
    per_source_accuracy: Dict[str, Decimal]
    activity_data_quality: Decimal
    ef_quality: Decimal

@dataclass
class TransparencyDetail:
    """Documentation checklist for transparency."""
    ef_sources_documented: bool
    instrument_types_disclosed: bool
    coverage_pct_stated: bool
    residual_mix_source_identified: bool
    exclusions_justified: bool
    methodology_cited: bool
    base_year_stated: bool
    gwp_stated: bool
    gases_stated: bool

@dataclass
class QualityRecommendation:
    """Quality improvement recommendation."""
    dimension: QualityDimension
    severity: FlagSeverity
    description: str
    remediation: str
    impact_on_score: Decimal

@dataclass
class ReconciliationFlag:
    """A flag raised during reconciliation."""
    flag_type: FlagType
    severity: FlagSeverity
    facility_id: Optional[str]
    energy_type: Optional[EnergyType]
    description: str
    remediation: str

@dataclass
class DualReportingTable:
    """Framework-agnostic dual reporting table."""
    rows: List[DualReportingRow]
    total_location_tco2e: Decimal
    total_market_tco2e: Decimal
    total_discrepancy_pct: Decimal
    coverage_percentage: Decimal
    instrument_breakdown: Dict[str, Decimal]  # instrument_type -> MWh
    metadata: Dict[str, Any]

@dataclass
class DualReportingRow:
    """Single row in a dual reporting table."""
    energy_type: EnergyType
    location_tco2e: Decimal
    market_tco2e: Decimal
    discrepancy_pct: Decimal
    activity_data: Decimal
    activity_unit: str

@dataclass
class FrameworkDisclosure:
    """Framework-specific disclosure package."""
    framework: ReportingFramework
    datapoints: Dict[str, Any]
    tables: List[DualReportingTable]
    narrative: Optional[str]
    footnotes: List[str]
    compliance_status: ComplianceStatus
    generated_at: datetime

@dataclass
class TrendSeries:
    """Time series of a single metric."""
    metric_name: str
    unit: str
    datapoints: List[Tuple[date, Decimal]]
    direction: TrendDirection
    cagr: Optional[Decimal]

@dataclass
class TrendReport:
    """Complete trend analysis for both methods."""
    location_trend: TrendSeries
    market_trend: TrendSeries
    discrepancy_trend: TrendSeries
    coverage_trend: TrendSeries
    intensity_trends: Dict[IntensityDenominator, TrendSeries]
    target_trajectory: Optional[TargetTrajectory]

@dataclass
class TargetTrajectory:
    """SBTi or custom target progress tracking."""
    target_type: str              # "location_absolute", "market_absolute", "re100"
    base_year: int
    base_emissions: Decimal
    target_year: int
    target_emissions: Decimal
    current_year: int
    current_emissions: Decimal
    progress_pct: Decimal
    on_track: bool
    required_annual_reduction: Decimal

@dataclass
class ComplianceScorecard:
    """Multi-framework compliance scorecard."""
    framework_results: Dict[ReportingFramework, ComplianceResult]
    overall_status: ComplianceStatus
    violations: List[ComplianceViolation]
    future_readiness: Optional[FutureReadinessReport]

@dataclass
class ComplianceResult:
    """Per-framework compliance result."""
    framework: ReportingFramework
    status: ComplianceStatus
    checks_passed: int
    checks_total: int
    violations: List[ComplianceViolation]
    score: Decimal  # 0.0 to 1.0

@dataclass
class ComplianceViolation:
    """Individual compliance violation."""
    framework: ReportingFramework
    requirement: str
    description: str
    severity: FlagSeverity
    remediation: str
    reference: str  # e.g., "ESRS E1-6 para 49(b)"

@dataclass
class FutureReadinessReport:
    """Assessment against proposed Scope 2 revisions (2025-2027)."""
    hourly_matching_ready: bool
    deliverability_compliant: bool
    exempt_from_hourly: bool
    instrument_age_compliant: bool
    issues: List[str]
    recommendations: List[str]

@dataclass
class ReconciliationRequest:
    """Input request for the reconciliation pipeline."""
    organization_id: str
    reporting_period_start: date
    reporting_period_end: date
    frameworks: List[ReportingFramework]
    include_trends: bool
    trend_periods: Optional[int]  # Number of historical periods
    targets: Optional[List[TargetTrajectory]]
    pipeline_config: Optional[PipelineConfig]

@dataclass
class ReconciliationReport:
    """Final comprehensive reconciliation report."""
    report_id: str
    request: ReconciliationRequest
    workspace: ReconciliationWorkspace
    discrepancy_report: DiscrepancyReport
    quality_assessment: QualityAssessment
    flags: List[ReconciliationFlag]
    reporting_tables: Dict[ReportingFramework, FrameworkDisclosure]
    trend_report: Optional[TrendReport]
    compliance_scorecard: ComplianceScorecard
    executive_summary: str
    generated_at: datetime
    pipeline_duration_ms: int
```

### 10.3 Constants

```python
# --- Materiality Thresholds ---
MATERIALITY_THRESHOLDS = {
    MaterialityLevel.IMMATERIAL: Decimal("0.05"),      # < 5%
    MaterialityLevel.MINOR: Decimal("0.15"),           # 5-15%
    MaterialityLevel.MATERIAL: Decimal("0.50"),        # 15-50%
    MaterialityLevel.SIGNIFICANT: Decimal("1.00"),     # 50-100%
    MaterialityLevel.EXTREME: Decimal("999.99"),       # > 100%
}

# --- Quality Score Weights ---
QUALITY_WEIGHTS = {
    QualityDimension.COMPLETENESS: Decimal("0.30"),
    QualityDimension.CONSISTENCY: Decimal("0.25"),
    QualityDimension.ACCURACY: Decimal("0.25"),
    QualityDimension.TRANSPARENCY: Decimal("0.20"),
}

# --- Quality Grade Thresholds ---
QUALITY_GRADE_THRESHOLDS = {
    QualityGrade.A: Decimal("0.90"),
    QualityGrade.B: Decimal("0.80"),
    QualityGrade.C: Decimal("0.70"),
    QualityGrade.D: Decimal("0.60"),
    QualityGrade.F: Decimal("0.00"),
}

# --- Data Source Accuracy Scores ---
ACTIVITY_DATA_ACCURACY = {
    "calibrated_meter": Decimal("1.00"),
    "utility_invoice": Decimal("0.90"),
    "estimated_benchmark": Decimal("0.60"),
    "extrapolated_partial": Decimal("0.70"),
    "supplier_report": Decimal("0.85"),
}

EF_SOURCE_ACCURACY = {
    "supplier_audited": Decimal("0.95"),
    "supplier_unaudited": Decimal("0.80"),
    "subregional_egrid": Decimal("0.90"),
    "national_iea": Decimal("0.85"),
    "default_ipcc": Decimal("0.70"),
    "default_defra": Decimal("0.75"),
    "residual_mix_aib": Decimal("0.90"),
    "residual_mix_estimated": Decimal("0.70"),
    "custom_user": Decimal("0.60"),
}

# --- Residual Mix Factors (kgCO2e/kWh) ---
RESIDUAL_MIX_FACTORS = {
    "EU_AVG": Decimal("0.380"),
    "DE": Decimal("0.520"),
    "FR": Decimal("0.085"),
    "GB": Decimal("0.285"),
    "ES": Decimal("0.290"),
    "IT": Decimal("0.420"),
    "NL": Decimal("0.450"),
    "SE": Decimal("0.045"),
    "NO": Decimal("0.360"),
    "PL": Decimal("0.720"),
    "FI": Decimal("0.150"),
    "DK": Decimal("0.280"),
    "AT": Decimal("0.180"),
    "BE": Decimal("0.310"),
    "PT": Decimal("0.220"),
    "IE": Decimal("0.380"),
    "CZ": Decimal("0.550"),
    "US_AVG": Decimal("0.425"),
    "US_CAMX": Decimal("0.285"),
    "US_ERCT": Decimal("0.420"),
    "US_MROE": Decimal("0.520"),
    "US_SRSO": Decimal("0.440"),
    "US_RFCW": Decimal("0.490"),
    "US_NEWE": Decimal("0.310"),
    "US_NYLI": Decimal("0.350"),
    "US_NYCW": Decimal("0.280"),
    "AU": Decimal("0.750"),
    "JP": Decimal("0.520"),
    "SG": Decimal("0.425"),
    "KR": Decimal("0.510"),
    "GLOBAL": Decimal("0.500"),
}

# --- Grid Average Factors (kgCO2e/kWh) for comparison ---
GRID_AVERAGE_FACTORS = {
    "EU_AVG": Decimal("0.296"),
    "DE": Decimal("0.350"),
    "FR": Decimal("0.052"),
    "GB": Decimal("0.207"),
    "ES": Decimal("0.187"),
    "IT": Decimal("0.315"),
    "NL": Decimal("0.336"),
    "SE": Decimal("0.013"),
    "NO": Decimal("0.008"),
    "PL": Decimal("0.635"),
    "US_AVG": Decimal("0.386"),
    "US_CAMX": Decimal("0.225"),
    "US_ERCT": Decimal("0.380"),
    "US_MROE": Decimal("0.482"),
    "AU": Decimal("0.656"),
    "JP": Decimal("0.465"),
    "SG": Decimal("0.408"),
    "KR": Decimal("0.459"),
    "GLOBAL": Decimal("0.436"),
}

# --- Framework Required Disclosures ---
GHGP_REQUIRED_DISCLOSURES = [
    "location_based_total",
    "market_based_total",
    "activity_data",
    "emission_factors",
    "instrument_categories",
    "coverage_percentage",
    "quality_criteria_met",
    "gases_included",
    "gwp_source",
    "base_year",
    "biogenic_co2",
    "exclusions",
]

CSRD_E1_6_DATAPOINTS = [
    "E1-6_para49a_location_based",
    "E1-6_para49b_market_based",
    "E1-6_para52_total_ghg_location",
    "E1-6_para52_total_ghg_market",
    "E1-6_18_instrument_percentage",
    "E1-6_19_instrument_types",
    "E1-6_20_bundled_instrument_pct",
    "E1-6_21_generation_attributes_pct",
    "E1-6_22_unbundled_eac_pct",
    "E1-6_24_biogenic_co2",
]

CDP_C63_COLUMNS = [
    "reporting_year",
    "scope2_location_tco2e",
    "scope2_market_tco2e",
]

CDP_C64_COLUMNS = [
    "instrument_type",
    "mwh_covered",
    "emission_factor",
    "tracking_standard",
    "country",
]

SBTI_PROGRESS_FIELDS = [
    "target_method",
    "base_year",
    "base_emissions_tco2e",
    "target_year",
    "target_emissions_tco2e",
    "current_year",
    "current_emissions_loc_tco2e",
    "current_emissions_mkt_tco2e",
    "progress_pct",
    "on_track",
    "re100_coverage_pct",
]

GRI_305_2_DISCLOSURES = [
    "305-2-a_location_based",
    "305-2-b_market_based",
    "305-2-c_gases_included",
    "305-2-d_base_year",
    "305-2-e_ef_sources_gwp",
    "305-2-f_consolidation_approach",
    "305-2-g_methodology",
]

# --- Instrument Hierarchy Priorities ---
EF_HIERARCHY_PRIORITIES = {
    EFHierarchyPriority.SUPPLIER_WITH_CERTS: 1,
    EFHierarchyPriority.SUPPLIER_NO_CERTS: 2,
    EFHierarchyPriority.BUNDLED_CERTIFICATES: 3,
    EFHierarchyPriority.UNBUNDLED_CERTIFICATES: 4,
    EFHierarchyPriority.RESIDUAL_MIX: 5,
    EFHierarchyPriority.GRID_AVERAGE_FALLBACK: 6,
}

# --- Scope 2 Quality Criteria Weights ---
QUALITY_CRITERIA_WEIGHTS = {
    Scope2QualityCriterion.CONVEY_GHG_RATE: Decimal("0.15"),
    Scope2QualityCriterion.UNIQUE_CLAIM: Decimal("0.20"),
    Scope2QualityCriterion.TRACKED_RETIRED: Decimal("0.15"),
    Scope2QualityCriterion.TEMPORAL_MATCH: Decimal("0.10"),
    Scope2QualityCriterion.GEOGRAPHIC_MATCH: Decimal("0.15"),
    Scope2QualityCriterion.SUPPLIER_SPECIFIC: Decimal("0.10"),
    Scope2QualityCriterion.DIRECT_CONTRACT: Decimal("0.10"),
    Scope2QualityCriterion.RESIDUAL_MIX_AVAILABLE: Decimal("0.05"),
}

# --- Trend Direction Thresholds ---
TREND_DIRECTION_THRESHOLDS = {
    TrendDirection.DECREASING_FAST: Decimal("-0.10"),
    TrendDirection.DECREASING: Decimal("-0.02"),
    TrendDirection.STABLE: Decimal("0.02"),
    TrendDirection.INCREASING: Decimal("0.10"),
    TrendDirection.INCREASING_FAST: Decimal("999.99"),
}

# --- GWP Values (reused from upstream) ---
GWP_VALUES = {
    GWPSource.AR4: {"co2": Decimal("1"), "ch4": Decimal("25"), "n2o": Decimal("298")},
    GWPSource.AR5: {"co2": Decimal("1"), "ch4": Decimal("28"), "n2o": Decimal("265")},
    GWPSource.AR6: {"co2": Decimal("1"), "ch4": Decimal("27.9"), "n2o": Decimal("273")},
    GWPSource.AR6_20YR: {"co2": Decimal("1"), "ch4": Decimal("81.2"), "n2o": Decimal("273")},
}

# --- Flag Severity Mapping ---
FLAG_SEVERITY_MAP = {
    FlagType.MISSING_LOCATION: FlagSeverity.CRITICAL,
    FlagType.MISSING_MARKET: FlagSeverity.CRITICAL,
    FlagType.ACTIVITY_DATA_MISMATCH: FlagSeverity.HIGH,
    FlagType.GWP_MISMATCH: FlagSeverity.HIGH,
    FlagType.LARGE_DISCREPANCY: FlagSeverity.HIGH,
    FlagType.MARKET_ABOVE_LOCATION: FlagSeverity.MEDIUM,
    FlagType.INSTRUMENT_QUALITY_FAIL: FlagSeverity.HIGH,
    FlagType.TEMPORAL_MISMATCH: FlagSeverity.MEDIUM,
    FlagType.GEOGRAPHIC_MISMATCH: FlagSeverity.MEDIUM,
    FlagType.COVERAGE_GAP: FlagSeverity.MEDIUM,
    FlagType.MISSING_RESIDUAL_MIX: FlagSeverity.HIGH,
    FlagType.FACILITY_MISSING: FlagSeverity.CRITICAL,
    FlagType.BASE_YEAR_INCONSISTENT: FlagSeverity.MEDIUM,
    FlagType.BIOGENIC_MISSING: FlagSeverity.LOW,
    FlagType.EF_OUTDATED: FlagSeverity.MEDIUM,
    FlagType.DOUBLE_COUNTING: FlagSeverity.CRITICAL,
    FlagType.FUTURE_NON_COMPLIANT: FlagSeverity.INFO,
}

# --- RE100 Targets ---
RE100_TARGETS = {
    2025: Decimal("0.80"),  # 80% renewable by 2025
    2030: Decimal("1.00"),  # 100% renewable by 2030
    2040: Decimal("1.00"),
    2050: Decimal("1.00"),
}

# --- Agent Configuration ---
AGENT_CONFIG = {
    "agent_id": "GL-MRV-X-024",
    "internal_label": "AGENT-MRV-013",
    "category": "Layer 3 - MRV / Accounting Agents (Scope 2)",
    "package": "greenlang/dual_reporting_reconciliation/",
    "db_migration": "V064",
    "metrics_prefix": "gl_drr_",
    "table_prefix": "drr_",
    "api_prefix": "/api/v1/dual-reporting-reconciliation",
    "upstream_agents": ["MRV-009", "MRV-010", "MRV-011", "MRV-012"],
}
```

### 10.4 Database Schema Outline (V064)

**Tables (12):**
1. `drr_reconciliation_runs` -- Reconciliation run metadata
2. `drr_energy_register` -- Energy purchase register linking both methods
3. `drr_discrepancies` -- Discrepancy records at all levels
4. `drr_quality_assessments` -- Quality scores per dimension
5. `drr_flags` -- Reconciliation flags with severity
6. `drr_framework_disclosures` -- Generated framework-specific outputs
7. `drr_trend_series` -- Historical trend data points
8. `drr_target_trajectories` -- SBTi/custom target tracking
9. `drr_compliance_results` -- Per-framework compliance results
10. `drr_compliance_violations` -- Individual violations
11. `drr_instrument_quality_checks` -- Per-instrument quality criteria results
12. `drr_audit_entries` -- Provenance and audit trail

**Hypertables (3):**
- `drr_reconciliation_events` -- Time-series reconciliation events
- `drr_quality_events` -- Quality score change events
- `drr_compliance_events` -- Compliance check events

**Continuous Aggregates (2):**
- `drr_daily_reconciliation_stats` -- Daily reconciliation statistics
- `drr_monthly_quality_trends` -- Monthly quality score trends

### 10.5 REST API Outline (20 Endpoints)

Prefix: `/api/v1/dual-reporting-reconciliation`

| Method | Path | Permission | Description |
|--------|------|-----------|-------------|
| POST | /reconcile | drr:reconcile | Run full reconciliation pipeline |
| POST | /reconcile/partial | drr:reconcile | Run partial (select engines) |
| GET | /reconciliations | drr:read | List reconciliation runs |
| GET | /reconciliations/{id} | drr:read | Get reconciliation report |
| DELETE | /reconciliations/{id} | drr:delete | Delete run |
| GET | /reconciliations/{id}/discrepancies | drr:read | Get discrepancy report |
| GET | /reconciliations/{id}/quality | drr:read | Get quality assessment |
| GET | /reconciliations/{id}/flags | drr:read | Get flags |
| GET | /reconciliations/{id}/tables/{framework} | drr:read | Get framework table |
| GET | /reconciliations/{id}/tables/{framework}/export | drr:export | Export (JSON/CSV/Excel) |
| GET | /reconciliations/{id}/trends | drr:read | Get trend analysis |
| GET | /reconciliations/{id}/compliance | drr:read | Get compliance scorecard |
| GET | /reconciliations/{id}/compliance/{framework} | drr:read | Per-framework compliance |
| GET | /reconciliations/{id}/summary | drr:read | Executive summary |
| GET | /trends/{org_id} | drr:read | Multi-year trend for organization |
| GET | /targets/{org_id} | drr:read | Target trajectory for organization |
| PUT | /targets/{org_id} | drr:write | Update target configuration |
| GET | /quality/history/{org_id} | drr:read | Quality score history |
| GET | /health | drr:health | Health check |
| GET | /metrics | drr:metrics | Prometheus metrics |

### 10.6 Metrics (Prometheus)

Prefix: `gl_drr_`

| Metric | Type | Description |
|--------|------|-------------|
| `gl_drr_reconciliations_total` | Counter | Total reconciliation runs |
| `gl_drr_reconciliation_duration_seconds` | Histogram | Pipeline duration |
| `gl_drr_discrepancy_absolute_tco2e` | Gauge | Current absolute discrepancy |
| `gl_drr_discrepancy_relative_pct` | Gauge | Current relative discrepancy % |
| `gl_drr_quality_composite_score` | Gauge | Current composite quality score |
| `gl_drr_quality_completeness_score` | Gauge | Current completeness score |
| `gl_drr_quality_consistency_score` | Gauge | Current consistency score |
| `gl_drr_quality_accuracy_score` | Gauge | Current accuracy score |
| `gl_drr_quality_transparency_score` | Gauge | Current transparency score |
| `gl_drr_flags_total` | Counter | Total flags by severity |
| `gl_drr_compliance_score` | Gauge | Per-framework compliance score |
| `gl_drr_re100_coverage_pct` | Gauge | RE100 coverage percentage |
| `gl_drr_location_emissions_tco2e` | Gauge | Total location-based emissions |
| `gl_drr_market_emissions_tco2e` | Gauge | Total market-based emissions |
| `gl_drr_coverage_pct` | Gauge | Instrument coverage percentage |
| `gl_drr_pipeline_stage_duration_seconds` | Histogram | Per-stage duration |
| `gl_drr_upstream_collection_errors` | Counter | Upstream collection failures |

---

## 11. Key Research Sources

This research document was compiled from the following authoritative sources:

1. GHG Protocol Scope 2 Guidance (2015) -- Primary standard for Scope 2 dual reporting
   - https://ghgprotocol.org/sites/default/files/2023-03/Scope%202%20Guidance.pdf
2. GHG Protocol Scope 2 FAQ
   - https://ghgprotocol.org/scope-2-frequently-asked-questions
3. GHG Protocol Scope 2 Public Consultation (2025)
   - https://ghgprotocol.org/blog/upcoming-scope-2-public-consultation-overview-revisions
4. GHG Protocol Scope 2 Hourly Matching and Deliverability
   - https://ghgprotocol.org/blog/upcoming-scope-2-public-consultation-hourly-matching-and-deliverability
5. Deloitte DART -- GHG Protocol Scope 2 Reporting
   - https://dart.deloitte.com/USDART/home/publications/deloitte/additional-deloitte-guidance/greenhouse-gas-protocol-reporting-considerations/chapter-5-ghg-protocol-scope-2/5-7-reporting
6. ESRS E1-6 Disclosure Requirements
   - https://sosesg.com/en/obbligo-dinformativa/e1-6-gross-scopes-1-2-3-total-ghg-emissions
7. EFRAG ESRS E1 Climate Change Standard
   - https://www.efrag.org/sites/default/files/media/document/2024-08/ESRS%20E1%20Delegated-act-2023-5303-annex-1_en.pdf
8. CDP Scope 2 Accounting Guidance
   - https://cdn.cdp.net/cdp-production/cms/guidance_docs/pdfs/000/000/415/original/CDP-Accounting-of-Scope-2-Emissions.pdf
9. SBTi Corporate Near-Term Criteria v5.3
   - https://files.sciencebasedtargets.org/production/files/SBTi-criteria.pdf
10. SBTi -- Addressing Scope 2 Reporting Challenges
    - https://sciencebasedtargets.org/blog/addressing-the-challenges-of-scope-2-emissions-reporting
11. GRI 305: Emissions 2016
    - https://www.globalreporting.org/publications/documents/english/gri-305-emissions-2016/
12. RE100 Technical Criteria (2025)
    - https://www.there100.org/sites/re100/files/2025-04/RE100%20technical%20criteria%20+%20appendices%20(15%20April%202025).pdf
13. RE100 Technical FAQs
    - https://www.there100.org/sites/re100/files/2025-10/RE100%20FAQs%20-%20Aug%202025.pdf
14. AIB European Residual Mix -- Association of Issuing Bodies
    - https://www.aib-net.org/
15. EPA eGRID -- US Environmental Protection Agency
    - https://www.epa.gov/egrid
16. Persefoni -- Scope 2 Dual Reporting Explained
    - https://www.persefoni.com/blog/scope-2-dual-reporting-market-and-location-based-carbon-accounting
17. "Almost 10 years of dual reporting of Scope 2: chaos or comparability?"
    - https://www.tandfonline.com/doi/full/10.1080/17583004.2025.2459920
18. Plana.earth -- Navigating Dual Reporting
    - https://plana.earth/academy/dual-reporting
19. Ecohz -- Mastering Scope 2 Emissions
    - https://www.ecohz.com/blog/master-scope2-emissions

---

## 12. Conclusion and Recommendations

### 12.1 Agent Justification

A dedicated Dual Reporting Reconciliation Agent (MRV-013) is strongly
justified because:

1. **Regulatory mandate**: GHG Protocol, CSRD, CDP, and GRI all require
   dual reporting with specific disclosure items that span multiple
   upstream calculation outputs.

2. **Cross-agent coordination**: No single upstream agent (009, 010, 011,
   012) has visibility into the other agents' outputs. Only a dedicated
   reconciliation agent can ensure completeness and consistency across all
   four.

3. **Quality assurance**: The 4-dimension quality scoring framework
   (completeness, consistency, accuracy, transparency) requires a holistic
   view that individual agents cannot provide.

4. **Multi-framework output**: Each regulatory framework has its own
   specific table format, datapoint requirements, and compliance checks.
   A centralized table generator avoids duplicating this logic.

5. **Trend analysis**: Year-over-year tracking, SBTi target progress, and
   RE100 coverage monitoring require persistent historical state that
   spans reporting periods.

6. **Future readiness**: The proposed 2025-2027 Scope 2 revisions
   (hourly matching, deliverability) will make automated reconciliation
   essential for compliance.

### 12.2 Architecture Recommendations

1. **Follow the 7-engine pattern** consistent with all existing MRV agents.
2. **Use the `drr_` table prefix** and `gl_drr_` metrics prefix.
3. **Database migration V064** following MRV-012's V063.
4. **Package location**: `greenlang/dual_reporting_reconciliation/`
5. **API prefix**: `/api/v1/dual-reporting-reconciliation`
6. **Target**: ~35-40 files, ~25-30K lines, 900+ tests
7. **Integration**: Register in auth_setup.py PERMISSION_MAP with
   `dual-reporting-reconciliation` permission prefix.

### 12.3 Implementation Priority

This agent should be implemented AFTER MRV-009, 010, 011, and 012 are
complete, as it depends on their outputs. However, the data models and
enums can be defined in parallel to establish the contract between agents.
