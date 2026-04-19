# PRD: AGENT-MRV-018 -- Scope 3 Category 5 Waste Generated in Operations Agent

## 1. Overview

| Field | Value |
|-------|-------|
| **Agent ID** | GL-MRV-S3-005 |
| **Internal Label** | AGENT-MRV-018 |
| **Category** | Layer 3 -- MRV / Accounting Agents (Scope 3) |
| **Package** | `greenlang/waste_generated/` |
| **DB Migration** | V069 |
| **Metrics Prefix** | `gl_wg_` |
| **Table Prefix** | `gl_wg_` |
| **API** | `/api/v1/waste-generated` |
| **Env Prefix** | `GL_WG_` |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |
| **Status** | In Development |

### Purpose

The Waste Generated in Operations Agent implements **GHG Protocol
Scope 3 Category 5: Waste Generated in Operations**, which covers
emissions from third-party disposal and treatment of waste generated
in the reporting company's owned or controlled operations during the
reporting year. Only waste treatment by THIRD PARTIES is in scope;
on-site waste treatment at the reporting company's own facilities is
reported under Scope 1 (direct combustion, process emissions) and
Scope 2 (purchased electricity for on-site treatment equipment).

Category 5 encompasses five distinct waste treatment pathways:

- **Landfill disposal** -- Waste deposited in engineered landfill
  sites where organic matter decomposes anaerobically, generating
  methane (CH4) over decades via the IPCC First Order Decay model.
  Includes managed landfills (with and without gas capture), unmanaged
  disposal sites, and semi-aerobic landfills.
- **Incineration and waste-to-energy** -- Thermal treatment of waste
  producing CO2 from fossil carbon content, plus trace CH4 and N2O.
  Includes mass-burn incineration (with and without energy recovery),
  fluidized bed combustion, and pyrolysis/gasification.
- **Recycling** -- Mechanical and chemical reprocessing of waste
  materials. Only transport-to-facility and sorting/processing
  emissions are in scope (cut-off approach). Avoided emissions from
  displacing virgin material are reported separately as memo items
  and are NOT deducted from Category 5 totals.
- **Composting and anaerobic digestion** -- Biological treatment of
  organic waste producing CH4 and N2O from aerobic decomposition
  (composting) or controlled anaerobic digestion with biogas capture
  and fugitive methane leakage.
- **Wastewater treatment** -- Treatment of liquid effluent from the
  reporting company's operations at third-party municipal or
  industrial wastewater treatment plants, generating CH4 from
  anaerobic degradation of organic matter and N2O from nitrification/
  denitrification of nitrogen compounds.

Category 5 is material for manufacturing, food and beverage, retail,
healthcare, construction, and hospitality companies. It typically
represents 1-5% of total Scope 3 emissions, with higher proportions
in industries generating significant organic waste (food processing
8-15%, healthcare 3-8%, construction and demolition 5-12%). The agent
automates the historically manual process of collecting waste audit
data, classifying waste streams by type and treatment method, applying
IPCC/EPA/DEFRA emission factors, handling multi-year landfill decay
projections, tracking diversion rates, and ensuring compliance with
the EU Waste Framework Directive, ESRS E5 circular economy
disclosures, and zero double-counting against Category 1 (cradle-to-gate)
and Category 12 (end-of-life treatment of sold products).

### Justification for Dedicated Agent

1. **Five distinct treatment pathways** -- Landfill, incineration,
   recycling, composting/anaerobic digestion, and wastewater treatment
   each require fundamentally different emission models, parameters,
   and regulatory references
2. **IPCC First Order Decay model** -- Landfill emissions follow a
   multi-year exponential decay curve requiring climate-zone-specific
   decay rate constants, DOC values per waste type, methane correction
   factors by landfill management type, and gas capture efficiency
   modeling -- a distinct calculation engine
3. **Waste classification complexity** -- 61 EPA WARM material types,
   20 European Waste Catalogue chapters, Basel Convention hazard
   classifications, and multiple regional taxonomies require a
   dedicated waste classification database engine
4. **Biogenic vs fossil carbon separation** -- Incineration emissions
   must separate fossil-origin CO2 (reported in Category 5) from
   biogenic-origin CO2 (reported as memo item only), requiring
   per-waste-type carbon content and fossil carbon fraction parameters
5. **Circular economy reporting** -- CSRD/ESRS E5 requires detailed
   circular economy disclosures (waste generation, diversion rates,
   recycling rates, hazardous waste management) that go beyond simple
   GHG accounting
6. **Wastewater complexity** -- Industry-specific organic loading
   rates (BOD/COD), treatment system MCF values, and effluent nitrogen
   calculations require a dedicated wastewater emissions engine
7. **Double-counting prevention** -- Waste emissions must not overlap
   with Category 1 (if supplier cradle-to-gate EF includes end-of-life
   of packaging), Category 12 (end-of-life of sold products), or
   Scope 1 (on-site waste treatment)
8. **Regulatory urgency** -- CSRD (FY2025+), EU Waste Framework
   Directive, EPA 40 CFR Part 98 Subpart HH/TT, SBTi, and CDP all
   require or strongly encourage Category 5 reporting with treatment-
   method-level granularity

### Standards & References

- GHG Protocol Corporate Value Chain (Scope 3) Standard (2011) -- Chapter 5
- GHG Protocol Scope 3 Technical Guidance (2013) -- Chapter 5: Category 5
- GHG Protocol Scope 3 Calculation Guidance (online)
- GHG Protocol Quantitative Uncertainty Guidance
- IPCC 2006 Guidelines for National GHG Inventories -- Volume 5: Waste
  (Chapters 2-6)
- IPCC 2019 Refinement to the 2006 Guidelines -- Volume 5: Waste
- EPA WARM v16 (Waste Reduction Model) -- 61 material types x 6 disposal
  methods, MTCO2e/short ton
- DEFRA/DESNZ Greenhouse Gas Reporting Conversion Factors (annual) --
  Waste disposal tables (landfill, incineration, recycling, composting)
- European Waste Catalogue (EWC) -- Commission Decision 2000/532/EC,
  20 chapters, 839 six-digit codes
- Basel Convention on the Control of Transboundary Movements of Hazardous
  Wastes -- Annex I hazard classes (H1-H13)
- EU Waste Framework Directive 2008/98/EC (as amended by 2018/851/EU) --
  Waste hierarchy, recycling targets, by-product criteria
- CSRD/ESRS E1 -- Scope 3 disclosure (E1-6 para 44)
- CSRD/ESRS E5 -- Resource use and circular economy (E5-1 through E5-6)
- EPA 40 CFR Part 98 Subpart HH -- Municipal Solid Waste Landfills
- EPA 40 CFR Part 98 Subpart TT -- Industrial Waste Landfills
- California SB 253 -- Mandatory Scope 3 by FY2027 for entities >$1B revenue
- CDP Climate Change Questionnaire -- C6.5 Scope 3 Category 5
- CDP Water Security Questionnaire -- W1.2 wastewater discharge
- SBTi Corporate Manual v5.3 -- Scope 3 target required if >40% of total
- GRI 306 -- Waste (2020 revision): waste generation, diversion, disposal
- ISO 14064-1:2018 -- Category 4 indirect GHG emissions (waste in operations)
- ISO 14001:2015 -- Environmental management (waste aspects)

### Terminology

| Term | Definition | Scope Mapping |
|------|-----------|---------------|
| **MSW** | Municipal Solid Waste -- mixed household and commercial waste | Common waste stream type |
| **C&D** | Construction and Demolition waste | Waste category |
| **DOC** | Degradable Organic Carbon (fraction of waste that is organic carbon) | Landfill FOD model parameter |
| **DOCf** | Fraction of DOC that decomposes under anaerobic conditions | Landfill FOD model parameter |
| **MCF** | Methane Correction Factor (0.0-1.0 by landfill management type) | Landfill/wastewater parameter |
| **FOD** | First Order Decay model for landfill methane generation | IPCC landfill methodology |
| **k** | Decay rate constant (yr-1) by climate zone and waste type | FOD model parameter |
| **DDOCm** | Mass of decomposable DOC deposited in landfill | FOD intermediate calculation |
| **R** | Recovered methane (flared or used for energy) | Landfill gas capture |
| **OX** | Oxidation factor (fraction of CH4 oxidized in landfill cover soil) | Landfill surface parameter |
| **F** | Fraction of CH4 in landfill gas (default 0.50) | Landfill gas composition |
| **dm** | Dry matter content of waste (fraction) | Incineration parameter |
| **CF** | Carbon content of dry matter (fraction) | Incineration parameter |
| **FCF** | Fossil Carbon Fraction (fraction of carbon that is fossil origin) | Incineration biogenic/fossil split |
| **OF** | Oxidation Factor for incineration (fraction of carbon oxidized) | Incineration parameter |
| **WtE** | Waste-to-Energy (incineration with energy recovery) | Treatment method type |
| **MRF** | Materials Recovery Facility (sorting facility for recyclables) | Recycling infrastructure |
| **BOD** | Biochemical Oxygen Demand (mg/L) | Wastewater organic load measure |
| **COD** | Chemical Oxygen Demand (mg/L) | Wastewater organic load measure |
| **TOW** | Total Organic load in Wastewater (kg COD/yr) | Wastewater CH4 calculation input |
| **Bo** | Maximum CH4 producing capacity (kg CH4/kg COD) | Wastewater CH4 parameter |
| **EWC** | European Waste Catalogue (6-digit classification codes) | EU waste classification system |
| **WARM** | Waste Reduction Model (EPA lifecycle waste emission factors) | US waste emission factor source |
| **Diversion rate** | Fraction of waste diverted from landfill/incineration to recycling/composting | Circular economy metric |
| **Cut-off approach** | Recycling emissions include only collection/sorting, not avoided virgin production | GHG Protocol recycling method |

---

## 2. Methodology

### 2.1 Category 5 Boundary Definition

Category 5 covers emissions from third-party disposal and treatment of
waste generated in the reporting company's OWNED OR CONTROLLED operations.
The critical boundary questions are: **who generates the waste?** and
**who treats the waste?**

**GHG Protocol boundary rule:**

```
IF waste is generated by the reporting company's operations
   AND treated by a THIRD PARTY (off-site):
  --> Category 5 (Waste Generated in Operations)

IF waste is generated by the reporting company's operations
   AND treated ON-SITE by the reporting company:
  --> Scope 1 (direct emissions from treatment)
  --> Scope 2 (electricity for treatment equipment)

IF waste is generated by customers using the reporting company's
   SOLD PRODUCTS at end-of-life:
  --> Category 12 (End-of-Life Treatment of Sold Products)

IF waste disposal emissions are already included in the cradle-to-gate
   EF of purchased goods/services:
  --> Category 1 (already counted)
```

### 2.2 Four Calculation Methods

The GHG Protocol Technical Guidance defines four methods for Category 5,
listed from most to least accurate:

| Rank | Method | Data Required | Accuracy | Coverage |
|------|--------|--------------|----------|----------|
| 1 | Supplier-Specific | Waste contractor-reported emissions per waste stream | Highest (+/-5-20%) | Lowest (key contractors) |
| 2 | Waste-Type-Specific | Mass by waste type x treatment method x type-specific EF | High (+/-10-30%) | Medium-High |
| 3 | Average-Data | Total waste mass x average EF per treatment method | Medium (+/-30-50%) | High |
| 4 | Spend-Based | Waste management spend x EEIO factors | Lowest (+/-50-100%) | Highest (all spend) |

**Decision tree for method selection:**

```
Is waste contractor-reported emission data available?
  YES --> Use Supplier-Specific Method
  NO  --> Is mass by waste type AND treatment method known?
            YES --> Use Waste-Type-Specific Method
            NO  --> Is total waste mass AND treatment method known?
                      YES --> Use Average-Data Method
                      NO  --> Is waste management spend data available?
                                YES --> Use Spend-Based Method
                                NO  --> Estimate using industry benchmarks
```

### 2.3 Waste-Type-Specific Method (Primary)

The waste-type-specific method is the most commonly used approach for
Category 5. It multiplies the mass of each waste type by its treatment-
method-specific emission factor.

**Core formula:**

```
Emissions_waste = SUM_i( Mass_i * EF_i )

Where:
  Mass_i = mass of waste type i sent to treatment method j (tonnes)
  EF_i   = emission factor for waste type i and treatment method j
           (kgCO2e/tonne)
```

**With transport to treatment facility:**

```
Emissions_total = SUM_i( Mass_i * EF_treatment_i )
               + SUM_i( Mass_i * Distance_i * EF_transport )
```

Where `EF_transport` is the road freight EF for waste collection
vehicles (typically 0.10-0.15 kgCO2e/tonne-km for refuse trucks).

### 2.4 Landfill Emissions -- IPCC First Order Decay Model

Landfill is the most complex treatment pathway. Organic waste deposited
in landfills decomposes anaerobically over decades, generating methane
according to the IPCC First Order Decay (FOD) model.

**Step 1: Calculate decomposable DOC mass deposited (DDOCm):**

```
DDOCm = W * DOC * DOCf * MCF

Where:
  W    = mass of waste deposited (tonnes)
  DOC  = degradable organic carbon fraction (by waste type)
  DOCf = fraction of DOC that decomposes (default 0.50)
  MCF  = methane correction factor (by landfill management type)
```

**Step 2: Calculate DDOCm remaining after each year (FOD):**

```
DDOCm_remaining(t) = DDOCm * e^(-k * t)

Where:
  k = decay rate constant (yr-1) specific to climate zone and waste type
  t = years since deposition
```

**Step 3: Calculate CH4 generated in year t:**

```
DDOCm_decomposed(t) = DDOCm_remaining(t-1) - DDOCm_remaining(t)

CH4_generated(t) = DDOCm_decomposed(t) * F * (16/12)

Where:
  F    = fraction of CH4 in landfill gas (default 0.50)
  16/12 = molecular weight ratio CH4/C
```

**Step 4: Calculate net CH4 emissions:**

```
CH4_emitted(t) = (CH4_generated(t) - R(t)) * (1 - OX)

Where:
  R(t) = recovered CH4 (captured for flaring or energy recovery)
  OX   = oxidation factor (CH4 oxidized in landfill cover soil)
```

**Step 5: Convert to CO2e:**

```
Emissions_CO2e(t) = CH4_emitted(t) * GWP_CH4
```

**Simplified single-year approach (when multi-year modeling not needed):**

```
Emissions_landfill = W * DOC * DOCf * MCF * F * (16/12)
                   * (1 - frac_recovered) * (1 - OX) * GWP_CH4
```

### 2.5 DOC Values by Waste Type

| Waste Type | DOC (fraction) | Source |
|-----------|---------------|--------|
| Food waste | 0.150 | IPCC 2006 Vol 5 Table 2.4 |
| Garden/yard waste | 0.200 | IPCC 2006 Vol 5 Table 2.4 |
| Paper/cardboard | 0.400 | IPCC 2006 Vol 5 Table 2.4 |
| Wood | 0.430 | IPCC 2006 Vol 5 Table 2.4 |
| Textiles | 0.240 | IPCC 2006 Vol 5 Table 2.4 |
| Disposable nappies/diapers | 0.240 | IPCC 2006 Vol 5 Table 2.4 |
| Rubber/leather | 0.390 | IPCC 2006 Vol 5 Table 2.4 |
| Plastics | 0.000 | IPCC 2006 (not degradable) |
| Glass | 0.000 | IPCC 2006 (not degradable) |
| Metals | 0.000 | IPCC 2006 (not degradable) |
| Other inert | 0.000 | IPCC 2006 (not degradable) |
| Mixed MSW (developed) | 0.160 | IPCC 2019 Refinement |
| Mixed MSW (developing) | 0.120 | IPCC 2019 Refinement |
| Sewage sludge | 0.050 | IPCC 2006 Vol 5 Table 2.4 |
| Construction & demolition | 0.080 | IPCC 2019 Refinement |

### 2.6 MCF Values by Landfill Type

| Landfill Type | MCF | Description | Source |
|--------------|-----|-------------|--------|
| Managed anaerobic | 1.0 | Engineered landfill, controlled placement, cover material | IPCC 2006 Vol 5 Table 3.1 |
| Managed semi-aerobic | 0.5 | Controlled placement with leachate drainage and gas venting | IPCC 2006 Vol 5 Table 3.1 |
| Unmanaged deep (>5m) | 0.8 | Uncontrolled disposal, depth >5m waste | IPCC 2006 Vol 5 Table 3.1 |
| Unmanaged shallow (<5m) | 0.4 | Uncontrolled disposal, depth <5m waste | IPCC 2006 Vol 5 Table 3.1 |
| Uncategorized | 0.6 | Default when management type unknown | IPCC 2006 Vol 5 Table 3.1 |

### 2.7 Decay Rate Constants (k) by Climate Zone and Waste Type

| Climate Zone | Food Waste | Garden Waste | Paper/Cardboard | Wood | Textiles | Other |
|-------------|-----------|-------------|----------------|------|----------|-------|
| Boreal/Temperate Dry | 0.06 | 0.03 | 0.04 | 0.02 | 0.04 | 0.05 |
| Temperate Wet | 0.185 | 0.10 | 0.06 | 0.03 | 0.06 | 0.09 |
| Tropical Dry | 0.085 | 0.05 | 0.045 | 0.025 | 0.045 | 0.065 |
| Tropical Wet | 0.40 | 0.17 | 0.07 | 0.035 | 0.07 | 0.17 |

Source: IPCC 2006 Vol 5 Table 3.3, IPCC 2019 Refinement updates.

### 2.8 Landfill Gas Capture Efficiency

| Gas Collection System | Capture Efficiency | Description | Source |
|----------------------|-------------------|-------------|--------|
| None | 0.00 | No gas collection system | Default |
| Active with operating cover | 0.75 | Active extraction with operating cover | EPA AP-42 |
| Active with temporary cover | 0.50 | Active extraction with temporary cover | EPA AP-42 |
| Active with clay cover | 0.65 | Active extraction with clay cap | EPA AP-42 |
| Active with geomembrane cover | 0.90 | Active extraction with synthetic cap | EPA AP-42 |
| Passive venting | 0.20 | Passive vent pipes, no active extraction | IPCC default |
| Flare only | 0.35 | Gas collected and flared, no energy recovery | Industry average |

**Oxidation factors:**

| Cover Type | OX | Source |
|-----------|-----|--------|
| No cover / bare soil | 0.00 | IPCC 2006 Vol 5 |
| Engineered soil cover | 0.10 | IPCC 2006 Vol 5 |
| Biocover / compost cover | 0.20 | IPCC 2019 Refinement |
| Geomembrane + soil | 0.10 | IPCC 2006 Vol 5 |

### 2.9 Incineration Emissions

Incineration emissions include CO2 from fossil carbon in the waste,
plus trace amounts of CH4 and N2O from incomplete combustion.

**CO2 emissions (IPCC Vol 5 Eq 5.1):**

```
CO2_fossil = SUM_j( SW_j * dm_j * CF_j * FCF_j * OF_j * 44/12 )

Where:
  SW_j  = mass of waste type j incinerated (tonnes wet weight)
  dm_j  = dry matter content (fraction of wet weight)
  CF_j  = carbon content of dry matter (fraction)
  FCF_j = fossil carbon fraction of total carbon (fraction)
  OF_j  = oxidation factor (fraction of carbon oxidized, default 1.0)
  44/12 = molecular weight ratio CO2/C
```

**CH4 emissions:**

```
CH4 = SUM_j( IW_j * EF_CH4_j )

Where:
  IW_j    = mass of waste type j incinerated (Gg)
  EF_CH4_j = CH4 emission factor by incinerator type (kg/Gg waste)
```

**N2O emissions:**

```
N2O = SUM_j( IW_j * EF_N2O_j )

Where:
  EF_N2O_j = N2O emission factor by incinerator type (kg/Gg waste)
```

**Total incineration emissions (CO2e):**

```
Emissions_incineration = CO2_fossil + (CH4 * GWP_CH4) + (N2O * GWP_N2O)
```

### 2.10 Incineration Parameters by Waste Type

| Waste Type | dm (dry matter) | CF (carbon fraction) | FCF (fossil C fraction) | OF |
|-----------|----------------|---------------------|-----------------------|-----|
| Paper/cardboard | 0.90 | 0.46 | 0.01 | 1.00 |
| Textiles (synthetic) | 0.80 | 0.46 | 1.00 | 1.00 |
| Textiles (natural) | 0.80 | 0.46 | 0.00 | 1.00 |
| Food waste | 0.40 | 0.38 | 0.00 | 1.00 |
| Wood (untreated) | 0.85 | 0.50 | 0.00 | 1.00 |
| Wood (treated) | 0.85 | 0.50 | 0.10 | 1.00 |
| Garden waste | 0.40 | 0.49 | 0.00 | 1.00 |
| Plastics (PE/PP/PS) | 1.00 | 0.75 | 1.00 | 1.00 |
| Plastics (PET) | 1.00 | 0.63 | 1.00 | 1.00 |
| Plastics (PVC) | 1.00 | 0.38 | 1.00 | 1.00 |
| Rubber | 0.84 | 0.67 | 0.80 | 1.00 |
| Leather | 0.84 | 0.46 | 0.20 | 1.00 |
| Disposable nappies | 0.90 | 0.34 | 0.10 | 1.00 |
| Electronics (WEEE) | 0.90 | 0.05 | 1.00 | 1.00 |
| Mixed MSW (developed) | 0.69 | 0.33 | 0.40 | 1.00 |
| Mixed MSW (developing) | 0.52 | 0.28 | 0.30 | 1.00 |

Source: IPCC 2006 Vol 5 Table 5.2 and IPCC 2019 Refinement.

### 2.11 Incineration CH4 and N2O Emission Factors

| Incinerator Type | CH4 EF (kg/Gg waste) | N2O EF (kg/Gg waste) | Source |
|-----------------|---------------------|---------------------|--------|
| Continuous stoker | 0.2 | 50 | IPCC 2006 Vol 5 Table 5.3 |
| Semi-continuous | 6.0 | 50 | IPCC 2006 Vol 5 Table 5.3 |
| Batch | 60.0 | 50 | IPCC 2006 Vol 5 Table 5.3 |
| Fluidized bed | 0.1 | 56 | IPCC 2019 Refinement |
| Open burning (uncontrolled) | 6500.0 | 150 | IPCC 2006 Vol 5 Table 5.3 |

### 2.12 Energy Recovery from Incineration

When waste is incinerated with energy recovery (Waste-to-Energy), the
recovered energy displaces grid electricity or district heating. This
avoided emission is reported as a memo item and is NOT deducted from
Category 5 totals under the GHG Protocol.

**Energy recovery calculation:**

```
Energy_recovered_kWh = Mass_waste * NCV * Efficiency

Where:
  NCV        = Net Calorific Value of waste (MJ/kg)
  Efficiency = Thermal/electrical conversion efficiency (0.15-0.30 typical)
```

**Avoided emissions (memo item only):**

```
Avoided_CO2e = Energy_recovered_kWh * Grid_EF_displaced
```

**Net Calorific Values by waste type:**

| Waste Type | NCV (MJ/kg wet) | Source |
|-----------|----------------|--------|
| Mixed MSW | 8.0 - 12.0 | IPCC 2006 Vol 5 |
| Paper/cardboard | 12.0 - 16.0 | IPCC 2006 |
| Plastics | 30.0 - 40.0 | IPCC 2006 |
| Food waste | 3.0 - 6.0 | IPCC 2006 |
| Wood | 14.0 - 16.0 | IPCC 2006 |
| Textiles | 15.0 - 18.0 | IPCC 2006 |
| Rubber/leather | 20.0 - 25.0 | IPCC 2006 |

### 2.13 Recycling and Composting Emissions

**Recycling (cut-off approach):**

Under the GHG Protocol's cut-off approach, recycling emissions in
Category 5 include ONLY the emissions from:
1. Transport of recyclables to the recycling facility
2. Sorting at the Materials Recovery Facility (MRF)
3. Pre-processing (baling, shredding, washing)

Avoided emissions from displacing virgin material production are
reported separately and are NOT deducted from Category 5.

```
Emissions_recycling = Mass * EF_recycling_process

Where:
  EF_recycling_process = process emissions per tonne at MRF + pre-processing
                         (typically 21-50 kgCO2e/tonne depending on material)
```

**Open-loop vs closed-loop recycling:**

| Recycling Type | Description | Emission Treatment |
|---------------|-------------|-------------------|
| Closed-loop | Material recycled back into same product type | Process emissions only |
| Open-loop | Material recycled into different, lower-grade product | Process emissions only; downcycling quality factor applied |

**Composting emissions:**

```
CH4_composting = Mass_wet * EF_CH4_composting
N2O_composting = Mass_wet * EF_N2O_composting

Emissions_composting = (CH4_composting * GWP_CH4)
                     + (N2O_composting * GWP_N2O)
```

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| EF_CH4 (composting) | 4.0 | g CH4/kg wet waste | IPCC 2006 Vol 5 Table 4.1 |
| EF_N2O (composting) | 0.30 | g N2O/kg wet waste | IPCC 2006 Vol 5 Table 4.1 |
| EF_CH4 (home composting) | 10.0 | g CH4/kg wet waste | IPCC 2019 Refinement |
| EF_N2O (home composting) | 0.60 | g N2O/kg wet waste | IPCC 2019 Refinement |

**Anaerobic digestion emissions:**

```
CH4_leakage = Biogas_produced * CH4_content * Leakage_rate

Emissions_AD = CH4_leakage * GWP_CH4
             + Digestate_emissions
```

| Plant Type | Leakage Rate (%) | Source |
|-----------|-----------------|--------|
| Enclosed, modern | 2.0 | IPCC 2019 |
| Enclosed, older | 3.5 | IPCC 2019 |
| Open digester | 5.0 | IPCC 2019 |
| Covered lagoon | 7.0 | IPCC 2019 |

### 2.14 Wastewater Treatment Emissions

**CH4 from wastewater treatment (IPCC Vol 5 Ch 6):**

```
CH4_wastewater = TOW * Bo * MCF_treatment

Where:
  TOW = total organic load in wastewater (kg COD/yr)
  Bo  = maximum CH4 producing capacity (default 0.25 kg CH4/kg COD)
  MCF = methane correction factor by treatment system
```

**TOW calculation:**

```
TOW = Volume_m3 * COD_mg_per_L * 1e-6 * 1000
    = Volume_m3 * COD_kg_per_m3
```

**N2O from wastewater effluent:**

```
N2O_effluent = N_effluent * EF_N2O_effluent * (44/28)

Where:
  N_effluent     = nitrogen in treated effluent discharged (kg N/yr)
  EF_N2O_effluent = 0.005 kg N2O-N/kg N (IPCC default)
  44/28          = molecular weight ratio N2O/N2
```

**MCF values by treatment system:**

| Treatment System | MCF | Description | Source |
|-----------------|-----|-------------|--------|
| Centralized aerobic (well managed) | 0.00 | Activated sludge, trickling filter | IPCC 2006 Vol 5 Table 6.3 |
| Centralized aerobic (not well managed) | 0.03 | Overloaded aerobic system | IPCC 2019 Refinement |
| Centralized anaerobic reactor | 0.80 | UASB, anaerobic filter | IPCC 2006 Vol 5 Table 6.3 |
| Anaerobic shallow lagoon (<2m) | 0.20 | Shallow stabilization pond | IPCC 2006 Vol 5 Table 6.3 |
| Anaerobic deep lagoon (>2m) | 0.80 | Deep anaerobic lagoon | IPCC 2006 Vol 5 Table 6.3 |
| Septic system | 0.50 | Septic tank + drain field | IPCC 2006 Vol 5 Table 6.3 |
| Untreated (river/sea discharge) | 0.10 | Direct discharge to water body | IPCC 2006 Vol 5 Table 6.3 |
| Constructed wetland | 0.05 | Engineered wetland treatment | IPCC 2019 Refinement |

**Bo values:**

| Basis | Bo Value | Unit | Source |
|-------|---------|------|--------|
| COD basis (default) | 0.25 | kg CH4/kg COD | IPCC 2006 Vol 5 |
| BOD basis | 0.60 | kg CH4/kg BOD | IPCC 2006 Vol 5 |

**Industry-specific wastewater loads:**

| Industry Type | COD (kg/m3) | BOD (kg/m3) | Wastewater Volume (m3/tonne product) | Source |
|--------------|------------|------------|-------------------------------------|--------|
| Starch production | 10.0 | 6.0 | 9.0 | IPCC 2006 Vol 5 Table 6.9 |
| Alcohol/spirits | 15.0 | 8.0 | 24.0 | IPCC 2006 Vol 5 Table 6.9 |
| Beer/malt beverages | 3.0 | 1.5 | 6.3 | IPCC 2006 Vol 5 Table 6.9 |
| Pulp & paper | 7.0 | 3.5 | 85.0 | IPCC 2006 Vol 5 Table 6.9 |
| Food processing (general) | 5.0 | 2.5 | 20.0 | IPCC 2006 Vol 5 Table 6.9 |
| Meat & poultry | 4.1 | 1.5 | 13.0 | IPCC 2006 Vol 5 Table 6.9 |
| Vegetables & fruits | 5.0 | 2.5 | 20.0 | IPCC 2006 Vol 5 Table 6.9 |
| Dairy | 2.7 | 1.5 | 7.0 | IPCC 2006 Vol 5 Table 6.9 |
| Sugar refining | 3.2 | 1.6 | 15.0 | IPCC 2006 Vol 5 Table 6.9 |
| Textile dyeing | 1.5 | 0.5 | 100.0 | IPCC 2006 Vol 5 Table 6.9 |
| Pharmaceuticals | 4.0 | 2.0 | 50.0 | Industry data |

### 2.15 EPA WARM Emission Factors (Selected)

EPA WARM v16 provides lifecycle emission factors for 61 material types
across 6 disposal methods. Factors are in MTCO2e per short ton
(converted to kgCO2e/tonne internally).

| Material | Landfill | Combustion | Recycling | Composting | AD | Source Recovery |
|----------|---------|-----------|----------|-----------|-----|---------------|
| Corrugated cardboard | 0.18 | 0.04 | -3.11 | -0.18 | -0.18 | N/A |
| Newspapers | -0.76 | -0.55 | -2.82 | -0.18 | N/A | N/A |
| Office paper | 1.17 | 0.04 | -2.85 | -0.18 | N/A | N/A |
| Mixed paper (general) | 0.55 | -0.16 | -2.87 | -0.18 | N/A | N/A |
| HDPE | 0.02 | 1.27 | -0.78 | N/A | N/A | N/A |
| LDPE | 0.02 | 1.27 | -0.89 | N/A | N/A | N/A |
| PET | 0.02 | 1.55 | -1.55 | N/A | N/A | N/A |
| Mixed plastics | 0.02 | 1.18 | -0.86 | N/A | N/A | N/A |
| Glass | 0.02 | 0.02 | -0.28 | N/A | N/A | N/A |
| Aluminum cans | 0.02 | 0.02 | -9.13 | N/A | N/A | N/A |
| Steel cans | 0.02 | -1.52 | -1.83 | N/A | N/A | N/A |
| Mixed metals | 0.02 | -0.75 | -4.49 | N/A | N/A | N/A |
| Food waste | 0.52 | 0.04 | N/A | -0.18 | -0.08 | N/A |
| Yard trimmings | -0.16 | -0.14 | N/A | -0.18 | N/A | N/A |
| Mixed MSW | 0.36 | 0.04 | N/A | N/A | N/A | N/A |
| Dimensional lumber | -0.14 | -0.41 | -2.47 | N/A | N/A | N/A |
| Wood flooring | -0.14 | -0.41 | -2.59 | N/A | N/A | N/A |
| Concrete | 0.02 | N/A | -0.01 | N/A | N/A | N/A |
| Asphalt shingles | 0.02 | 0.16 | -0.44 | N/A | N/A | N/A |
| Carpet | 0.02 | 1.09 | -2.29 | N/A | N/A | N/A |
| Tires | 0.02 | 0.42 | -0.48 | N/A | N/A | N/A |

Note: Negative values in recycling/composting columns represent avoided
emissions (lifecycle credit) and are reported as MEMO ITEMS ONLY -- they
are NOT deducted from Category 5 totals under the GHG Protocol cut-off
approach.

**Unit conversion: WARM factors are in MTCO2e/short ton.** Internal
conversion:

```
EF_kgCO2e_per_tonne = EF_MTCO2e_per_short_ton * 1000 / 0.90718
                    = EF_MTCO2e_per_short_ton * 1102.31
```

### 2.16 DEFRA/BEIS Waste Emission Factors (Selected)

DEFRA/DESNZ provides waste disposal emission factors in kgCO2e per
tonne of waste by treatment method and waste type.

| Waste Type | Landfill (kgCO2e/t) | Incineration (kgCO2e/t) | Recycling (kgCO2e/t) | Composting (kgCO2e/t) | Source |
|-----------|--------------------|-----------------------|--------------------|--------------------|--------|
| Paper/cardboard (mixed) | 1042 | 21 | 21 | N/A | DEFRA 2025 |
| Paper/cardboard (corrugated) | 748 | 21 | 21 | N/A | DEFRA 2025 |
| Plastics (average) | 9 | 2129 | 21 | N/A | DEFRA 2025 |
| Plastics (HDPE) | 9 | 2106 | 21 | N/A | DEFRA 2025 |
| Plastics (PET) | 9 | 2153 | 21 | N/A | DEFRA 2025 |
| Glass | 9 | 9 | 21 | N/A | DEFRA 2025 |
| Metals (mixed) | 9 | 9 | 21 | N/A | DEFRA 2025 |
| Metals (aluminium) | 9 | 9 | 21 | N/A | DEFRA 2025 |
| Metals (steel) | 9 | 9 | 21 | N/A | DEFRA 2025 |
| Food waste | 586 | 21 | N/A | 116 | DEFRA 2025 |
| Garden waste | 578 | 21 | N/A | 116 | DEFRA 2025 |
| Wood | 843 | 21 | 21 | N/A | DEFRA 2025 |
| Textiles | 868 | 1413 | 21 | N/A | DEFRA 2025 |
| WEEE (electrical) | 9 | 9 | 21 | N/A | DEFRA 2025 |
| Construction & demolition | 9 | 9 | 21 | N/A | DEFRA 2025 |
| Mixed commercial waste | 466 | 445 | 21 | N/A | DEFRA 2025 |
| Mixed municipal waste | 578 | 445 | 21 | N/A | DEFRA 2025 |
| Hazardous waste (average) | 9 | 1500 | N/A | N/A | DEFRA 2025 |

### 2.17 Spend-Based Method

The spend-based method estimates emissions by multiplying waste
management spend by EEIO emission factors for waste treatment sectors.

**Core formula:**

```
Emissions_spend = Waste_spend * EEIO_factor_waste_sector
```

**With currency conversion and inflation adjustment:**

```
Emissions_spend = (Waste_spend_local / FX_rate_to_base)
                * (CPI_base_year / CPI_spend_year)
                * EEIO_factor
```

**EEIO waste sector factors:**

| NAICS Code | Sector Description | EF (kgCO2e/USD) | Source |
|------------|-------------------|-----------------|--------|
| 562111 | Solid waste collection | 0.480 | EPA USEEIO v1.2 |
| 562119 | Other waste collection | 0.420 | EPA USEEIO v1.2 |
| 562211 | Hazardous waste treatment and disposal | 0.650 | EPA USEEIO v1.2 |
| 562212 | Solid waste landfill | 0.580 | EPA USEEIO v1.2 |
| 562213 | Solid waste combustors and incinerators | 0.720 | EPA USEEIO v1.2 |
| 562219 | Other nonhazardous waste treatment | 0.380 | EPA USEEIO v1.2 |
| 562910 | Remediation services | 0.310 | EPA USEEIO v1.2 |
| 562920 | Materials recovery facilities | 0.280 | EPA USEEIO v1.2 |
| 562991 | Septic tank services | 0.350 | EPA USEEIO v1.2 |
| 562998 | Other waste management services | 0.400 | EPA USEEIO v1.2 |

**EXIOBASE waste sector factors:**

| Sector | Region | EF (kgCO2e/EUR) | Source |
|--------|--------|-----------------|--------|
| Waste collection, treatment, disposal | EU average | 0.520 | EXIOBASE 3.8 |
| Recycling of metals | EU average | 0.350 | EXIOBASE 3.8 |
| Recycling of non-metals | EU average | 0.290 | EXIOBASE 3.8 |
| Sewage and refuse disposal | EU average | 0.440 | EXIOBASE 3.8 |

### 2.18 Data Quality Indicator (DQI)

Per GHG Protocol Scope 3 Standard Chapter 7, five data quality indicators
are assessed on a 1-5 scale:

| Indicator | Score 1 (Very Good) | Score 3 (Fair) | Score 5 (Very Poor) |
|-----------|--------------------|-----------------|--------------------|
| Temporal | Data from reporting year | Data within 6 years | Data older than 10 years |
| Geographical | Same country/region | Same continent | Global average |
| Technological | Same waste type and treatment method | Related waste category | Generic waste stream |
| Completeness | All waste streams and facilities included | 50-80% of waste covered | Less than 20% covered |
| Reliability | Waste audit data or contractor invoices | Established database (DEFRA/EPA) | Estimate or assumption |

**Composite DQI:**

```
DQI_composite = (DQI_temporal + DQI_geographical + DQI_technological
                + DQI_completeness + DQI_reliability) / 5
```

**Quality classification:**

| DQI Range | Classification | Recommended Action |
|-----------|---------------|-------------------|
| 1.0 - 1.5 | Very Good | Maintain current data quality |
| 1.6 - 2.5 | Good | Monitor for improvements |
| 2.6 - 3.5 | Fair | Prioritize improvement plan |
| 3.6 - 4.5 | Poor | Active improvement required |
| 4.6 - 5.0 | Very Poor | Urgent data quality intervention |

### 2.19 Uncertainty Ranges

| Method | Typical DQI Range | Uncertainty Range | Confidence Level |
|--------|------------------|------------------|-----------------|
| Supplier-specific (contractor data) | 1.0 - 1.5 | +/- 5-15% | Very High |
| Waste-type-specific (waste audit + DEFRA) | 1.5 - 2.5 | +/- 15-35% | High |
| Waste-type-specific (transfer notes + EPA WARM) | 2.0 - 3.0 | +/- 20-50% | Medium-High |
| Average-data (total waste x avg EF) | 3.0 - 4.0 | +/- 40-70% | Low-Medium |
| Spend-based (EEIO waste sectors) | 3.5 - 4.5 | +/- 50-100% | Low |
| Landfill FOD (Tier 1 IPCC defaults) | 2.5 - 3.5 | +/- 30-60% | Medium |
| Landfill FOD (Tier 2 country-specific) | 1.5 - 2.5 | +/- 15-30% | High |
| Landfill FOD (Tier 3 facility-specific) | 1.0 - 2.0 | +/- 5-20% | Very High |

**Pedigree matrix uncertainty factors:**

| DQI Score | Uncertainty Factor (sigma) |
|-----------|--------------------------|
| 1 | 1.00 (no additional uncertainty) |
| 2 | 1.05 (+/- 5% additional) |
| 3 | 1.10 (+/- 10% additional) |
| 4 | 1.20 (+/- 20% additional) |
| 5 | 1.50 (+/- 50% additional) |

**Combined uncertainty:**

```
Sigma_combined = sqrt(sigma_base^2 + sigma_temporal^2 + sigma_geo^2
                    + sigma_tech^2 + sigma_completeness^2 + sigma_reliability^2)
```

### 2.20 Category Boundaries & Double-Counting Prevention

**Included in Category 5:**

| Sub-Activity | What Is Included | Boundary |
|-------------|-----------------|----------|
| Landfill disposal | CH4 from anaerobic decomposition of organic waste at 3rd-party landfills | Waste gate to landfill emissions over decay period |
| Incineration | CO2/CH4/N2O from thermal treatment at 3rd-party incinerators | Waste gate to stack emissions |
| Recycling | Transport + MRF sorting emissions at 3rd-party recyclers | Waste gate to sorted material output |
| Composting | CH4/N2O from aerobic decomposition at 3rd-party composting facilities | Waste gate to compost output |
| Anaerobic digestion | CH4 leakage from AD plants + digestate handling | Waste gate to biogas/digestate output |
| Wastewater treatment | CH4/N2O from 3rd-party wastewater treatment of company effluent | Effluent discharge to treated water release |

**Excluded from Category 5:**

| Exclusion | Reason | Where Reported |
|-----------|--------|---------------|
| On-site waste treatment | Company-controlled = Scope 1/2 | Scope 1 (MRV-001 to MRV-008) |
| End-of-life of sold products | Customer waste = Category 12 | Future AGENT-MRV-025 |
| Packaging waste in cradle-to-gate | Already in supplier EF | Category 1 (AGENT-MRV-014) |
| Avoided emissions from recycling | Memo item, not deducted | Separate disclosure |
| Energy recovery credit from WtE | Memo item, not deducted | Separate disclosure |

**Double-Counting Prevention Rules:**

| Rule | Scope/Category Boundary | Enforcement |
|------|------------------------|-------------|
| DOUBLE_COUNT_CAT1 | vs Category 1 (Purchased Goods) | If supplier cradle-to-gate EF INCLUDES end-of-life/disposal of packaging, do NOT add separate Cat 5 for that waste |
| DOUBLE_COUNT_CAT12 | vs Category 12 (End-of-Life Sold Products) | Company's own operational waste = Cat 5; customer end-of-life = Cat 12; same waste NEVER in both |
| DOUBLE_COUNT_SCOPE1 | vs Scope 1 (Direct Emissions) | On-site waste treatment = Scope 1, not Cat 5; facility boundary check |
| DOUBLE_COUNT_SCOPE2 | vs Scope 2 (Purchased Energy) | Electricity for on-site waste equipment = Scope 2, not Cat 5 |
| RECYCLING_CREDIT | Avoided emissions from recycling | NOT deducted from Cat 5; reported as separate memo item only |
| WTE_CREDIT | Energy recovery credit from incineration | NOT deducted from Cat 5; reported as separate memo item only |

**Cross-category validation checks:**

```
Check 1: On-site vs off-site boundary
  For each waste stream, verify treatment is by a THIRD PARTY.
  If treatment is on-site at reporting company's facility, flag as
  Scope 1/2 (not Category 5).

Check 2: Cradle-to-gate overlap
  For each purchased good, if the Category 1 emission factor is
  "cradle-to-grave" or includes end-of-life disposal, flag any
  Category 5 entries for the same waste stream as potential
  double-counting.

Check 3: Category 12 overlap
  For each waste stream, verify that the waste is generated by the
  reporting company's OPERATIONS (not by customers using sold products).
  Customer waste = Category 12.

Check 4: Recycling credit enforcement
  Verify that avoided emissions from recycling are reported as memo
  items ONLY and are not subtracted from Category 5 totals.

Check 5: WtE credit enforcement
  Verify that energy recovery credits from waste-to-energy incineration
  are reported as memo items ONLY and are not subtracted from
  Category 5 totals.
```

### 2.21 Coverage & Materiality

**Category 5 as percentage of total Scope 3 by sector:**

| Industry Sector | Cat 5 as % of Total S3 | Cat 5 as % of S1+S2+S3 | Primary Driver |
|----------------|----------------------|----------------------|----------------|
| Food & beverage manufacturing | 3 - 8% | 2 - 5% | Organic waste, packaging waste |
| Food service / hospitality | 5 - 15% | 3 - 8% | Food waste, packaging |
| Healthcare | 3 - 8% | 2 - 5% | Clinical waste, packaging |
| Retail (grocery) | 2 - 6% | 1 - 4% | Food waste, packaging |
| Construction | 5 - 12% | 3 - 8% | C&D waste |
| Manufacturing (general) | 1 - 5% | 1 - 3% | Process waste, packaging |
| Chemicals | 2 - 6% | 1 - 4% | Hazardous waste, process waste |
| Mining & metals | 2 - 8% | 1 - 5% | Process waste, tailings |
| Pharmaceuticals | 2 - 5% | 1 - 3% | Hazardous waste, packaging |
| Technology / Electronics | 1 - 3% | 0.5 - 2% | E-waste, packaging |
| Financial services | 0.5 - 2% | 0.3 - 1% | Office waste only |
| Agriculture | 3 - 10% | 2 - 6% | Organic waste, packaging |

**Coverage thresholds:**

| Level | Target | Description |
|-------|--------|-------------|
| Minimum viable | >= 80% of waste mass or spend | Required for credible reporting |
| Good practice | >= 90% of waste mass by type and treatment | Recommended by CDP/SBTi |
| Best practice | >= 95% with waste-type-specific for all material streams | Leading practice |

### 2.22 Emission Factor Selection Hierarchy

The agent implements a 6-level EF priority hierarchy for Category 5:

| Priority | Source | DQI Score | Applicability |
|----------|--------|-----------|--------------|
| 1 | Waste contractor-specific reported emissions | 1.0-1.5 | All treatment methods |
| 2 | IPCC Tier 2/3 with country-specific parameters | 1.5-2.5 | Landfill, wastewater |
| 3 | DEFRA/DESNZ waste conversion factors (current year) | 2.0-3.0 | All treatment methods |
| 4 | EPA WARM v16 (waste-type x treatment method) | 2.0-3.0 | All treatment methods |
| 5 | IPCC Tier 1 default emission factors | 2.5-3.5 | All treatment methods |
| 6 | EEIO waste sector factors (spend-based) | 3.5-5.0 | Spend-based fallback |

### 2.23 Key Formulas Summary

**Landfill (simplified single-year):**

```
Emissions_landfill = W * DOC * DOCf * MCF * F * (16/12)
                   * (1 - frac_recovered) * (1 - OX) * GWP_CH4
```

**Landfill (FOD multi-year):**

```
CH4_emitted(t) = (DDOCm * (e^(-k*(t-1)) - e^(-k*t)) * F * (16/12)
               - R(t)) * (1 - OX)
Emissions(t)   = CH4_emitted(t) * GWP_CH4
```

**Incineration:**

```
CO2_fossil = SUM_j(SW_j * dm_j * CF_j * FCF_j * OF_j * 44/12)
CH4        = SUM_j(IW_j * EF_CH4_j)
N2O        = SUM_j(IW_j * EF_N2O_j)
Total      = CO2_fossil + CH4*GWP_CH4 + N2O*GWP_N2O
```

**Recycling (cut-off):**

```
Emissions_recycling = Mass * EF_process_only
```

**Composting:**

```
Emissions_composting = Mass * (EF_CH4 * GWP_CH4 + EF_N2O * GWP_N2O)
```

**Anaerobic digestion:**

```
Emissions_AD = Biogas * CH4_content * Leakage_rate * GWP_CH4
```

**Wastewater:**

```
CH4 = TOW * Bo * MCF_treatment
N2O = N_effluent * EF_N2O * (44/28)
Total = CH4 * GWP_CH4 + N2O * GWP_N2O
```

**Spend-based:**

```
Emissions_spend = Waste_spend * (1/FX) * (CPI_base/CPI_yr) * EEIO_factor
```

**Total Category 5:**

```
Emissions_cat5 = SUM(Emissions_landfill)
               + SUM(Emissions_incineration)
               + SUM(Emissions_recycling)
               + SUM(Emissions_composting)
               + SUM(Emissions_AD)
               + SUM(Emissions_wastewater)
```

**Diversion rate:**

```
Diversion_rate = (Mass_recycled + Mass_composted + Mass_AD + Mass_reused)
               / Total_waste_generated
```

**Emissions intensity:**

```
Intensity_revenue = Emissions_cat5 / Revenue_total        (tCO2e per $M)
Intensity_mass    = Emissions_cat5 / Total_waste_generated (kgCO2e per tonne)
Intensity_product = Emissions_cat5 / Total_products_made   (kgCO2e per unit)
```

---

## 3. Architecture

### 3.1 Seven-Engine Architecture

```
+-----------------------------------------------------------+
|                    AGENT-MRV-018                           |
|         Waste Generated in Operations Agent                |
|                                                            |
|  +------------------------------------------------------+ |
|  | Engine 1: WasteClassificationDatabaseEngine           | |
|  |   - EPA WARM v16 factors (61 materials x 6 methods)  | |
|  |   - DEFRA/BEIS waste conversion factors (annual)      | |
|  |   - IPCC Vol 5 waste sector defaults                  | |
|  |   - European Waste Catalogue (EWC) 20 chapters        | |
|  |   - EPA waste categories                              | |
|  |   - Basel Convention hazard classifications (H1-H13)  | |
|  |   - MSW composition profiles by region                | |
|  |   - Waste-to-treatment method compatibility matrix    | |
|  |   - DOC, DOCf, MCF, k values per waste type/zone     | |
|  |   - Incineration parameters (dm, CF, FCF, OF)         | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 2: LandfillEmissionsEngine                     | |
|  |   - IPCC First Order Decay (FOD) model                | |
|  |   - DDOCm = W * DOC * DOCf * MCF                      | |
|  |   - Multi-year decay projection (k by zone x type)    | |
|  |   - Landfill gas capture efficiency (35-90%)           | |
|  |   - Oxidation factor (0.0 or 0.1)                     | |
|  |   - MCF by site type (5 landfill types)               | |
|  |   - Simplified single-year and full FOD modes          | |
|  |   - CH4 -> CO2e conversion (AR4/AR5/AR6)              | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 3: IncinerationEmissionsEngine                 | |
|  |   - CO2: SW * dm * CF * FCF * OF * 44/12              | |
|  |   - CH4: IW * EF_CH4 (by incinerator type)            | |
|  |   - N2O: IW * EF_N2O (by incinerator type)            | |
|  |   - Biogenic vs fossil carbon separation               | |
|  |   - Energy recovery tracking (WtE plants)             | |
|  |   - 4 incinerator types + open burning                | |
|  |   - 16+ waste type composition parameters             | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 4: RecyclingCompostingEngine                   | |
|  |   - Recycling: cut-off approach (process only)        | |
|  |   - Open-loop vs closed-loop distinction              | |
|  |   - Avoided emissions (memo item, NOT deducted)       | |
|  |   - Composting: CH4 + N2O per kg wet waste            | |
|  |   - Anaerobic digestion: CH4 leakage + digestate      | |
|  |   - Downcycling quality factor                        | |
|  |   - Diversion rate calculation                        | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 5: WastewaterEmissionsEngine                   | |
|  |   - CH4 = TOW * Bo * MCF (IPCC Vol 5 Ch 6)           | |
|  |   - N2O = N_effluent * EF * 44/28                     | |
|  |   - MCF by treatment system (8 types)                 | |
|  |   - Bo: 0.25 kg CH4/kg COD (default)                  | |
|  |   - Industry-specific wastewater loads (11 industries)| |
|  |   - Sludge management emissions                       | |
|  |   - COD/BOD basis support                             | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 6: ComplianceCheckerEngine                     | |
|  |   - 7 frameworks: GHG Protocol Scope 3, ISO 14064,   | |
|  |     CSRD/ESRS E1+E5, CDP, SBTi, EU Waste Framework   | |
|  |     Directive, EPA 40 CFR Part 98 (Subpart HH/TT)    | |
|  |   - Double-counting prevention rules:                 | |
|  |     CAT1_OVERLAP, CAT12_OVERLAP, SCOPE1_BOUNDARY,     | |
|  |     RECYCLING_CREDIT, WTE_CREDIT                      | |
|  |   - ESRS E5 circular economy disclosures (E5-1..E5-5) | |
|  |   - Waste hierarchy compliance                        | |
|  |   - Diversion rate verification                       | |
|  |   - Hazardous waste regulatory checks                 | |
|  +------------------------------------------------------+ |
|                          |                                 |
|  +------------------------------------------------------+ |
|  | Engine 7: WasteGeneratedPipelineEngine                | |
|  |   - 10-stage pipeline orchestration                   | |
|  |   - Batch multi-period processing                     | |
|  |   - Multi-facility aggregation                        | |
|  |   - Treatment method routing                          | |
|  |   - Export (JSON/CSV/Excel/PDF)                       | |
|  |   - Compliance-ready outputs (CDP, CSRD, GRI 306)     | |
|  |   - Provenance chain assembly                         | |
|  +------------------------------------------------------+ |
+-----------------------------------------------------------+
```

### 3.2 Ten-Stage Pipeline

```
Stage 1: VALIDATE
  - Schema validation of waste stream, composition, and treatment records
  - Required field checks (waste type, mass, treatment method)
  - Data type enforcement (Decimal for quantities and factors)
  - Duplicate detection (same waste stream/facility/period)
  - Unit consistency validation (tonnes/kg/lbs, wet/dry basis)
  - Waste type vs treatment method compatibility check

Stage 2: CLASSIFY
  - Waste category classification (14 categories)
  - EWC code mapping (6-digit code to category)
  - Treatment method classification (11 methods)
  - Hazard classification (Basel Convention H1-H13)
  - Waste stream type assignment (6 stream types)
  - Calculation method selection (supplier/type-specific/average/spend)
  - Climate zone assignment (for landfill k values)
  - Incinerator type classification

Stage 3: NORMALIZE
  - Unit conversion to EF reference units (tonnes, wet/dry basis)
  - Currency conversion for spend-based records
  - CPI deflation to EEIO base year
  - Mass normalization (kg to tonnes, lbs to tonnes, short tons to tonnes)
  - Wet-to-dry weight conversion where needed
  - Period normalization (monthly/quarterly/annual)
  - EPA WARM factor conversion (MTCO2e/short ton to kgCO2e/tonne)

Stage 4: RESOLVE_EFS
  - 6-level EF hierarchy resolution for each waste stream
  - Treatment-method-specific EF selection
  - Waste-type-specific parameter lookup (DOC, dm, CF, FCF, OF)
  - Climate-zone-specific k value lookup for landfill
  - MCF lookup by landfill type
  - EEIO sector EF for spend-based records
  - Missing EF flagging with fallback
  - Gas capture efficiency lookup

Stage 5: CALCULATE_TREATMENT
  - Landfill: FOD model or simplified single-year
  - Incineration: CO2_fossil + CH4 + N2O calculation
  - Recycling: Process-only emissions (cut-off approach)
  - Composting: CH4 + N2O from aerobic decomposition
  - Anaerobic digestion: CH4 leakage calculation
  - Wastewater: CH4 from organic load + N2O from effluent
  - Per-gas breakdown (CO2, CH4, N2O)
  - GWP application (AR4/AR5/AR6)
  - Biogenic CO2 separation (memo item)

Stage 6: CALCULATE_TRANSPORT
  - Transport of waste to treatment facility
  - Collection vehicle emissions (refuse truck EFs)
  - Distance estimation (facility to treatment site)
  - Weight-based transport emission allocation

Stage 7: ALLOCATE
  - Multi-facility waste allocation
  - Shared waste contractor allocation
  - Waste composition allocation for mixed streams
  - Period allocation for multi-year landfill decay

Stage 8: COMPLIANCE
  - 7-framework compliance check
  - Double-counting prevention verification (Cat 1, Cat 12, Scope 1)
  - Recycling credit enforcement (memo only)
  - WtE credit enforcement (memo only)
  - ESRS E5 circular economy disclosure check
  - Waste hierarchy compliance
  - Diversion rate verification
  - Hazardous waste regulatory checks
  - Data quality minimum thresholds
  - Gap identification and recommendations

Stage 9: AGGREGATE
  - Total Category 5 emissions
  - By treatment method (landfill, incineration, recycling, composting, AD, wastewater)
  - By waste type (14 categories)
  - By waste stream (6 stream types)
  - By facility / site
  - By waste contractor
  - By reporting period
  - Intensity metrics (per tonne waste, per $M revenue, per unit product)
  - Diversion rate analysis
  - Waste composition profile
  - Year-over-year change decomposition
  - Hot-spot analysis (top waste types, top treatment methods)

Stage 10: SEAL
  - SHA-256 provenance hash
  - Audit trail assembly
  - Export generation (JSON/CSV/Excel/PDF)
  - Result persistence
  - Provenance chain linking to waste audit source data
```

### 3.3 File Structure

```
greenlang/waste_generated/
+-- __init__.py                              # Lazy imports, module exports
+-- models.py                                # Pydantic v2 models, enums, constants
+-- config.py                                # GL_WG_ prefixed configuration
+-- metrics.py                               # Prometheus metrics (gl_wg_*)
+-- provenance.py                            # SHA-256 provenance chain
+-- waste_classification_database.py         # Engine 1: Waste EF database
+-- landfill_emissions.py                    # Engine 2: Landfill FOD model
+-- incineration_emissions.py                # Engine 3: Incineration calculation
+-- recycling_composting.py                  # Engine 4: Recycling/composting/AD
+-- wastewater_emissions.py                  # Engine 5: Wastewater CH4/N2O
+-- compliance_checker.py                    # Engine 6: Compliance checking
+-- waste_generated_pipeline.py              # Engine 7: Pipeline orchestration
+-- setup.py                                 # Service facade
+-- api/
    +-- __init__.py                          # API package
    +-- router.py                            # FastAPI REST endpoints

tests/unit/mrv/test_waste_generated/
+-- __init__.py
+-- conftest.py
+-- test_models.py
+-- test_config.py
+-- test_metrics.py
+-- test_provenance.py
+-- test_waste_classification_database.py
+-- test_landfill_emissions.py
+-- test_incineration_emissions.py
+-- test_recycling_composting.py
+-- test_wastewater_emissions.py
+-- test_compliance_checker.py
+-- test_waste_generated_pipeline.py
+-- test_setup.py
+-- test_api.py

deployment/database/migrations/sql/
+-- V069__waste_generated_service.sql
```

### 3.4 Database Schema (V069)

16 tables, 3 hypertables, 2 continuous aggregates:

| Table | Description | Type |
|-------|-------------|------|
| `gl_wg_waste_streams` | Waste stream definitions (type, source facility, contractor) | Regular |
| `gl_wg_waste_composition` | Waste composition breakdown by material type (%) | Regular |
| `gl_wg_emission_factors` | Emission factor library (EPA WARM, DEFRA, IPCC, custom) | Seed (200+ rows) |
| `gl_wg_landfill_params` | Landfill model parameters (DOC, MCF, k, gas capture, OX) | Seed (60+ rows) |
| `gl_wg_incineration_params` | Incineration parameters (dm, CF, FCF, OF, CH4/N2O EFs) | Seed (30+ rows) |
| `gl_wg_calculations` | Calculation results with treatment method breakdown | Hypertable (partitioned by calculated_at) |
| `gl_wg_calculation_details` | Detailed per-waste-stream breakdown per calculation | Regular |
| `gl_wg_landfill_results` | Landfill-specific results (FOD model outputs, decay projections) | Hypertable (partitioned by calculated_at) |
| `gl_wg_incineration_results` | Incineration-specific results (fossil/biogenic split, energy recovery) | Regular |
| `gl_wg_recycling_results` | Recycling/composting/AD results (process emissions, avoided memo) | Regular |
| `gl_wg_wastewater_results` | Wastewater-specific results (CH4, N2O, organic load, treatment type) | Hypertable (partitioned by calculated_at) |
| `gl_wg_compliance_checks` | Compliance check results (7 frameworks) | Regular |
| `gl_wg_uncertainty_analyses` | Uncertainty analysis results (Monte Carlo, analytical) | Regular |
| `gl_wg_aggregations` | Period aggregations by dimension (type, method, facility, period) | Regular |
| `gl_wg_diversion_analyses` | Diversion rate analysis results (by facility, period) | Regular |
| `gl_wg_provenance` | Provenance records (SHA-256 hashes, audit trail) | Regular |

**Hypertables (3):**

- `gl_wg_calculations` -- partitioned by `calculated_at` (monthly chunks)
- `gl_wg_landfill_results` -- partitioned by `calculated_at` (monthly chunks)
- `gl_wg_wastewater_results` -- partitioned by `calculated_at` (monthly chunks)

**Continuous aggregates (2):**

- `gl_wg_hourly_stats` -- hourly emission totals by treatment method
- `gl_wg_daily_stats` -- daily emission totals by treatment method and waste type

**Key seed data:**

- `gl_wg_emission_factors`: 200+ rows from EPA WARM v16 (61 materials
  x 6 methods, converted to kgCO2e/tonne), DEFRA 2025 waste factors
  (18 waste types x 4 treatment methods), IPCC defaults, EEIO waste
  sector factors (10 NAICS + 4 EXIOBASE sectors)
- `gl_wg_landfill_params`: 60+ rows covering DOC per waste type (15),
  MCF per landfill type (5), k values per climate zone x waste type (24),
  gas capture efficiency per system type (7), oxidation factors (4)
- `gl_wg_incineration_params`: 30+ rows covering dm/CF/FCF/OF per waste
  type (16), CH4 EF per incinerator type (5), N2O EF per incinerator
  type (5), NCV per waste type (7)

**Schema design principles:**

- Row-Level Security (RLS) on all tenant-facing tables via `tenant_id`
- TimescaleDB hypertables on `gl_wg_calculations`, `gl_wg_landfill_results`,
  and `gl_wg_wastewater_results` (partitioned by temporal columns)
- Continuous aggregates for hourly and daily emission statistics
- Foreign key relationships from `gl_wg_calculation_details` to
  `gl_wg_calculations`
- Foreign key relationships from `gl_wg_waste_composition` to
  `gl_wg_waste_streams`
- GIN indexes on JSONB columns for metadata queries
- B-tree indexes on waste_type, treatment_method, facility_id, tenant_id
- Partial indexes for active/current records
- Composite indexes on (tenant_id, reporting_period, treatment_method)
  for fast lookups

### 3.5 API Endpoints (20)

| # | Method | Endpoint | Description |
|---|--------|----------|-------------|
| 1 | POST | `/calculate` | Run single waste stream or multi-stream calculation |
| 2 | POST | `/calculate/batch` | Batch calculate emissions for multiple waste streams |
| 3 | POST | `/calculate/landfill` | Landfill-specific calculation (FOD model) |
| 4 | POST | `/calculate/incineration` | Incineration-specific calculation |
| 5 | POST | `/calculate/recycling` | Recycling-specific calculation |
| 6 | POST | `/calculate/composting` | Composting/AD-specific calculation |
| 7 | POST | `/calculate/anaerobic-digestion` | Anaerobic digestion-specific calculation |
| 8 | POST | `/calculate/wastewater` | Wastewater-specific calculation |
| 9 | GET | `/calculations/{calculation_id}` | Get calculation result with treatment breakdown |
| 10 | GET | `/calculations` | List calculations with filtering (type, method, period) |
| 11 | DELETE | `/calculations/{calculation_id}` | Delete a calculation |
| 12 | GET | `/emission-factors` | List waste EFs with filtering (type, method, source) |
| 13 | GET | `/emission-factors/{waste_type}` | Get emission factor for specific waste type |
| 14 | GET | `/waste-types` | List supported waste types with EWC mappings |
| 15 | GET | `/treatment-methods` | List supported treatment methods |
| 16 | POST | `/compliance/check` | Run compliance check (all 7 frameworks) |
| 17 | POST | `/uncertainty/analyze` | Run uncertainty analysis on a calculation |
| 18 | GET | `/aggregations/{period}` | Get aggregated results by period |
| 19 | POST | `/diversion/analyze` | Run diversion rate analysis |
| 20 | GET | `/provenance/{calculation_id}` | Get provenance chain for a calculation |

---

## 4. Technical Requirements

### 4.1 Zero-Hallucination Guarantees

- All emission calculations use Python `Decimal` (8 decimal places)
- No LLM calls in any calculation path -- deterministic lookups only
- Every calculation step recorded in a provenance trace
- SHA-256 provenance hash for every emission result
- Bit-perfect reproducibility: same input always produces same output
- Landfill FOD model uses exact-match DOC, MCF, k by waste type, landfill type, climate zone
- Incineration parameters are exact-match by waste type (dm, CF, FCF, OF)
- Incineration CH4/N2O EFs are exact-match by incinerator type
- Wastewater MCF is exact-match by treatment system type
- EPA WARM factors are exact-match by material type x treatment method
- DEFRA factors are exact-match by waste type x treatment method
- GWP values are deterministic lookup by assessment report version (AR4/AR5/AR6)
- Biogenic vs fossil carbon separation uses deterministic FCF lookup
- Diversion rate is deterministic arithmetic on mass values

### 4.2 Enumerations (26)

| Enum | Values | Description |
|------|--------|-------------|
| `CalculationMethod` | SUPPLIER_SPECIFIC, WASTE_TYPE_SPECIFIC, AVERAGE_DATA, SPEND_BASED | 4 calculation methods |
| `WasteTreatmentMethod` | LANDFILL, LANDFILL_WITH_GAS_CAPTURE, LANDFILL_WITH_ENERGY_RECOVERY, INCINERATION, INCINERATION_WITH_ENERGY_RECOVERY, RECYCLING_OPEN_LOOP, RECYCLING_CLOSED_LOOP, COMPOSTING, ANAEROBIC_DIGESTION, WASTEWATER_TREATMENT, OTHER | 11 treatment methods |
| `WasteCategory` | PAPER_CARDBOARD, PLASTICS, GLASS, METALS, FOOD_WASTE, GARDEN_WASTE, TEXTILES, WOOD, RUBBER_LEATHER, ELECTRONICS, CONSTRUCTION_DEMOLITION, HAZARDOUS, MIXED_MSW, OTHER | 14 waste categories |
| `WasteStream` | MUNICIPAL_SOLID_WASTE, COMMERCIAL_INDUSTRIAL, CONSTRUCTION_DEMOLITION, HAZARDOUS, WASTEWATER, SPECIAL | 6 waste stream types |
| `EWCChapter` | CH_01 through CH_20 | 20 EWC chapters |
| `LandfillType` | MANAGED_ANAEROBIC, MANAGED_SEMI_AEROBIC, UNMANAGED_DEEP, UNMANAGED_SHALLOW, UNCATEGORIZED | 5 landfill types |
| `ClimateZone` | BOREAL_TEMPERATE_DRY, TEMPERATE_WET, TROPICAL_DRY, TROPICAL_WET | 4 IPCC climate zones |
| `IncineratorType` | CONTINUOUS_STOKER, SEMI_CONTINUOUS, BATCH, FLUIDIZED_BED, OPEN_BURNING | 5 incinerator types |
| `RecyclingType` | OPEN_LOOP, CLOSED_LOOP | 2 recycling types |
| `WastewaterTreatmentSystem` | CENTRALIZED_AEROBIC, CENTRALIZED_AEROBIC_NOT_MANAGED, CENTRALIZED_ANAEROBIC, ANAEROBIC_REACTOR, LAGOON_SHALLOW, LAGOON_DEEP, SEPTIC, UNTREATED, CONSTRUCTED_WETLAND | 9 wastewater treatment systems |
| `GasCollectionSystem` | NONE, ACTIVE_OPERATING, ACTIVE_TEMP_COVER, ACTIVE_CLAY_COVER, ACTIVE_GEOMEMBRANE, PASSIVE_VENTING, FLARE_ONLY | 7 gas collection system types |
| `EFSource` | EPA_WARM, DEFRA_BEIS, IPCC_2006, IPCC_2019, CUSTOM | 5 EF sources |
| `ComplianceFramework` | GHG_PROTOCOL, ISO_14064, CSRD_ESRS, CDP, SBTI, EU_WASTE_DIRECTIVE, EPA_40CFR98 | 7 regulatory frameworks |
| `DataQualityTier` | TIER_1, TIER_2, TIER_3 | 3 IPCC data quality tiers |
| `WasteDataSource` | WASTE_AUDIT, TRANSFER_NOTES, PROCUREMENT_ESTIMATE, SPEND_ESTIMATE | 4 waste data source types |
| `ProvenanceStage` | VALIDATE, CLASSIFY, NORMALIZE, RESOLVE_EFS, CALCULATE_TREATMENT, CALCULATE_TRANSPORT, ALLOCATE, COMPLIANCE, AGGREGATE, SEAL | 10 pipeline stages |
| `UncertaintyMethod` | IPCC_DEFAULT, MONTE_CARLO, ERROR_PROPAGATION | 3 uncertainty methods |
| `DivertedDestination` | REUSE, RECYCLING, COMPOSTING, ANAEROBIC_DIGESTION, OTHER_RECOVERY | 5 diversion destinations |
| `DisposalDestination` | LANDFILL, INCINERATION, OTHER_DISPOSAL | 3 disposal destinations |
| `HazardClass` | H1, H2, H3, H4_1, H4_2, H4_3, H5_1, H5_2, H6_1, H6_2, H8, H10, H11, H12, H13 | 15 Basel Convention hazard classes |
| `IndustryWastewaterType` | STARCH, ALCOHOL, BEER_MALT, PULP_PAPER, FOOD, MEAT_POULTRY, VEGETABLES, DAIRY, SUGAR, TEXTILE, PHARMACEUTICAL, OTHER | 12 industry wastewater types |
| `DQIDimension` | TEMPORAL, GEOGRAPHICAL, TECHNOLOGICAL, COMPLETENESS, RELIABILITY | 5 DQI dimensions |
| `DQIScore` | VERY_GOOD (1), GOOD (2), FAIR (3), POOR (4), VERY_POOR (5) | 5 quality scores |
| `ComplianceStatus` | COMPLIANT, PARTIAL, NON_COMPLIANT | 3 compliance statuses |
| `GWPSource` | AR4, AR5, AR6, AR6_20YR | 4 GWP assessment report versions |
| `EmissionGas` | CO2_FOSSIL, CO2_BIOGENIC, CH4, N2O, CO2E | 5 emission gas types |

### 4.3 Models (28)

| Model | Description | Key Fields |
|-------|-------------|------------|
| `WasteStreamInput` | Waste stream input record | stream_id, facility_id, waste_category, ewc_code, treatment_method, mass_tonnes, wet_dry_basis, contractor_id, period |
| `WasteCompositionInput` | Waste composition breakdown | composition_id, stream_id, material_type, fraction_pct, doc_override, notes |
| `LandfillInput` | Landfill-specific parameters | landfill_type, climate_zone, gas_collection, oxidation_factor, years_projection, doc_override, mcf_override |
| `IncinerationInput` | Incineration-specific parameters | incinerator_type, energy_recovery, thermal_efficiency, dm_override, cf_override, fcf_override |
| `RecyclingInput` | Recycling-specific parameters | recycling_type, quality_factor, avoided_emissions_calc |
| `CompostingInput` | Composting-specific parameters | composting_type, is_home_composting, mass_wet, ch4_ef_override, n2o_ef_override |
| `AnaerobicDigestionInput` | AD-specific parameters | plant_type, biogas_ch4_content, leakage_rate_override, digestate_handling |
| `WastewaterInput` | Wastewater-specific parameters | treatment_system, volume_m3, cod_mg_per_l, bod_mg_per_l, n_effluent_kg, industry_type, bo_override, mcf_override |
| `WasteEmissionFactor` | Waste emission factor record | ef_id, waste_type, treatment_method, ef_kgco2e_per_tonne, source, source_year, region |
| `WasteClassificationResult` | Classification output | waste_category, ewc_code, ewc_chapter, treatment_method, hazard_class, data_quality_tier |
| `LandfillEmissionsResult` | Landfill calculation output | ch4_generated_tonnes, ch4_recovered_tonnes, ch4_oxidized_tonnes, ch4_emitted_tonnes, co2e_tonnes, fod_parameters, decay_projection |
| `IncinerationEmissionsResult` | Incineration calculation output | co2_fossil_tonnes, co2_biogenic_tonnes, ch4_tonnes, n2o_tonnes, co2e_tonnes, energy_recovered_kwh, avoided_co2e_memo |
| `RecyclingCompostingResult` | Recycling/composting/AD output | process_emissions_co2e, avoided_emissions_memo_co2e, ch4_tonnes, n2o_tonnes, diversion_mass_tonnes |
| `WastewaterEmissionsResult` | Wastewater calculation output | ch4_tonnes, n2o_tonnes, co2e_tonnes, tow_kg_cod, mcf_applied, treatment_system |
| `WasteCalculationResult` | Complete single-stream result | calculation_id, stream_id, treatment_method, total_co2e, gas_breakdown, ef_source, method, dqi_score, provenance_hash, processing_time_ms |
| `WasteBatchResult` | Batch calculation output | batch_id, results, total_co2e, by_treatment_method, by_waste_type, success_count, failure_count, processing_time_ms |
| `WasteAggregation` | Single aggregation dimension | dimension, key, co2e_tonnes, mass_tonnes, fraction_of_total |
| `WasteAggregationResult` | Aggregated emissions output | aggregation_id, period, total_co2e, by_treatment_method, by_waste_type, by_facility, intensity_metrics, diversion_rate |
| `ComplianceCheckInput` | Compliance check request | calculation_ids, frameworks, organization_context |
| `ComplianceCheckResult` | Compliance check output | result_id, framework, status, score, findings, gaps, recommendations |
| `UncertaintyInput` | Uncertainty analysis request | calculation_id, method, iterations, confidence_level |
| `UncertaintyResult` | Uncertainty analysis output | result_id, method, mean, median, p5, p95, std_dev, confidence_interval |
| `DataQualityInput` | DQI assessment request | calculation_id, temporal, geographical, technological, completeness, reliability |
| `DataQualityResult` | DQI assessment output | composite_dqi, classification, dimension_scores, recommended_actions |
| `ProvenanceRecord` | Provenance chain record | record_id, stage, input_hash, output_hash, timestamp, parameters_used |
| `ProvenanceChainResult` | Full provenance chain | chain_id, records, final_hash, is_valid |
| `WasteMetricsSummary` | Metrics summary | total_waste_tonnes, total_co2e, diversion_rate, intensity_per_tonne, intensity_per_revenue, by_treatment |
| `WasteDiversionAnalysis` | Diversion analysis output | total_generated, total_diverted, total_disposed, diversion_rate, by_destination, by_facility, trend |
| `WasteCompositionProfile` | Composition profile | profile_id, region, waste_stream, material_fractions, source |

### 4.4 Constant Tables (15)

#### 4.4.1 GWP_VALUES

| Gas | AR4 (100yr) | AR5 (100yr) | AR6 (100yr) | AR6 (20yr) |
|-----|-------------|-------------|-------------|------------|
| CO2 | 1 | 1 | 1 | 1 |
| CH4 | 25 | 28 | 27.9 | 82.5 |
| N2O | 298 | 265 | 273 | 273 |

#### 4.4.2 DOC_VALUES

See Section 2.5 for complete DOC values by waste type (15 entries).

#### 4.4.3 MCF_VALUES

See Section 2.6 for complete MCF values by landfill type (5 entries).

#### 4.4.4 DECAY_RATE_CONSTANTS

See Section 2.7 for complete k values by climate zone x waste type
(24 entries across 4 zones x 6 waste types).

#### 4.4.5 GAS_CAPTURE_EFFICIENCY

See Section 2.8 for complete gas capture efficiency values by system
type (7 entries) and oxidation factors by cover type (4 entries).

#### 4.4.6 INCINERATION_PARAMETERS

See Section 2.10 for complete dm, CF, FCF, OF values per waste type
(16 entries).

#### 4.4.7 CH4_N2O_EF_INCINERATION

See Section 2.11 for complete CH4 and N2O emission factors by
incinerator type (5 entries).

#### 4.4.8 COMPOSTING_EF

| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| CH4 (industrial composting) | 4.0 | g CH4/kg wet waste | IPCC 2006 |
| N2O (industrial composting) | 0.30 | g N2O/kg wet waste | IPCC 2006 |
| CH4 (home composting) | 10.0 | g CH4/kg wet waste | IPCC 2019 |
| N2O (home composting) | 0.60 | g N2O/kg wet waste | IPCC 2019 |

#### 4.4.9 ANAEROBIC_DIGESTION_LEAKAGE

See Section 2.13 for complete leakage rates by plant type (4 entries).

#### 4.4.10 WASTEWATER_MCF

See Section 2.14 for complete MCF values by treatment system (8 entries).

#### 4.4.11 WASTEWATER_Bo

See Section 2.14 for Bo values on COD and BOD basis (2 entries).

#### 4.4.12 EPA_WARM_FACTORS

See Section 2.15 for selected EPA WARM v16 factors (61 material types
x 6 disposal methods -- full table seeded in database).

#### 4.4.13 DEFRA_WASTE_FACTORS

See Section 2.16 for selected DEFRA/DESNZ waste emission factors
(18 waste types x 4 treatment methods).

#### 4.4.14 EEIO_WASTE_FACTORS

See Section 2.17 for EEIO waste sector factors (10 NAICS sectors
+ 4 EXIOBASE sectors).

#### 4.4.15 MSW_COMPOSITION

| Material | Developed Countries (%) | Developing Countries (%) | Source |
|----------|----------------------|------------------------|--------|
| Food waste | 28 | 55 | IPCC 2019, World Bank |
| Garden waste | 10 | 5 | IPCC 2019 |
| Paper/cardboard | 25 | 8 | IPCC 2019 |
| Plastics | 12 | 10 | IPCC 2019 |
| Glass | 5 | 3 | IPCC 2019 |
| Metals | 5 | 3 | IPCC 2019 |
| Textiles | 3 | 3 | IPCC 2019 |
| Wood | 4 | 2 | IPCC 2019 |
| Rubber/leather | 1 | 1 | IPCC 2019 |
| Other inert | 7 | 10 | IPCC 2019 |

### 4.5 Regulatory Frameworks (7)

1. **GHG Protocol Scope 3 Standard** -- Chapter 5 Category 5 definition;
   waste types, treatment methods, method hierarchy; Chapter 7 DQI;
   Chapter 9 reporting requirements; on-site vs off-site boundary rule;
   cut-off approach for recycling credits.

2. **ISO 14064-1:2018** -- Clause 5.2.4: Category 4 indirect GHG emissions
   from products used by organization (waste generated by operations);
   requires documented methodology and uncertainty assessment.

3. **CSRD/ESRS E1** -- E1-6 para 44a/44b/44c Scope 3 by category,
   methodology, data sources; para 46 intensity metrics.

4. **CSRD/ESRS E5** -- Resource use and circular economy:
   E5-1: Policies on resource use and circular economy
   E5-2: Actions and resources (waste reduction, diversion targets)
   E5-3: Targets (diversion rate, waste reduction, recycling rate)
   E5-4: Resource inflows (material composition of inputs)
   E5-5: Resource outflows (waste by type and treatment, hazardous waste,
          radioactive waste)
   E5-6: Potential financial effects of resource use and circular economy

5. **CDP Climate Change** -- C6.5 Category 5 relevance assessment and
   calculation; waste by treatment method breakdown; methodology;
   waste reduction initiatives; year-over-year explanation.

6. **SBTi v5.3** -- Scope 3 target required if >40% of total; 67%
   coverage; waste reduction engagement targets.

7. **EU Waste Framework Directive 2008/98/EC** -- Waste hierarchy
   (prevention > reuse > recycling > recovery > disposal); recycling
   targets (55% municipal waste by 2025, 60% by 2030, 65% by 2035);
   landfill reduction targets; hazardous waste management requirements.

8. **EPA 40 CFR Part 98 Subpart HH/TT** -- Mandatory GHG reporting for
   MSW landfills (Subpart HH) and industrial waste landfills (Subpart TT);
   FOD model parameters; reporting thresholds.

**Category 5-specific compliance rules:**

| Rule Code | Description | Framework |
|-----------|-------------|-----------|
| `TREATMENT_BOUNDARY` | Only 3rd-party waste treatment in Cat 5 | GHG Protocol |
| `ONSITE_EXCLUSION` | On-site treatment excluded (Scope 1/2) | GHG Protocol |
| `CUTOFF_APPROACH` | Recycling credits reported as memo only | GHG Protocol |
| `WTE_CREDIT_MEMO` | WtE energy recovery reported as memo only | GHG Protocol |
| `DOUBLE_COUNT_CAT1` | No overlap with cradle-to-gate supplier factors | GHG Protocol |
| `DOUBLE_COUNT_CAT12` | No overlap with end-of-life of sold products | GHG Protocol |
| `WASTE_HIERARCHY` | Waste hierarchy preference documented | EU WFD |
| `DIVERSION_TARGET` | Diversion rate vs target tracking | EU WFD, ESRS E5 |
| `HAZARDOUS_CLASSIFICATION` | Hazardous waste properly classified | Basel, EU WFD |
| `BIOGENIC_SEPARATION` | Fossil vs biogenic CO2 separated for incineration | IPCC, GHG Protocol |
| `FOD_PARAMETERS` | Landfill FOD parameters documented and justified | IPCC, EPA 40 CFR |
| `DATA_QUALITY_MINIMUM` | Method hierarchy compliance; DQI documented | All |

**Framework required disclosures:**

| Framework | Required Disclosures for Category 5 |
|-----------|-------------------------------------|
| GHG Protocol Scope 3 | Total Cat 5 emissions (tCO2e), breakdown by treatment method (landfill/incineration/recycling/composting/AD/wastewater), methodology per treatment, waste mass by type, on-site vs off-site distinction |
| ISO 14064-1 | Documented methodology, emission factors and sources, uncertainty assessment, base year recalculation policy |
| CSRD/ESRS E1 | E1-6 para 44: Cat 5 gross emissions with treatment breakdown, methodology, % supplier-specific |
| CSRD/ESRS E5 | E5-5: Total waste generated (tonnes), breakdown by type and treatment, diversion rate, hazardous waste, recycling rate, circular economy targets and progress |
| CDP | C6.5: Cat 5 relevance, emissions by treatment, methodology, waste reduction initiatives, YoY explanation |
| SBTi | Scope 3 screening, Cat 5 significance, waste reduction engagement targets, coverage validation |
| EU WFD | Waste hierarchy compliance, diversion targets, recycling rates, hazardous waste classification |
| EPA 40 CFR | FOD model documentation, landfill gas capture data, waste composition, reporting threshold compliance |

### 4.6 Performance Targets

| Metric | Target |
|--------|--------|
| Single waste stream calculation (type-specific, non-landfill) | < 15ms |
| Single landfill calculation (simplified single-year) | < 20ms |
| Single landfill calculation (FOD multi-year, 30yr projection) | < 50ms |
| Single incineration calculation (per-waste-type) | < 15ms |
| Single wastewater calculation | < 15ms |
| Batch calculation (100 waste streams) | < 2s |
| Batch calculation (1,000 waste streams) | < 10s |
| EF resolution (6-level hierarchy, single stream) | < 10ms |
| Waste classification (single stream with EWC mapping) | < 5ms |
| Composting/AD calculation (single stream) | < 10ms |
| DQI scoring (full inventory) | < 200ms |
| Compliance check (all 7 frameworks) | < 300ms |
| Full pipeline (500 waste streams, mixed methods) | < 15s |
| Diversion rate analysis (full inventory) | < 200ms |
| Uncertainty analysis (Monte Carlo, 10,000 iterations) | < 5s |

---

## 5. Acceptance Criteria

### 5.1 Core Calculation -- Landfill Emissions

- [ ] IPCC First Order Decay (FOD) model: DDOCm = W * DOC * DOCf * MCF
- [ ] Multi-year decay projection with climate-zone-specific k values
- [ ] DOC values for 15 waste types (food, garden, paper, wood, textiles, etc.)
- [ ] MCF values for 5 landfill types (managed/unmanaged/semi-aerobic)
- [ ] Decay rate constants (k) for 4 climate zones x 6 waste types (24 values)
- [ ] Landfill gas capture efficiency for 7 system types (0-90%)
- [ ] Oxidation factor for 4 cover types (0.0-0.20)
- [ ] Simplified single-year calculation mode
- [ ] CH4 to CO2e conversion with GWP (AR4/AR5/AR6)
- [ ] Net emissions: (generated - recovered) * (1 - OX)

### 5.2 Core Calculation -- Incineration Emissions

- [ ] CO2_fossil = SW * dm * CF * FCF * OF * 44/12 for 16+ waste types
- [ ] CH4 emissions by 5 incinerator types
- [ ] N2O emissions by 5 incinerator types
- [ ] Biogenic vs fossil CO2 separation (FCF parameter)
- [ ] Biogenic CO2 reported as memo item only
- [ ] Energy recovery tracking (NCV * mass * efficiency)
- [ ] Avoided emissions from WtE as memo item (NOT deducted)
- [ ] Per-gas breakdown (CO2_fossil, CO2_biogenic, CH4, N2O)

### 5.3 Core Calculation -- Recycling and Composting

- [ ] Recycling: cut-off approach (process emissions only)
- [ ] Open-loop vs closed-loop recycling distinction
- [ ] Avoided emissions calculated but reported as memo item only
- [ ] Composting: CH4 = Mass * EF_CH4, N2O = Mass * EF_N2O
- [ ] Industrial vs home composting EF differentiation
- [ ] Anaerobic digestion: CH4 leakage by plant type (2-7%)
- [ ] Digestate handling emissions
- [ ] Downcycling quality factor for open-loop recycling

### 5.4 Core Calculation -- Wastewater Emissions

- [ ] CH4 = TOW * Bo * MCF for 9 treatment system types
- [ ] N2O = N_effluent * EF_N2O * 44/28
- [ ] Bo on COD basis (0.25 kg CH4/kg COD) and BOD basis (0.60 kg CH4/kg BOD)
- [ ] MCF values for 9 treatment systems (0.00 to 0.80)
- [ ] Industry-specific wastewater loads for 12 industry types
- [ ] TOW calculation from volume and COD/BOD concentration
- [ ] Sludge management emissions
- [ ] Per-gas breakdown (CH4, N2O)

### 5.5 Waste Classification

- [ ] 14 waste categories with EF mapping
- [ ] EWC 6-digit code mapping (20 chapters)
- [ ] Basel Convention hazard classification (H1-H13)
- [ ] Waste-to-treatment method compatibility matrix
- [ ] MSW composition profiles by region (developed vs developing)
- [ ] EPA WARM material type mapping (61 types)
- [ ] DEFRA waste type mapping (18 types)

### 5.6 Category Boundary & Double-Counting Prevention

- [ ] On-site vs off-site boundary enforcement (off-site only = Cat 5)
- [ ] Cradle-to-gate overlap check vs Category 1
- [ ] End-of-life sold products overlap check vs Category 12
- [ ] Recycling credit enforcement (memo only, NOT deducted)
- [ ] WtE credit enforcement (memo only, NOT deducted)
- [ ] Scope 1 boundary check (on-site treatment = Scope 1)
- [ ] Facility ownership verification

### 5.7 Data Quality & Uncertainty

- [ ] 5-dimension DQI scoring (temporal, geographical, technological, completeness, reliability)
- [ ] Composite DQI score (1.0-5.0 scale)
- [ ] Quality classification (Very Good through Very Poor)
- [ ] Pedigree matrix uncertainty quantification
- [ ] Weighted DQI for total inventory (emission-weighted)
- [ ] Uncertainty analysis (Monte Carlo, analytical, IPCC default)
- [ ] Landfill FOD uncertainty propagation

### 5.8 Compliance

- [ ] 7 regulatory framework compliance checks
- [ ] 12 Category 5-specific compliance rules
- [ ] ESRS E5 circular economy disclosure completeness
- [ ] EU Waste Framework Directive hierarchy compliance
- [ ] Diversion rate calculation and target tracking
- [ ] Hazardous waste classification validation
- [ ] Biogenic/fossil CO2 separation enforcement
- [ ] FOD parameter documentation check
- [ ] Gap identification with actionable recommendations

### 5.9 Infrastructure

- [ ] 20 REST API endpoints at `/api/v1/waste-generated`
- [ ] V069 database migration (16 tables, 3 hypertables, 2 continuous aggregates)
- [ ] SHA-256 provenance on every calculation result
- [ ] Prometheus metrics with `gl_wg_` prefix (12 metrics)
- [ ] Auth integration (route_protector.py + auth_setup.py with wg_router)
- [ ] 20 PERMISSION_MAP entries for waste-generated
- [ ] 600+ unit tests
- [ ] All calculations use Python `Decimal` (no floating point in emission path)
- [ ] Export in JSON, CSV, Excel, and PDF formats
- [ ] Row-Level Security (RLS) on all tenant-facing tables
- [ ] Provenance chain linking to waste audit source data

---

## 6. Prometheus Metrics (12)

All prefixed `gl_wg_`:

| # | Metric | Type | Labels |
|---|--------|------|--------|
| 1 | calculations_total | Counter | treatment_method, method, status |
| 2 | emissions_kg_co2e_total | Counter | treatment_method, waste_type, gas |
| 3 | landfill_calculations_total | Counter | landfill_type, climate_zone, status |
| 4 | incineration_calculations_total | Counter | incinerator_type, status |
| 5 | wastewater_calculations_total | Counter | treatment_system, status |
| 6 | factor_selections_total | Counter | method, source, treatment_method |
| 7 | compliance_checks_total | Counter | framework, status |
| 8 | batch_jobs_total | Counter | status |
| 9 | calculation_duration_seconds | Histogram | operation |
| 10 | batch_size | Histogram | method |
| 11 | active_calculations | Gauge | - |
| 12 | waste_streams_processed | Gauge | treatment_method |

---

## 7. Auth Integration

### 7.1 PERMISSION_MAP Entries (20)

The following 20 permission entries are added to `PERMISSION_MAP` in
`auth_setup.py` for the waste-generated resource:

| # | Method | Path Pattern | Permission |
|---|--------|-------------|------------|
| 1 | POST | /api/v1/waste-generated/calculate | waste-generated:execute |
| 2 | POST | /api/v1/waste-generated/calculate/batch | waste-generated:execute |
| 3 | POST | /api/v1/waste-generated/calculate/landfill | waste-generated:execute |
| 4 | POST | /api/v1/waste-generated/calculate/incineration | waste-generated:execute |
| 5 | POST | /api/v1/waste-generated/calculate/recycling | waste-generated:execute |
| 6 | POST | /api/v1/waste-generated/calculate/composting | waste-generated:execute |
| 7 | POST | /api/v1/waste-generated/calculate/anaerobic-digestion | waste-generated:execute |
| 8 | POST | /api/v1/waste-generated/calculate/wastewater | waste-generated:execute |
| 9 | GET | /api/v1/waste-generated/calculations/{id} | waste-generated:read |
| 10 | GET | /api/v1/waste-generated/calculations | waste-generated:read |
| 11 | DELETE | /api/v1/waste-generated/calculations/{id} | waste-generated:write |
| 12 | GET | /api/v1/waste-generated/emission-factors | waste-generated:read |
| 13 | GET | /api/v1/waste-generated/emission-factors/{waste_type} | waste-generated:read |
| 14 | GET | /api/v1/waste-generated/waste-types | waste-generated:read |
| 15 | GET | /api/v1/waste-generated/treatment-methods | waste-generated:read |
| 16 | POST | /api/v1/waste-generated/compliance/check | waste-generated:execute |
| 17 | POST | /api/v1/waste-generated/uncertainty/analyze | waste-generated:execute |
| 18 | GET | /api/v1/waste-generated/aggregations/{period} | waste-generated:read |
| 19 | POST | /api/v1/waste-generated/diversion/analyze | waste-generated:execute |
| 20 | GET | /api/v1/waste-generated/provenance/{id} | waste-generated:read |

### 7.2 Router Registration

In `auth_setup.py`, add:

```python
from greenlang.waste_generated.api.router import router as wg_router
```

Register in `configure_auth(app)`:

```python
app.include_router(wg_router, prefix="/api/v1/waste-generated", tags=["waste-generated"])
```

Apply route protection:

```python
protect_routes(app, wg_router, permission_map=PERMISSION_MAP)
```

---

## 8. Test Suite (600+ tests)

### 8.1 Test Files (15)

| # | File | Tests | Coverage Area |
|---|------|-------|--------------|
| 1 | `test_models.py` | 60 | All 26 enums, 28 models, 15 constant tables, frozen immutability |
| 2 | `test_config.py` | 15 | GL_WG_ env vars, defaults, validation |
| 3 | `test_metrics.py` | 15 | 12 Prometheus metrics registration, labels, types |
| 4 | `test_provenance.py` | 20 | SHA-256 hashing, chain linking, determinism |
| 5 | `test_waste_classification_database.py` | 65 | EF lookup by waste type/treatment method, EWC mapping, EPA WARM, DEFRA, hierarchy |
| 6 | `test_landfill_emissions.py` | 85 | FOD model, DOC/MCF/k values, gas capture, OX, multi-year decay, simplified mode |
| 7 | `test_incineration_emissions.py` | 60 | CO2_fossil, CH4, N2O, biogenic separation, energy recovery, all incinerator types |
| 8 | `test_recycling_composting.py` | 55 | Cut-off approach, open/closed loop, composting CH4/N2O, AD leakage, avoided emissions memo |
| 9 | `test_wastewater_emissions.py` | 55 | CH4 from TOW, N2O from effluent, MCF by treatment, Bo COD/BOD, industry loads |
| 10 | `test_compliance_checker.py` | 60 | 7 frameworks, 12 rules, double-counting, ESRS E5, waste hierarchy, diversion |
| 11 | `test_waste_generated_pipeline.py` | 55 | 10 stages, batch processing, error handling, provenance |
| 12 | `test_setup.py` | 15 | Service facade initialization, engine wiring |
| 13 | `test_api.py` | 30 | 20 endpoints, request/response validation, error responses |
| 14 | `conftest.py` | 10 | Shared fixtures, mock data, test waste streams |
| 15 | `__init__.py` | 0 | Package marker |
| | **TOTAL** | **600+** | |

### 8.2 Key Test Scenarios

**Landfill calculation tests:**

- Food waste: 100 tonnes, managed anaerobic landfill, temperate wet
  climate zone, no gas capture, OX=0.10, DOC=0.15, MCF=1.0, k=0.185
  --> expected kgCO2e
- Paper/cardboard: 50 tonnes, managed landfill with active gas capture
  (75% efficiency), geomembrane cover (OX=0.10), DOC=0.40, MCF=1.0
  --> expected kgCO2e (significantly reduced by gas capture)
- Mixed MSW: 500 tonnes, unmanaged deep, tropical wet, MCF=0.8,
  DOC=0.16, k=0.17 --> expected kgCO2e
- Wood: 30 tonnes, FOD 30-year projection, boreal/temperate dry,
  k=0.02 (slow decay) --> year-by-year emission profile
- Zero-DOC material (glass): 200 tonnes landfill --> 0 CH4 (no organic)

**Incineration calculation tests:**

- Mixed plastics: 50 tonnes, continuous stoker, FCF=1.0, dm=1.0,
  CF=0.75 --> high fossil CO2, minimal CH4/N2O
- Food waste: 100 tonnes, fluidized bed, FCF=0.0 (all biogenic),
  dm=0.40 --> zero fossil CO2, biogenic CO2 as memo, CH4+N2O only
- Mixed MSW: 200 tonnes, WtE with 25% efficiency, NCV=10 MJ/kg
  --> fossil CO2 + CH4 + N2O + energy recovery memo
- Textiles (50/50 natural/synthetic): 20 tonnes --> split FCF calculation

**Recycling and composting tests:**

- Recycling (closed-loop, aluminum): 10 tonnes --> process emissions only,
  avoided emissions (-9.13 MTCO2e/short ton) as memo
- Composting: 50 tonnes food waste, industrial --> CH4 + N2O calculation
- Composting: 10 tonnes garden waste, home composting --> higher EFs
- AD (modern enclosed): 100 tonnes food waste, 2% leakage --> CH4
- AD (covered lagoon): 50 tonnes, 7% leakage --> higher CH4

**Wastewater calculation tests:**

- Food processing: 10,000 m3, COD=5.0 kg/m3, centralized aerobic
  (MCF=0.00) --> minimal CH4, N2O from effluent only
- Meat processing: 5,000 m3, COD=4.1 kg/m3, anaerobic lagoon deep
  (MCF=0.80) --> high CH4, plus N2O
- Brewery: 20,000 m3, COD=3.0 kg/m3, UASB reactor (MCF=0.80)
  --> CH4 from organic load + N2O
- Pulp & paper: 50,000 m3, COD=7.0 kg/m3, centralized aerobic -->
  low CH4 (well-managed), N2O from nitrogen

**Compliance tests:**

- On-site incinerator flagged as Scope 1 (not Cat 5)
- Recycling credit subtracted from total rejected (memo only)
- WtE energy recovery subtracted from total rejected (memo only)
- Category 1 overlap flagged when supplier EF includes end-of-life
- Category 12 overlap flagged for customer product waste
- ESRS E5 completeness check for circular economy disclosures
- EU WFD waste hierarchy compliance assessment
- Diversion rate vs 55%/60%/65% EU targets

**Boundary tests:**

- On-site composting: rejected as Scope 1 (not Cat 5)
- Customer end-of-life waste: rejected as Category 12
- Packaging waste in cradle-to-gate EF: flagged as Category 1 overlap
- Third-party landfill: accepted as Category 5
- Third-party wastewater treatment: accepted as Category 5

---

## 9. Key Differentiators from Adjacent Categories

### 9.1 Category 5 vs Category 12

| Aspect | Category 5 (Waste Generated) | Category 12 (End-of-Life Sold Products) |
|--------|-------------------------------|----------------------------------------|
| Waste generator | Reporting company's own operations | Customers using reporting company's sold products |
| Waste source | Production waste, packaging, office waste, food waste, wastewater | Sold products at end-of-life (product + packaging) |
| Treatment location | 3rd-party treatment of company's waste | 3rd-party treatment of customer waste |
| Control | Company controls waste generation | Customer controls disposal decisions |
| Data availability | Waste audit, transfer notes, contractor invoices | Assumptions about product lifespan and end-of-life fate |
| Agent | AGENT-MRV-018 (this agent) | Future AGENT-MRV-025 |

### 9.2 Category 5 vs Category 1

| Aspect | Category 1 (Purchased Goods) | Category 5 (Waste Generated) |
|--------|-----------------------------|---------------------------------|
| What is covered | Cradle-to-gate emissions of purchased goods | Disposal/treatment of waste generated in operations |
| Waste inclusion | Some Cat 1 EFs include end-of-life of packaging | Separate waste treatment emissions |
| Double-counting risk | If Cat 1 EF includes end-of-life/disposal, Cat 5 should exclude that waste | Flag cradle-to-grave EFs that include end-of-life component |
| Agent | AGENT-MRV-014 | AGENT-MRV-018 (this agent) |

### 9.3 Category 5 vs Scope 1

| Aspect | Scope 1 (Direct) | Category 5 (Waste Generated) |
|--------|------------------|---------------------------------|
| Treatment location | On-site at reporting company | Off-site at 3rd-party facility |
| Ownership | Company owns/controls treatment equipment | 3rd-party waste contractor |
| Examples | On-site incinerator, on-site wastewater plant | Municipal landfill, commercial incinerator |
| Agent | AGENT-MRV-001 through MRV-008 | AGENT-MRV-018 (this agent) |

### 9.4 Category 5 vs Scope 2

| Aspect | Scope 2 (Purchased Energy) | Category 5 (Waste Generated) |
|--------|--------------------------|--------------------------------|
| Energy scope | Electricity for on-site waste equipment | Emissions from waste treatment processes |
| Examples | Power for on-site compactor, on-site MRF | CH4 from landfill decomposition, CO2 from incineration |
| Agent | AGENT-MRV-009 through MRV-013 | AGENT-MRV-018 (this agent) |

---

## 10. Dependencies

| Component | Purpose |
|-----------|---------|
| Python 3.11+ | Runtime |
| Pydantic v2 | Data models, validation |
| FastAPI | REST API framework |
| prometheus_client | Prometheus metrics |
| psycopg[binary] | PostgreSQL driver |
| TimescaleDB | Hypertables and continuous aggregates |
| AGENT-MRV-014 | Purchased Goods Agent (Cat 1 double-counting check: cradle-to-gate EF end-of-life overlap) |
| AGENT-MRV-008 | On-site Waste Treatment Agent (Scope 1 boundary check: on-site vs off-site) |
| AGENT-DATA-002 | Excel/CSV Normalizer (waste audit spreadsheets, transfer notes) |
| AGENT-DATA-003 | ERP/Finance Connector (waste contractor invoices, spend data) |
| AGENT-DATA-008 | Supplier Questionnaire Processor (waste contractor emission reports) |
| AGENT-DATA-009 | Spend Data Categorizer (waste management spend NAICS classification) |
| AGENT-DATA-010 | Data Quality Profiler (input data quality scoring) |
| AGENT-FOUND-001 | Orchestrator (DAG pipeline execution) |
| AGENT-FOUND-003 | Unit & Reference Normalizer (tonnes/kg, wet/dry basis conversion) |
| AGENT-FOUND-005 | Citations & Evidence Agent (EF source citations, EPA WARM/DEFRA/IPCC) |
| AGENT-FOUND-008 | Reproducibility Agent (artifact hashing, drift detection) |
| AGENT-FOUND-009 | QA Test Harness (golden file testing) |
| AGENT-FOUND-010 | Observability Agent (metrics, traces, SLO tracking) |

---

## 11. Changelog

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2026-02-25 | Initial PRD |
