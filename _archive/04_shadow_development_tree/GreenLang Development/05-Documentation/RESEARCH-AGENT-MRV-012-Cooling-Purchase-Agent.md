# Research: AGENT-MRV-012 -- Scope 2 Cooling Purchase Agent

## Executive Summary

This document provides the technical research foundation for AGENT-MRV-012,
a dedicated Scope 2 Cooling Purchase Agent. The research justifies separating
cooling from the existing MRV-011 Steam/Heat Purchase Agent and provides
all emission factors, formulas, technology parameters, and regulatory
requirements needed for implementation.

**Key finding**: Purchased cooling is fundamentally different from purchased
steam and heat in its generation technologies, efficiency metrics, temporal
emission profiles, refrigerant considerations, and regional significance.
A dedicated agent is warranted to properly handle COP/EER calculations,
multi-technology cooling plants, thermal energy storage temporal shifting,
free cooling modes, refrigerant leakage tracking, part-load efficiency
modeling, and cooling-specific regulatory requirements.

---

## 1. Justification for a Separate Cooling Agent

### 1.1 Why Cooling Differs from Steam/Heat

MRV-011 treats cooling as a secondary concern within a steam/heat-focused
agent. However, cooling has unique characteristics that justify dedicated
treatment:

| Dimension | Steam/Heat (MRV-011) | Cooling (MRV-012) |
|-----------|---------------------|-------------------|
| **Efficiency metric** | Boiler efficiency (eta, 0.65-0.98) | COP/EER (0.6-30.0), IPLV/NPLV |
| **Generation technologies** | Boilers (fuel-fired, electric) | Electric chillers (3 types), absorption chillers (3 effects), free cooling, TES |
| **Energy domain** | Single (thermal from combustion) | Dual (electrical AND thermal via absorption) |
| **Temporal behavior** | Steady demand, seasonal | Highly variable, diurnal peaks, TES load-shifting |
| **Refrigerant impact** | None | HFC/HFO leakage (Scope 1 cross-reference) |
| **Part-load behavior** | Linear efficiency degradation | Complex COP curves, IPLV/NPLV weighting |
| **Free source potential** | Waste heat only | Sea/lake/river water, cold ambient air |
| **Network losses** | 5-15% (high-temp, well-insulated) | 3-8% (low delta-T, pump-energy dominant) |
| **Regional concentration** | Northern Europe, Northern US/Canada | Middle East, SE Asia, US Sun Belt, data centers |
| **Market size** | ~USD 200B (heating) | USD 27B growing to 56B by 2033 (8.3% CAGR) |
| **Data center relevance** | Minimal | Primary energy consumer |
| **Unit complexity** | GJ, MWh, MMBtu | ton-hours, kWh_th, TR, BTU, GJ (5+ units) |

### 1.2 Technical Complexity Justification

A cooling-only agent can properly implement:

1. **Multi-technology cooling plants**: Real district cooling plants combine
   electric chillers, absorption chillers, free cooling, and TES. Each
   technology has different COP curves, energy inputs, and emission factors.
   The combined plant emission factor must be weighted by each technology's
   contribution.

2. **Part-load efficiency modeling (IPLV/NPLV)**: Chillers rarely operate at
   full load. The AHRI-standard IPLV formula weights performance at 25%, 50%,
   75%, and 100% load points. This requires tracking load profiles and applying
   weighted COP values -- complexity that does not exist for steam boilers.

3. **Temporal emission shifting via TES**: Ice storage and chilled water TES
   shift cooling production to off-peak hours when grid carbon intensity may
   differ by 2-5x. This requires time-of-use grid emission factors and
   temporal allocation algorithms.

4. **Absorption cooling bridge domain**: Absorption chillers consume heat
   (steam, hot water, waste heat, direct-fired gas) to produce cooling. This
   bridges the thermal and electrical energy domains, requiring both heat-source
   EFs and cooling COP calculations simultaneously.

5. **Free cooling near-zero emissions**: Sea/lake/river water cooling and cold
   ambient air cooling have effective COPs of 15-30 (pump energy only),
   requiring different calculation pathways than mechanical refrigeration.

6. **Refrigerant leakage cross-reference**: District cooling systems use
   refrigerants (R-134a, R-1234ze, R-410A, R-717) that may leak, creating
   Scope 1 emissions. MRV-012 should flag and cross-reference with MRV-002
   (Refrigerants & F-Gas Agent).

---

## 2. Cooling Generation Technologies

### 2.1 Electric Chillers (Vapor Compression Cycle)

Electric chillers use a mechanical vapor compression cycle with an electric
motor driving a compressor. They are the dominant technology in district
cooling worldwide.

#### 2.1.1 Centrifugal Chillers

- **Capacity range**: 300-25,000 kW (85-7,000 tons)
- **Compressor type**: Single or multi-stage centrifugal impeller
- **Common refrigerants**: R-134a, R-1234ze(E), R-513A, R-1233zd(E)
- **Best application**: Large district cooling plants, base-load operation
- **Part-load behavior**: Excellent with VFD (variable frequency drive)

| Condition | COP Range | Default COP | kW/ton Range |
|-----------|-----------|-------------|-------------|
| Water-cooled, full load | 5.0-7.5 | 6.1 | 0.47-0.70 |
| Water-cooled, IPLV | 7.0-12.0 | 9.5 | 0.29-0.50 |
| Air-cooled, full load | 2.8-3.5 | 3.1 | 1.00-1.25 |
| Air-cooled, IPLV | 4.0-5.5 | 4.5 | 0.64-0.88 |

ASHRAE 90.1-2019 minimum efficiency for water-cooled centrifugal chillers
(>=300 tons): 0.560 kW/ton full load (COP 6.28), 0.519 IPLV (COP 6.77).

#### 2.1.2 Screw Chillers (Rotary Screw Compressor)

- **Capacity range**: 70-500 tons (250-1,750 kW)
- **Compressor type**: Twin rotary screw, slide valve unloading
- **Common refrigerants**: R-134a, R-1234ze(E), R-513A
- **Best application**: Medium plants, variable load
- **Part-load behavior**: Good with slide valve, moderate with VFD

| Condition | COP Range | Default COP | kW/ton Range |
|-----------|-----------|-------------|-------------|
| Water-cooled, full load | 4.0-5.8 | 4.7 | 0.61-0.88 |
| Water-cooled, IPLV | 5.5-8.5 | 6.8 | 0.41-0.64 |
| Air-cooled, full load | 2.5-3.2 | 2.9 | 1.10-1.40 |
| Air-cooled, IPLV | 3.5-5.0 | 4.2 | 0.70-1.00 |

ASHRAE 90.1-2019 minimum for water-cooled positive displacement >=300 tons:
0.6395 kW/ton full load (COP 5.50), 0.5722 IPLV (COP 6.15).

#### 2.1.3 Reciprocating Chillers (Piston Compressor)

- **Capacity range**: 5-150 tons (17-530 kW)
- **Compressor type**: Reciprocating piston, multi-cylinder
- **Common refrigerants**: R-134a, R-407C, R-410A
- **Best application**: Small plants, backup/peaking
- **Part-load behavior**: Stepped (cylinder unloading)

| Condition | COP Range | Default COP | kW/ton Range |
|-----------|-----------|-------------|-------------|
| Water-cooled, full load | 3.5-5.0 | 4.2 | 0.70-1.00 |
| Water-cooled, IPLV | 4.5-6.5 | 5.5 | 0.54-0.78 |
| Air-cooled, full load | 2.4-3.0 | 2.7 | 1.17-1.46 |
| Air-cooled, IPLV | 3.2-4.5 | 3.8 | 0.78-1.10 |

#### 2.1.4 Scroll Chillers

- **Capacity range**: 5-60 tons (17-210 kW)
- **Compressor type**: Orbital scroll compressor
- **Common refrigerants**: R-410A, R-32, R-454B
- **Best application**: Small buildings, supplemental cooling
- **Part-load behavior**: Moderate (VFD or on/off staging)

| Condition | COP Range | Default COP | kW/ton Range |
|-----------|-----------|-------------|-------------|
| Water-cooled, full load | 3.8-5.2 | 4.5 | 0.68-0.92 |
| Air-cooled, full load | 2.5-3.3 | 2.9 | 1.06-1.40 |

### 2.2 Absorption Chillers

Absorption chillers use a thermochemical process (LiBr/water or ammonia/water)
driven by heat instead of electricity. They consume steam, hot water, waste
heat, or direct-fired gas to produce cooling.

#### 2.2.1 Single-Effect Absorption

- **Heat source temperature**: >= 80 deg C (hot water) or 15 psig steam
- **Working pair**: LiBr/water (most common), NH3/water
- **Capacity range**: 10-1,500 tons
- **Parasitic electrical**: 0.01-0.02 kW/ton (pumps, controls)
- **Best for**: Low-grade waste heat recovery, CHP integration

| Heat Source | COP Range | Default COP |
|-------------|-----------|-------------|
| Hot water (85-95 deg C) | 0.60-0.75 | 0.68 |
| Low-pressure steam (100-120 deg C) | 0.65-0.80 | 0.72 |
| Exhaust gas (150-250 deg C) | 0.65-0.83 | 0.75 |
| Direct-fired (gas) | 0.80-0.95 | 0.88 |

#### 2.2.2 Double-Effect Absorption

- **Heat source temperature**: >= 140 deg C or 115 psig steam
- **Working pair**: LiBr/water
- **Capacity range**: 100-3,000 tons
- **Parasitic electrical**: 0.01-0.03 kW/ton
- **Best for**: Steam-rich industrial sites, CHP plants

| Heat Source | COP Range | Default COP |
|-------------|-----------|-------------|
| High-pressure steam (140-170 deg C) | 1.00-1.40 | 1.20 |
| Direct-fired (gas) | 1.10-1.50 | 1.30 |
| Exhaust gas (350-500 deg C) | 1.10-1.43 | 1.25 |

#### 2.2.3 Triple-Effect Absorption

- **Heat source temperature**: >= 200 deg C
- **Working pair**: LiBr/water (advanced cycle)
- **Capacity range**: 200-2,000 tons (limited availability)
- **Parasitic electrical**: 0.02-0.04 kW/ton
- **Best for**: High-temperature waste heat, combined cycle plants

| Heat Source | COP Range | Default COP |
|-------------|-----------|-------------|
| High-pressure steam (>200 deg C) | 1.50-1.80 | 1.60 |
| Direct-fired (gas) | 1.60-1.90 | 1.75 |

### 2.3 Free Cooling Systems

Free cooling exploits natural cold sources (water bodies, cold air) to provide
cooling with minimal energy input (pumps and heat exchangers only).

#### 2.3.1 Sea Water Cooling

- **Required water temperature**: < 8 deg C (full free cooling), 8-15 deg C (partial)
- **Effective COP**: 15-30 (pump energy only)
- **Emission reduction vs mechanical**: 50-70% (Copenhagen: 66% reduction)
- **Applicable regions**: Scandinavia, Northern Europe, Northern US/Canada, deep ocean
- **Example**: Copenhagen District Cooling uses harbor seawater at < 5.2 deg C
  for direct free cooling, saving 66% CO2 vs individual building chillers

#### 2.3.2 Lake/River Water Cooling

- **Required water temperature**: < 10 deg C (full), 10-16 deg C (partial)
- **Effective COP**: 12-25
- **Depth requirement**: Typically > 30m for year-round cold water
- **Example**: Toronto Enwave Deep Lake Water Cooling from Lake Ontario (4 deg C at 83m depth)
- **Example**: Cornell University Lake Source Cooling from Cayuga Lake

#### 2.3.3 Cold Ambient Air (Economizer / Airside Free Cooling)

- **Required ambient temperature**: < 10 deg C (full), 10-18 deg C (partial)
- **Effective COP**: 10-20 (fan energy only)
- **Best for**: Data centers, Northern climates
- **Hours available per year**: 2,000-6,000+ depending on climate

#### 2.3.4 Underground/Aquifer Thermal Energy Storage (ATES)

- **Water temperature**: 5-10 deg C from cold well
- **Effective COP**: 15-40
- **Best for**: Dutch-style aquifer systems
- **Example**: Netherlands has > 3,000 ATES systems

### 2.4 Thermal Energy Storage (TES)

TES systems store cooling capacity produced during off-peak hours for use
during peak demand periods.

#### 2.4.1 Ice Storage

- **Storage medium**: Ice at 0 deg C (latent heat: 334 kJ/kg)
- **Charge COP**: 3.0-4.0 (lower than normal chilling due to lower evaporator temp)
- **Discharge COP**: Effectively infinite (melting ice, pump energy only)
- **Overall cycle COP**: 2.5-3.5
- **Peak demand reduction**: 30-50% of peak chiller capacity
- **Emission impact**: Depends on off-peak vs peak grid carbon intensity
  - If nighttime grid EF is 30-50% lower: net 15-25% emission reduction
  - If nighttime grid EF is higher (wind-heavy grids): net emission increase

| Metric | Value |
|--------|-------|
| Storage density | 45-55 kWh_th/m3 |
| Charge temperature | -6 to -3 deg C |
| Discharge temperature | 1 to 4 deg C |
| Standby losses | 1-3% per day |
| Cycle efficiency | 85-95% (thermal) |

#### 2.4.2 Chilled Water Storage

- **Storage medium**: Water at 4-6 deg C (sensible heat)
- **Charge COP**: 4.5-6.5 (same as normal chilling)
- **Storage density**: 7-10 kWh_th/m3
- **Standby losses**: 0.5-2% per day
- **Cycle efficiency**: 90-97% (thermal)
- **Advantage**: No COP penalty during charging (unlike ice)

#### 2.4.3 Phase Change Material (PCM) Storage

- **Storage medium**: Eutectic salts, paraffins at various temperatures
- **Storage density**: 30-60 kWh_th/m3
- **Charge temperature**: Varies by PCM (typically 0-10 deg C)
- **Cycle efficiency**: 80-90%
- **Advantage**: Higher density than water, flexible temperature

### 2.5 Combined/Hybrid Cooling Plants

Real district cooling plants typically combine multiple technologies:

**Example: Typical Middle East District Cooling Plant (10,000 TR)**
- Base load: 4x 2,500 TR centrifugal chillers (COP 6.0)
- Peak shaving: 5,000 TR-hr ice TES
- Partial free cooling: None (hot climate)
- Weighted plant COP at design: 5.2

**Example: Typical Nordic District Cooling Plant (5,000 TR)**
- Base load: 2x 1,500 TR centrifugal chillers (COP 6.5)
- Free cooling: Sea water heat exchanger (COP 20, available 4,000 hrs/yr)
- Absorption: 1x 500 TR double-effect from waste heat (COP 1.2)
- Weighted annual plant COP: 8.5-12.0

---

## 3. Emission Calculation Formulas

### 3.1 Core Formula: Electric Cooling

```
Electrical_Input (kWh_e) = Cooling_Output (kWh_th) / COP

Emissions (kgCO2e) = Electrical_Input (kWh_e) x Grid_EF (kgCO2e/kWh_e)
```

Expanded with network losses and auxiliary energy:
```
Total_Cooling_Delivered (kWh_th) = Metered_Cooling / (1 - Distribution_Loss%)

Chiller_Output (kWh_th) = Total_Cooling_Delivered + TES_Standby_Loss

Electrical_Input (kWh_e) = Chiller_Output / COP_effective

Auxiliary_Input (kWh_e) = Pump_Energy + Cooling_Tower_Fan + Controls

Total_Electrical (kWh_e) = Electrical_Input + Auxiliary_Input

Emissions (kgCO2e) = Total_Electrical x Grid_EF
```

### 3.2 Core Formula: Absorption Cooling

```
Heat_Input (GJ) = Cooling_Output (GJ_th) / COP_absorption

Emissions (kgCO2e) = Heat_Input (GJ) x Heat_Source_EF (kgCO2e/GJ)
                     + Parasitic_Electrical (kWh) x Grid_EF (kgCO2e/kWh)
```

Where Heat_Source_EF depends on the heat source:
- **Natural gas boiler**: 56.1 kgCO2/GJ / boiler_efficiency
- **Steam from CHP**: Allocated CHP emissions (per CHP allocation method)
- **Waste heat**: 0 kgCO2e/GJ (no marginal emissions)
- **District heating**: DH_network_EF (kgCO2e/GJ)

### 3.3 Core Formula: Free Cooling

```
Pump_Energy (kWh_e) = Cooling_Output (kWh_th) / Effective_COP_free

Emissions (kgCO2e) = Pump_Energy (kWh_e) x Grid_EF (kgCO2e/kWh)
```

Where Effective_COP_free ranges from 12-30 depending on system type and
pumping distance.

### 3.4 Formula: Thermal Energy Storage (Temporal Shifting)

For TES systems, emissions must be calculated using the grid EF at the
time of CHARGING (energy consumption), not at the time of DISCHARGE
(cooling delivery):

```
Charge_Energy (kWh_e) = TES_Capacity (kWh_th) / COP_charge / Cycle_Efficiency

Emissions_TES (kgCO2e) = Charge_Energy x Grid_EF_charge_period

Emissions_per_kWh_th_delivered = Emissions_TES / (TES_Capacity x Discharge_Efficiency)
```

For time-of-use emission accounting:
```
Hourly_Emissions (kgCO2e) = Sum over h in [hours]:
    Cooling_from_chiller(h) / COP(h) x Grid_EF(h)
  + Cooling_from_TES_discharge(h) x TES_Emission_Intensity
  + Cooling_from_free(h) / COP_free x Grid_EF(h)
```

### 3.5 Formula: Combined/Hybrid Plant

For a plant with multiple cooling sources:
```
Total_Emissions = Sum over t in [technologies]:
    Cooling_Output_t / COP_t x EF_input_t x (1 + Aux_Fraction_t)

Plant_Emission_Factor (kgCO2e/kWh_th) = Total_Emissions / Total_Cooling_Delivered

Weighted_Plant_COP = Total_Cooling_Delivered / Total_Primary_Energy_Input
```

### 3.6 Formula: Part-Load (IPLV/NPLV)

AHRI Standard 550/590 defines IPLV as:
```
IPLV = 0.01 x A + 0.42 x B + 0.45 x C + 0.12 x D
```
Where:
- A = Efficiency at 100% load
- B = Efficiency at 75% load
- C = Efficiency at 50% load
- D = Efficiency at 25% load

For emission calculations using part-load:
```
COP_effective = IPLV_COP  (when annual/monthly average is appropriate)
```

Or for detailed hourly analysis:
```
COP_at_load = f(load_fraction, ambient_temp, condenser_water_temp)
```
The COP curve is typically modeled as a polynomial:
```
COP(PLR) = a0 + a1*PLR + a2*PLR^2 + a3*PLR^3
```
Where PLR = part load ratio (0.0 to 1.0), and coefficients a0-a3 are
chiller-specific.

### 3.7 Formula: Distribution Network Losses

```
Cooling_at_Plant = Cooling_at_Customer / (1 - Loss_Fraction)

Loss_Fraction = Thermal_Loss + Pump_Heat_Gain

Pump_Energy (kWh) = Flow_Rate (m3/s) x Pressure_Drop (Pa) / (Pump_Eff x Motor_Eff)
```

Typical distribution parameters:
| Parameter | District Cooling | District Heating |
|-----------|-----------------|-----------------|
| Supply temperature | 4-7 deg C | 70-120 deg C |
| Return temperature | 12-16 deg C | 40-70 deg C |
| Delta-T | 6-10 deg C | 30-50 deg C |
| Thermal loss | 1-3% | 5-15% |
| Pump energy as % of cooling | 3-8% | 1-3% |
| Total distribution overhead | 5-10% | 6-18% |

### 3.8 Formula: Refrigerant Leakage (Cross-Reference)

Refrigerant leakage from chillers is technically Scope 1 for the district
cooling operator, but may be Scope 3 for the purchaser. MRV-012 should
flag this for cross-reference with MRV-002:

```
Leakage_Emissions (kgCO2e) = Charge (kg) x Leak_Rate (%) x GWP

Annual_Leak_Rate typical: 2-10% for centrifugal, 0.5-5% for hermetic screw
```

Common chiller refrigerants and GWP:
| Refrigerant | Type | GWP (AR5) | GWP (AR6) | Phase-down Status |
|-------------|------|-----------|-----------|-------------------|
| R-134a | HFC | 1,430 | 1,530 | Being phased down |
| R-410A | HFC | 2,088 | 2,256 | Banned in new equipment (2024+) |
| R-407C | HFC | 1,774 | 1,908 | Being phased down |
| R-1234ze(E) | HFO | 7 | 7 | Preferred replacement |
| R-1234yf | HFO | 4 | <1 | Automotive primarily |
| R-1233zd(E) | HFO | 4.5 | 1 | Low-pressure chiller replacement |
| R-513A | HFO blend | 631 | 631 | Transitional |
| R-515B | HFO blend | 299 | 293 | Transitional |
| R-717 (Ammonia) | Natural | 0 | 0 | Industrial/district cooling |
| R-718 (Water) | Natural | 0 | 0 | Absorption chillers |
| R-744 (CO2) | Natural | 1 | 1 | Transcritical systems |

---

## 4. Unit Conversions

### 4.1 Cooling Energy Units

| From | To | Factor |
|------|----|--------|
| 1 ton-hour (refrigeration) | kWh_thermal | 3.5169 |
| 1 ton-hour (refrigeration) | BTU | 12,000 |
| 1 ton-hour (refrigeration) | MJ | 12.661 |
| 1 ton-hour (refrigeration) | GJ | 0.012661 |
| 1 kWh_thermal | BTU | 3,412.14 |
| 1 kWh_thermal | MJ | 3.6 |
| 1 GJ | kWh_thermal | 277.778 |
| 1 GJ | ton-hours | 78.963 |
| 1 GJ | MMBtu | 0.9478 |
| 1 MWh_thermal | GJ | 3.6 |
| 1 MMBtu | kWh_thermal | 293.071 |
| 1 Therm | kWh_thermal | 29.307 |
| 1 Frigorie | Wh_thermal | 1.163 |

### 4.2 Cooling Capacity (Power) Units

| From | To | Factor |
|------|----|--------|
| 1 ton (refrigeration, TR) | kW_thermal | 3.5169 |
| 1 ton (refrigeration, TR) | BTU/hr | 12,000 |
| 1 ton (refrigeration, TR) | kcal/hr | 3,024 |
| 1 kW_thermal | BTU/hr | 3,412.14 |
| 1 MW_thermal | TR | 284.35 |

### 4.3 Efficiency Conversions

| Conversion | Formula |
|-----------|---------|
| EER to COP | COP = EER / 3.41214 |
| COP to EER | EER = COP x 3.41214 |
| kW/ton to COP | COP = 3.5169 / (kW/ton) |
| COP to kW/ton | kW/ton = 3.5169 / COP |
| kW/ton to EER | EER = 12.0 / (kW/ton) |
| SEER to COP (approx) | COP = SEER / 3.41214 |

Examples:
- COP 6.0 = EER 20.47 = 0.586 kW/ton
- COP 4.5 = EER 15.35 = 0.782 kW/ton
- COP 1.2 (absorption) = EER 4.09 = 2.931 kW/ton
- COP 20.0 (free cooling) = EER 68.24 = 0.176 kW/ton

---

## 5. Emission Factors for Cooling

### 5.1 District Cooling System Emission Factors by Region

These are composite emission factors reflecting the typical technology mix
and grid carbon intensity in each region:

| Region | DC EF (kgCO2e/GJ_th) | DC EF (kgCO2e/kWh_th) | Typical Plant COP | Grid EF Used |
|--------|----------------------|----------------------|-------------------|-------------|
| UAE / Gulf States | 55-85 | 0.198-0.306 | 5.0-5.5 | 0.40-0.55 kgCO2e/kWh |
| Singapore | 40-60 | 0.144-0.216 | 5.5-6.5 | 0.408 kgCO2e/kWh |
| US Sun Belt (Texas) | 35-55 | 0.126-0.198 | 5.5-6.0 | 0.380 kgCO2e/kWh |
| US Sun Belt (California) | 18-30 | 0.065-0.108 | 5.5-6.0 | 0.225 kgCO2e/kWh |
| Japan | 30-50 | 0.108-0.180 | 5.0-6.0 | 0.465 kgCO2e/kWh |
| South Korea | 32-52 | 0.115-0.187 | 5.0-5.5 | 0.415 kgCO2e/kWh |
| India | 40-70 | 0.144-0.252 | 4.5-5.5 | 0.708 kgCO2e/kWh |
| China (coastal) | 35-65 | 0.126-0.234 | 5.0-5.5 | 0.555 kgCO2e/kWh |
| Germany | 25-40 | 0.090-0.144 | 5.5-6.5 | 0.338 kgCO2e/kWh |
| France | 8-15 | 0.029-0.054 | 5.5-6.5 | 0.056 kgCO2e/kWh |
| Nordic (with free cooling) | 3-10 | 0.011-0.036 | 8.0-15.0 | 0.020-0.050 kgCO2e/kWh |
| Global Default | 35-55 | 0.126-0.198 | 5.0-6.0 | 0.436 kgCO2e/kWh |

Derivation:
```
DC_EF (kgCO2e/kWh_th) = Grid_EF (kgCO2e/kWh_e) / Plant_COP
                         x (1 + Distribution_Loss%)
                         x (1 + Auxiliary_Energy%)
```

### 5.2 Technology-Specific Default COP Values

For implementation in the CoolingDatabaseEngine:

| Technology ID | Technology Name | COP_min | COP_max | COP_default | COP_IPLV | Energy Source | Condenser |
|---------------|----------------|---------|---------|-------------|----------|---------------|-----------|
| CENTRIFUGAL_WC | Centrifugal (water-cooled) | 5.0 | 7.5 | 6.1 | 9.5 | Electricity | Water |
| CENTRIFUGAL_AC | Centrifugal (air-cooled) | 2.8 | 3.5 | 3.1 | 4.5 | Electricity | Air |
| SCREW_WC | Screw (water-cooled) | 4.0 | 5.8 | 4.7 | 6.8 | Electricity | Water |
| SCREW_AC | Screw (air-cooled) | 2.5 | 3.2 | 2.9 | 4.2 | Electricity | Air |
| RECIPROCATING_WC | Reciprocating (water-cooled) | 3.5 | 5.0 | 4.2 | 5.5 | Electricity | Water |
| RECIPROCATING_AC | Reciprocating (air-cooled) | 2.4 | 3.0 | 2.7 | 3.8 | Electricity | Air |
| SCROLL_WC | Scroll (water-cooled) | 3.8 | 5.2 | 4.5 | 5.8 | Electricity | Water |
| SCROLL_AC | Scroll (air-cooled) | 2.5 | 3.3 | 2.9 | 4.0 | Electricity | Air |
| ABSORPTION_SINGLE | Absorption (single-effect) | 0.60 | 0.83 | 0.70 | N/A | Heat/Steam | Water |
| ABSORPTION_DOUBLE | Absorption (double-effect) | 1.00 | 1.50 | 1.20 | N/A | Heat/Steam | Water |
| ABSORPTION_TRIPLE | Absorption (triple-effect) | 1.50 | 1.90 | 1.60 | N/A | Heat/Steam | Water |
| FREE_SEAWATER | Free cooling (seawater) | 15.0 | 30.0 | 20.0 | N/A | Electricity (pumps) | N/A |
| FREE_LAKE | Free cooling (lake/river) | 12.0 | 25.0 | 18.0 | N/A | Electricity (pumps) | N/A |
| FREE_AIR | Free cooling (ambient air) | 10.0 | 20.0 | 15.0 | N/A | Electricity (fans) | N/A |
| FREE_ATES | Free cooling (aquifer TES) | 15.0 | 40.0 | 25.0 | N/A | Electricity (pumps) | N/A |
| TES_ICE | Ice thermal storage | 2.5 | 3.5 | 3.0 | N/A | Electricity | Water |
| TES_CHILLED_WATER | Chilled water storage | 4.5 | 6.5 | 5.5 | N/A | Electricity | Water |
| TES_PCM | Phase change material | 3.5 | 5.5 | 4.5 | N/A | Electricity | Water |

### 5.3 Heat Source Emission Factors for Absorption Chillers

| Heat Source | EF (kgCO2e/GJ_heat) | Notes |
|-------------|---------------------|-------|
| Natural gas boiler (eta=0.85) | 66.0 | 56.1 / 0.85 |
| Natural gas boiler (eta=0.92) | 61.0 | High-efficiency condensing |
| Fuel oil boiler (eta=0.82) | 90.4 | 74.1 / 0.82 |
| Coal boiler (eta=0.78) | 121.3 | 94.6 / 0.78 |
| Biomass boiler (eta=0.70) | 0.0* | Biogenic CO2 separate |
| Waste heat recovery | 0.0 | No marginal emissions |
| CHP steam (efficiency alloc) | Varies | Depends on CHP allocation |
| District heating network | 18-120 | Region-dependent |
| Electric boiler (grid) | Grid-dependent | Grid_EF / 0.98 |
| Geothermal | 0.0 | Zero-carbon heat |
| Solar thermal | 0.0 | Zero-carbon heat |

*Biomass CH4 and N2O are non-biogenic and reported separately.

### 5.4 Auxiliary Energy Factors

| Auxiliary Component | Energy as % of Cooling | Typical kW/TR |
|--------------------|----------------------|-------------|
| Condenser water pump | 2-4% | 0.03-0.06 |
| Chilled water pump (primary) | 2-3% | 0.02-0.04 |
| Chilled water pump (secondary/distribution) | 3-8% | 0.04-0.12 |
| Cooling tower fan | 2-4% | 0.03-0.06 |
| Controls and lighting | 0.5-1% | 0.01 |
| Total auxiliary (typical) | 8-15% | 0.12-0.28 |

---

## 6. Part-Load Efficiency (IPLV/NPLV) Detailed

### 6.1 IPLV Standard Conditions (AHRI 550/590)

| Load Point | % of Full Load | Condenser Water Temp (deg F) | Weight |
|-----------|---------------|------------------------------|--------|
| A | 100% | 85.0 | 0.01 (1%) |
| B | 75% | 75.0 | 0.42 (42%) |
| C | 50% | 65.0 | 0.45 (45%) |
| D | 25% | 65.0 | 0.12 (12%) |

The key insight is that chillers operate at 50-75% load 87% of the time.
At these partial loads, COP is typically 15-40% better than at full load
for centrifugal chillers with VFD, due to lower condenser water temperatures
and reduced compressor work.

### 6.2 NPLV (Non-Standard Part Load Value)

NPLV uses the same weighting formula but with actual site conditions:
- Actual condenser water temperatures at each load point
- Actual leaving chilled water temperature
- Actual fouling factors

### 6.3 Example: COP at Various Load Points (Centrifugal with VFD)

| Load % | COP (IPLV conditions) | kW/ton |
|--------|----------------------|--------|
| 100% | 6.10 | 0.577 |
| 75% | 8.50 | 0.414 |
| 50% | 10.20 | 0.345 |
| 25% | 8.80 | 0.400 |

IPLV = 0.01 x 6.10 + 0.42 x 8.50 + 0.45 x 10.20 + 0.12 x 8.80 = 9.26 COP

### 6.4 Absorption Chiller Part-Load Behavior

Absorption chillers have different part-load characteristics than electric
chillers. Their COP remains relatively flat across load ranges but decreases
slightly at very low loads. The IPLV formula is not standardized for
absorption chillers; instead, a flat COP is typically used, or a load-weighted
average based on the specific chiller's performance curve.

---

## 7. Thermal Energy Storage: Emission Timing

### 7.1 The Temporal Shifting Problem

TES creates a temporal disconnect between energy consumption (charging)
and cooling delivery (discharging). For accurate GHG accounting:

1. **Simple approach**: Use annual average grid EF for all calculations
2. **Monthly approach**: Use monthly average grid EF for the charging month
3. **Hourly approach**: Use hourly marginal grid EF at time of charging

### 7.2 Emission Impact Scenarios

**Scenario A: Coal-heavy grid (e.g., India, China)**
- Peak (daytime): Grid EF = 0.80 kgCO2e/kWh
- Off-peak (nighttime): Grid EF = 0.65 kgCO2e/kWh
- TES charging at night: 19% emission reduction from temporal shift

**Scenario B: Renewable-heavy grid (e.g., California)**
- Peak (evening): Grid EF = 0.35 kgCO2e/kWh
- Midday (solar): Grid EF = 0.10 kgCO2e/kWh
- TES charging at midday: 71% emission reduction from temporal shift

**Scenario C: Wind-heavy grid (e.g., Denmark)**
- Daytime (low wind): Grid EF = 0.30 kgCO2e/kWh
- Nighttime (high wind): Grid EF = 0.05 kgCO2e/kWh
- TES charging at night: 83% emission reduction from temporal shift

### 7.3 TES Efficiency Penalties

TES introduces thermal losses that partially offset temporal benefits:
| TES Type | Charge COP Penalty | Standby Loss/day | Round-Trip Eff |
|----------|-------------------|-------------------|----------------|
| Ice | -25 to -40% | 1-3% | 80-90% |
| Chilled Water | 0% | 0.5-2% | 90-97% |
| PCM | -5 to -15% | 1-3% | 80-90% |

---

## 8. District Cooling Network Characteristics

### 8.1 Network Typology

| Network Type | Pipe Length | Customers | Typical Capacity |
|-------------|------------|-----------|-----------------|
| Campus | 0.5-3 km | 5-20 buildings | 1,000-5,000 TR |
| Municipal (small) | 3-10 km | 20-100 buildings | 5,000-20,000 TR |
| Municipal (large) | 10-50 km | 100-500 buildings | 20,000-100,000 TR |
| Industrial park | 2-10 km | 5-30 facilities | 5,000-50,000 TR |
| Data center cluster | 0.5-5 km | 1-10 data centers | 5,000-50,000 TR |

### 8.2 Distribution Loss Parameters

| Parameter | Range | Default |
|-----------|-------|---------|
| Thermal loss (insulated pipe) | 1-3% | 2% |
| Pump heat gain | 2-5% | 3% |
| Total distribution overhead | 3-8% | 5% |
| Pump specific power | 15-40 W/TR | 25 W/TR |
| Distribution pipe delta-T | 6-10 deg C | 8 deg C |

### 8.3 Comparison: Cooling vs Heating Distribution

| Characteristic | District Cooling | District Heating |
|---------------|-----------------|-----------------|
| Supply temp | 4-7 deg C | 70-120 deg C |
| Return temp | 12-16 deg C | 40-70 deg C |
| Delta-T | 6-10 deg C | 30-50 deg C |
| Thermal loss driver | Pipe surface, ambient | Pipe surface, ambient |
| Thermal loss magnitude | 1-3% (small delta-T with ambient) | 5-15% (large delta-T with ambient) |
| Pump energy significance | HIGH (3-8% of cooling) | LOW (1-3% of heating) |
| Total overhead | 5-10% | 6-18% |
| Dominant loss mechanism | Pump energy | Thermal radiation |

This comparison illustrates a fundamental difference: cooling distribution
losses are dominated by pumping energy (electrical), while heating losses
are dominated by thermal radiation. This matters for emission calculations
because pump energy uses grid EFs while thermal losses use the thermal
energy EF.

---

## 9. Regional Cooling Demand Patterns

### 9.1 Middle East / Gulf States

- **Annual cooling hours**: 6,000-8,000 (year-round)
- **Peak design temperature**: 48-52 deg C outdoor
- **Dominant technology**: Electric centrifugal chillers
- **Market leader**: UAE (Tabreed: 1.2 million TR)
- **Grid EF**: 0.40-0.55 kgCO2e/kWh
- **Key challenge**: Very high ambient temperatures reduce chiller COP
- **Free cooling potential**: Minimal (seawater 25-35 deg C)

### 9.2 Southeast Asia

- **Annual cooling hours**: 7,000-8,760 (year-round, some 24/7)
- **Peak design temperature**: 35-38 deg C outdoor
- **Dominant technology**: Electric centrifugal + screw
- **Key markets**: Singapore (Marina Bay), Malaysia, Thailand
- **Grid EF**: 0.40-0.70 kgCO2e/kWh
- **Key challenge**: High humidity increases latent load
- **Free cooling potential**: Minimal (seawater 28-32 deg C)

### 9.3 US Sun Belt

- **Annual cooling hours**: 3,000-5,000
- **Peak design temperature**: 38-45 deg C outdoor
- **Dominant technology**: Electric centrifugal + ice TES
- **Key markets**: Texas, Arizona, Florida, Las Vegas
- **Grid EF**: 0.22-0.45 kgCO2e/kWh (varies by subregion)
- **Key challenge**: Peak demand coincides with grid stress
- **Free cooling potential**: Limited (winter only)

### 9.4 Data Centers (Global)

- **Annual cooling hours**: 8,760 (24/7/365)
- **Cooling load**: 30-70% of total data center energy
- **Dominant technology**: Centrifugal + free cooling + liquid cooling
- **Key trend**: PUE targets driving efficiency (PUE 1.1-1.3 target)
- **Free cooling potential**: High in Northern climates (2,000-6,000 hrs)
- **TES integration**: Growing for demand response and grid services

### 9.5 Northern Europe / Nordics

- **Annual cooling hours**: 1,000-3,000
- **Peak design temperature**: 28-35 deg C outdoor
- **Dominant technology**: Free cooling (sea/lake) + electric backup
- **Key examples**: Copenhagen (HOFOR), Stockholm (Fortum Varme)
- **Grid EF**: 0.02-0.20 kgCO2e/kWh
- **Free cooling potential**: Very high (4,000-6,000 hrs/yr)
- **Copenhagen case**: 66% CO2 reduction via seawater free cooling

---

## 10. Regulatory Framework Requirements

### 10.1 GHG Protocol Scope 2 Guidance (2015)

**Relevant chapters**: 3 (boundary), 6 (calculation), 7 (instruments)

Requirements for purchased cooling:
- Cooling is explicitly listed as a Scope 2 source alongside electricity,
  steam, and heat
- Dual reporting required: location-based AND market-based
- Location-based: Use grid-average or default cooling EFs
- Market-based: Use supplier-specific EFs or contractual instrument EFs
- If supplier-specific EF unavailable, use residual mix factors
- Activity data: Metered cooling consumption in energy units (GJ, MWh)

**Cooling-specific guidance**:
- For district cooling from electric chillers, calculate emissions using
  electricity EF divided by plant COP
- For absorption cooling, calculate using heat source EF and absorption COP
- Network losses should be included or documented
- CHP-derived cooling requires allocation per GHG Protocol CHP guidance

### 10.2 ISO 14064-1:2018

- Category 2 (indirect energy emissions) covers purchased cooling
- Requires documentation of quantification methodology
- Requires uncertainty assessment
- Base year recalculation triggers include changes in cooling technology,
  supplier, or COP

### 10.3 CSRD / ESRS E1 (Climate Change)

**Disclosure requirements for cooling**:
- E1-5: Total energy consumption and mix, including cooling energy purchased
- E1-6: Gross Scope 2 GHG emissions (location-based and market-based)
- Cooling must be reported separately from electricity where material
- District cooling consumption must be disaggregated by source where known
- Emission factors and methodology must be disclosed

**ESRS E1 paragraphs relevant to cooling**:
- AR 37-39: Scope 2 calculation methodology
- AR 44: Disaggregation of energy sources
- DR E1-5, paragraph 37: "energy consumption from... cooling"

### 10.4 CDP Climate Change Questionnaire

- C8.2a: Scope 2 location-based (includes cooling)
- C8.2b: Scope 2 market-based (includes cooling)
- C8.2e: Country/area-specific EFs including cooling
- C-CN8.1/C-RE8.1: Sector-specific cooling disclosures
- Cooling must be listed as a separate energy type where material

### 10.5 SBTi Corporate Manual

- Scope 2 targets must cover all Scope 2 sources including cooling
- Target boundary must include purchased cooling
- 1.5 deg C pathway requires 4.2% annual reduction in Scope 2
- Cooling efficiency improvements count toward target progress

### 10.6 ASHRAE Standard 90.1

Not a GHG reporting standard, but relevant for:
- Minimum chiller efficiency requirements (Table 6.8.1-3)
- Baseline COP values for energy modeling
- Mandatory compliance in most US jurisdictions
- Informs "best available technology" COP benchmarks

### 10.7 EU Energy Efficiency Directive (2023/1791)

- Article 23-24: District cooling networks must report efficiency
- District cooling providers must disclose emission factors to customers
- CHP allocation methods for trigeneration (heat + power + cooling)
- Cooling EF disclosure enables Scope 2 reporting by customers

### 10.8 EU F-Gas Regulation (2024/573)

- Applies to refrigerants used in district cooling chillers
- HFC phase-down schedule affects chiller refrigerant choices
- Leak detection and reporting requirements
- Cross-reference needed with MRV-002 for refrigerant leakage

### 10.9 Green Claims Directive (proposed)

- "Carbon neutral cooling" claims must be substantiated
- Cooling-related environmental claims need verified emission data
- Product/service environmental footprint for cooling services

---

## 11. Proposed Agent Architecture

### 11.1 Seven-Engine Architecture

Based on the research findings, the recommended architecture for MRV-012
follows the established 7-engine pattern:

```
Engine 1: CoolingDatabaseEngine
  - 18 cooling technology COP profiles (full/IPLV, by condenser type)
  - Regional DC emission factors (12 regions)
  - Heat source EFs for absorption (11 heat sources)
  - Refrigerant GWP data (11 refrigerants)
  - Part-load COP curves (polynomial coefficients)
  - Unit conversions (ton-hr, kWh_th, GJ, BTU, TR)
  - Auxiliary energy factors
  - Distribution loss parameters

Engine 2: ElectricCoolingCalculatorEngine
  - Vapor compression cycle emissions (COP-based)
  - Part-load IPLV/NPLV weighted calculations
  - COP curve interpolation (polynomial or tabular)
  - Condenser type adjustment (water-cooled vs air-cooled)
  - Auxiliary energy addition (pumps, fans, controls)
  - Per-gas breakdown (CO2, CH4, N2O via grid EF decomposition)

Engine 3: AbsorptionCoolingCalculatorEngine
  - Single/double/triple-effect COP calculations
  - Heat source EF resolution (gas, steam, waste heat, CHP, DH)
  - Parasitic electrical energy addition
  - Waste heat zero-marginal-emission pathway
  - CHP allocation integration (link to MRV-011 or standalone)
  - Biogenic CO2 separation for biomass heat sources

Engine 4: FreeCoolingCalculatorEngine
  - Seawater/lake/river/air/ATES free cooling
  - Effective COP based on pump/fan energy
  - Temperature-dependent availability (hours per year)
  - Hybrid mode (partial free + mechanical backup)
  - Near-zero emission pathway documentation

Engine 5: ThermalStorageEngine
  - Ice/chilled-water/PCM TES modeling
  - Charge COP penalty calculation
  - Temporal emission shifting (peak vs off-peak grid EF)
  - Round-trip efficiency and standby losses
  - Time-of-use emission factor integration
  - Hourly, monthly, or annual averaging modes

Engine 6: CoolingPlantAggregatorEngine
  - Multi-technology plant combining (weighted COP)
  - Distribution network loss adjustment
  - Plant-level emission factor derivation
  - Uncertainty quantification (Monte Carlo, analytical)
  - Compliance checking (9 regulatory frameworks)
  - Refrigerant leakage cross-reference flagging

Engine 7: CoolingPurchasePipelineEngine
  - Full pipeline orchestration (DB->Calc->Agg->Uncertainty->Compliance)
  - Batch processing
  - Provenance chain (SHA-256)
  - Time-series aggregation (facility/technology/period/supplier)
  - Location-based AND market-based dual reporting
```

### 11.2 Key Differences from MRV-011 Engine Structure

| MRV-011 Engine | MRV-012 Engine | Difference |
|---------------|---------------|------------|
| SteamHeatDatabaseEngine | CoolingDatabaseEngine | 18 technologies vs 14 fuels; COP curves vs boiler eta |
| SteamEmissionsCalculatorEngine | ElectricCoolingCalculatorEngine | COP/IPLV/NPLV vs boiler efficiency; part-load curves |
| HeatCoolingCalculatorEngine | AbsorptionCoolingCalculatorEngine | Dedicated absorption focus vs generic DH |
| CHPAllocationEngine | FreeCoolingCalculatorEngine | Entirely new domain (no equivalent in MRV-011) |
| -- | ThermalStorageEngine | Entirely new (temporal shifting has no MRV-011 analog) |
| UncertaintyQuantifierEngine | CoolingPlantAggregatorEngine | Multi-tech aggregation + uncertainty combined |
| ComplianceCheckerEngine | (within Aggregator) | 9 frameworks vs 7 |
| SteamHeatPipelineEngine | CoolingPurchasePipelineEngine | Similar orchestration pattern |

### 11.3 Proposed Enumerations (22)

| Enum | Values | Description |
|------|--------|-------------|
| CoolingTechnology | 18 values | All cooling technology types |
| CondenserType | WATER_COOLED, AIR_COOLED, EVAPORATIVE | Condenser/heat rejection type |
| CompressorType | CENTRIFUGAL, SCREW, RECIPROCATING, SCROLL | Electric chiller compressor type |
| AbsorptionEffect | SINGLE, DOUBLE, TRIPLE | Absorption chiller effect level |
| AbsorptionWorkingPair | LIBR_WATER, NH3_WATER | Absorption working fluid pair |
| FreeCoolingSource | SEAWATER, LAKE, RIVER, AMBIENT_AIR, ATES | Natural cold source |
| TESType | ICE, CHILLED_WATER, PCM | Thermal storage medium |
| TESMode | CHARGING, DISCHARGING, STANDBY, BYPASS | TES operating mode |
| HeatSourceType | NATURAL_GAS, FUEL_OIL, COAL, BIOMASS, WASTE_HEAT, CHP, DH_NETWORK, ELECTRIC, GEOTHERMAL, SOLAR_THERMAL, BIOGAS | Heat source for absorption |
| CoolingEnergyUnit | KWH_TH, TON_HOUR, GJ, MJ, BTU, MMBTU, THERM, MWH_TH, FRIGORIE | Cooling energy units |
| CoolingCapacityUnit | KW_TH, TR, BTU_HR, KCAL_HR, MW_TH | Cooling capacity (power) units |
| EfficiencyMetric | COP, EER, KW_PER_TON, IPLV, NPLV, SEER | Efficiency expression types |
| LoadPoint | FULL_LOAD, THREE_QUARTER, HALF_LOAD, QUARTER_LOAD | IPLV load points |
| EmissionGas | CO2, CH4, N2O, CO2E, BIOGENIC_CO2 | GHG gases |
| GWPSource | AR4, AR5, AR6, AR6_20YR | IPCC assessment report |
| Scope2Method | LOCATION_BASED, MARKET_BASED | GHG Protocol Scope 2 methods |
| DataQualityTier | TIER_1, TIER_2, TIER_3 | Data quality tier |
| ComplianceFramework | GHG_PROTOCOL, ISO_14064, CSRD_ESRS_E1, CDP, SBTI, ASHRAE_90_1, EU_EED, EU_FGAS, GREEN_CLAIMS | Regulatory frameworks |
| ComplianceStatus | COMPLIANT, NON_COMPLIANT, PARTIAL, NOT_APPLICABLE | Compliance result |
| NetworkType | CAMPUS, MUNICIPAL_SMALL, MUNICIPAL_LARGE, INDUSTRIAL, DATA_CENTER | DC network types |
| FacilityType | COMMERCIAL, INDUSTRIAL, DATA_CENTER, INSTITUTIONAL, RESIDENTIAL, CAMPUS, MIXED | Facility types |
| PlantOperatingMode | ELECTRIC_ONLY, ABSORPTION_ONLY, HYBRID, FREE_COOLING, TES_DISCHARGE | Plant operating mode |

### 11.4 Proposed Database Schema (V063)

Target: 16 tables, 3 hypertables, 2 continuous aggregates

| Table | Description | Type |
|-------|-------------|------|
| `cp_cooling_technologies` | 18 cooling technology COP profiles | Dimension |
| `cp_cop_curves` | Part-load COP polynomial coefficients | Dimension |
| `cp_regional_dc_factors` | Regional DC composite EFs | Dimension |
| `cp_heat_source_factors` | Heat source EFs for absorption | Dimension |
| `cp_refrigerant_data` | Refrigerant GWP and phase-down data | Dimension |
| `cp_auxiliary_factors` | Auxiliary energy percentages | Dimension |
| `cp_distribution_parameters` | Network loss parameters | Dimension |
| `cp_facilities` | Facility registry | Dimension |
| `cp_cooling_suppliers` | District cooling supplier info | Dimension |
| `cp_supplier_emission_factors` | Supplier-specific EFs | Dimension |
| `cp_calculations` | Calculation results | Hypertable |
| `cp_calculation_details` | Per-gas and per-technology breakdown | Regular |
| `cp_tes_events` | TES charge/discharge events | Hypertable |
| `cp_compliance_checks` | Compliance check results | Regular |
| `cp_batch_jobs` | Batch processing jobs | Regular |
| `cp_aggregations` | Aggregated results | Hypertable |
| `cp_hourly_stats` | Hourly calculation stats | Continuous Aggregate |
| `cp_daily_stats` | Daily calculation stats | Continuous Aggregate |

### 11.5 Proposed API Endpoints (24)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/calculate/electric` | Electric chiller cooling emissions |
| POST | `/calculate/absorption` | Absorption cooling emissions |
| POST | `/calculate/free-cooling` | Free cooling emissions |
| POST | `/calculate/tes` | TES-shifted cooling emissions |
| POST | `/calculate/plant` | Combined multi-tech plant emissions |
| POST | `/calculate/batch` | Batch calculation |
| GET | `/technologies` | List all cooling technologies |
| GET | `/technologies/{tech_id}` | Get technology COP profile |
| GET | `/technologies/{tech_id}/cop-curve` | Get part-load COP curve |
| GET | `/factors/regional/{region}` | Get regional DC emission factor |
| GET | `/factors/heat-source/{source}` | Get heat source EF |
| GET | `/factors/refrigerant/{refrigerant}` | Get refrigerant GWP |
| GET | `/factors/auxiliary` | Get auxiliary energy factors |
| GET | `/factors/distribution/{network_type}` | Get distribution parameters |
| POST | `/facilities` | Register facility |
| GET | `/facilities/{facility_id}` | Get facility |
| POST | `/suppliers` | Register cooling supplier |
| GET | `/suppliers/{supplier_id}` | Get supplier info |
| POST | `/uncertainty` | Run uncertainty analysis |
| POST | `/compliance/check` | Run compliance check |
| GET | `/compliance/frameworks` | List supported frameworks |
| POST | `/aggregate` | Aggregate results |
| GET | `/calculations/{calc_id}` | Get calculation result |
| GET | `/health` | Health check |

### 11.6 Integration Points

| Integration | Direction | Purpose |
|-------------|-----------|---------|
| MRV-009 (Scope 2 Location) | Input | Grid EFs for electric cooling |
| MRV-010 (Scope 2 Market) | Input | Market EFs, RECs/GOs for cooling |
| MRV-011 (Steam/Heat) | Input | CHP allocation for absorption heat; DH EFs |
| MRV-002 (Refrigerants) | Cross-ref | Refrigerant leakage flagging |
| AGENT-FOUND-001 (Orchestrator) | Output | DAG pipeline integration |
| AGENT-FOUND-010 (Observability) | Output | Metrics and tracing |
| AGENT-DATA-001..020 | Input | Data quality and lineage |

---

## 12. Acceptance Criteria Summary

- [ ] 18 cooling technology COP profiles (full load + IPLV)
- [ ] 12+ regional DC composite emission factors
- [ ] 11 heat source EFs for absorption chillers
- [ ] 11 refrigerant GWP values (AR5 and AR6)
- [ ] Part-load COP curve modeling (polynomial + tabular)
- [ ] IPLV/NPLV calculation per AHRI 550/590
- [ ] Free cooling (4 sources) with effective COP
- [ ] TES temporal emission shifting (ice, chilled water, PCM)
- [ ] Time-of-use grid EF integration for TES
- [ ] Multi-technology plant aggregation with weighted COP
- [ ] Distribution network loss adjustment
- [ ] Auxiliary energy (pumps, fans, controls) addition
- [ ] Absorption cooling with 3 effect levels
- [ ] Heat source resolution (11 types) for absorption
- [ ] Waste heat zero-marginal pathway
- [ ] Biogenic CO2 separation
- [ ] Refrigerant leakage cross-reference with MRV-002
- [ ] Location-based AND market-based dual reporting
- [ ] Tier 1/2/3 calculation support
- [ ] 9 regulatory framework compliance checks
- [ ] 9+ cooling energy unit conversions
- [ ] 6 efficiency metric conversions (COP/EER/kW-per-ton/IPLV/NPLV/SEER)
- [ ] Monte Carlo uncertainty (10,000 iterations)
- [ ] 24 REST API endpoints
- [ ] V063 database migration (16 tables, 3 hypertables, 2 CAs)
- [ ] 1,000+ unit tests with 85%+ coverage
- [ ] SHA-256 provenance on every result
- [ ] Decimal arithmetic throughout
- [ ] Thread-safe singleton pattern
- [ ] Auth integration (route_protector.py + auth_setup.py)

---

## 13. Research Sources

- GHG Protocol Scope 2 Guidance (2015) -- https://ghgprotocol.org/scope-2-guidance
- GHG Protocol Scope 2 Standard Development Plan (2024) -- https://ghgprotocol.org/sites/default/files/2025-01/S2-SDP-20241220.pdf
- US EPA Scope 1 and 2 Inventory Guidance -- https://www.epa.gov/climateleadership/scope-1-and-scope-2-inventory-guidance
- US DOE CHP Technologies: Absorption Chillers -- https://betterbuildingssolutioncenter.energy.gov/sites/default/files/attachments/CHP_Absorption_Chillers.pdf
- ASHRAE Standard 90.1-2019 -- https://www.ashrae.org/technical-resources/bookstore/standard-90-1
- ASHRAE 90.1-2019 Addendum x (chiller efficiency) -- https://www.ashrae.org/file%20library/technical%20resources/standards%20and%20guidelines/standards%20addenda/90_1_2019_x_20220204.pdf
- AHRI Standard 550/590 (chiller performance rating) -- https://www.ahrinet.org
- Coefficient of Performance (Wikipedia) -- https://en.wikipedia.org/wiki/Coefficient_of_performance
- EER/COP/IPLV/NPLV Differences -- https://www.gesonchiller.com/understanding-the-key-differences-between-cop-eer-apf-seer-iplv-and-nplv-in-air-conditioning-heat-pump-chillers/
- How to Calculate Chiller IPLV -- https://mepacademy.com/how-to-calculate-chiller-iplv/
- Copenhagen Seawater District Cooling (Danfoss) -- https://www.danfoss.com/en/service-and-support/case-stories/dds/seawater-cools-copenhagen-city-cutting-emissions-by-70/
- Alfa Laval Free Cooling -- https://www.alfalaval.com/media/stories/district-cooling/cooling-naturally/
- IEA Heat Pumps in District Systems -- https://www.iea.org/articles/heat-pumps-in-district-heating-and-cooling-systems
- IEA Emissions Factors 2025 Database Documentation -- https://iea.blob.core.windows.net/assets/2b5f6d31-3263-44bf-85bc-b754d1c69cd3/IEA_Methodology_Emission_Factors.pdf
- Chiller Refrigerant Regulations -- https://www.achrnews.com/articles/144913-chiller-refrigerant-choices-are-evolving-with-new-regulations
- US DOE Ice Storage Systems -- https://www.energy.gov/eere/buildings/articles/ice-storage-and-other-thermal-storage-related-systems
- WEF Thermal Energy Storage -- https://www.weforum.org/stories/2021/09/thermal-energy-storage-air-conditioning-renewables/
- District Cooling Market (Straits Research) -- https://straitsresearch.com/report/district-cooling-market
- ESRS E1 Climate Reporting -- https://climateseed.com/blog/esrs-e1-understanding-the-csrd-climate-standard
- EU Energy Efficiency Directive -- https://energy.ec.europa.eu/topics/energy-efficiency/energy-efficiency-targets-directive-and-rules/energy-efficiency-directive_en
- US DOE Purchasing Energy-Efficient Chillers -- https://www.energy.gov/femp/purchasing-energy-efficient-electric-chillers
- DBDH District Cooling Guide -- https://dbdh.org/all-about-district-energy/cooling/
- Climatiq Emission Factor Database -- https://www.climatiq.io
- EU JRC District Heating/Cooling Consumers Report -- https://publications.jrc.ec.europa.eu/repository/bitstream/JRC132057/JRC132057_01.pdf

---

## 14. Relationship to MRV-011 (Steam/Heat Purchase Agent)

### 14.1 Scope Boundary Clarification

With the introduction of MRV-012, the scope of MRV-011 should be narrowed:

| Source | MRV-011 (Steam/Heat) | MRV-012 (Cooling) |
|--------|---------------------|-------------------|
| Purchased steam | YES | NO |
| District heating | YES | NO |
| CHP allocation (for heat) | YES | For absorption heat input only |
| District cooling (electric) | NO (remove) | YES |
| District cooling (absorption) | NO (remove) | YES |
| Free cooling | NO | YES |
| Thermal energy storage | NO | YES |
| Cooling COP/EER/IPLV | NO | YES |
| Refrigerant leakage flag | NO | YES (cross-ref MRV-002) |

### 14.2 Migration Path

If MRV-011 has already been built with cooling included in Engine 3
(HeatCoolingCalculatorEngine), the migration path is:
1. Extract cooling logic from MRV-011 Engine 3 into MRV-012
2. Expand with dedicated engines (free cooling, TES, part-load)
3. MRV-011 Engine 3 becomes purely HeatCalculatorEngine
4. MRV-012 links to MRV-011 for absorption heat source EFs

---

*Research compiled: 2026-02-22*
*For: AGENT-MRV-012 Cooling Purchase Agent PRD development*
*Status: Complete -- ready for PRD authoring*
