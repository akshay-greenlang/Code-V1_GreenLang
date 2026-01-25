# EMISSION FACTORS DATA SOURCES

## ⚠️ DISCLAIMER

**THIS IS DEMO/SYNTHETIC DATA FOR ILLUSTRATION PURPOSES ONLY.**

These emission factors are derived from publicly available sources and are NOT official EU CBAM default values. For actual CBAM Transitional Registry filings, you MUST use the official default values published by the European Commission at:

**Official Source:** https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism_en

---

## DATA SOURCES BY PRODUCT GROUP

### 1. CEMENT & CEMENT CLINKER

**Primary Source:** IEA (International Energy Agency) - Cement Technology Roadmap 2018

- **URL:** https://www.iea.org/reports/technology-roadmap-low-carbon-transition-in-the-cement-industry
- **Vintage:** 2018
- **Methodology:** Life Cycle Assessment (LCA) - cradle-to-gate
- **Scope:** Process emissions (calcination) + fuel combustion + electricity
- **Values Used:**
  - Portland Cement (Grey): 0.900 tCO2e/ton (0.766 direct + 0.134 indirect)
  - Portland Cement (White): 1.005 tCO2e/ton (0.855 direct + 0.150 indirect)
  - Cement Clinker: 0.950 tCO2e/ton (0.850 direct + 0.100 indirect)

**Secondary Source:** IPCC 2006 Guidelines Volume 3, Chapter 2
- **URL:** https://www.ipcc-nggip.iges.or.jp/public/2006gl/
- **Used For:** Clinker emission factors cross-validation

**Notes:**
- Process emissions from limestone (CaCO3) calcination: ~0.525 tCO2/ton clinker
- Fuel combustion (coal, petcoke, alternative fuels): ~0.3-0.4 tCO2/ton
- Electricity (grinding): ~0.1-0.15 tCO2/ton (grid-dependent)
- White cement requires more energy due to purity requirements (+10-15%)

---

### 2. STEEL & IRON PRODUCTS

**Primary Source:** World Steel Association - Steel Climate Impact Report 2023

- **URL:** https://worldsteel.org/steel-topics/climate-change/
- **Vintage:** 2023
- **Methodology:** Industry-wide average based on member data
- **Scope:** Ironmaking + Steelmaking + Rolling (where applicable)
- **Values Used:**
  - Steel (Basic Oxygen Furnace - BOF): 2.000 tCO2e/ton (1.850 direct + 0.150 indirect)
  - Steel (Electric Arc Furnace - EAF): 0.800 tCO2e/ton (0.385 direct + 0.415 indirect)
  - Hot-rolled Steel: 2.150 tCO2e/ton (1.950 direct + 0.200 indirect)
  - Pig Iron: 1.900 tCO2e/ton (1.800 direct + 0.100 indirect)

**Secondary Source:** IPCC 2019 Refinement to 2006 Guidelines
- **URL:** https://www.ipcc-nggip.iges.or.jp/public/2019rf/
- **Used For:** Process emissions validation

**Notes:**
- **BOF Route (Primary Steel):** Iron ore → Blast furnace (pig iron) → BOF → Steel
  - High direct emissions from coke consumption in blast furnace
  - ~70% of global steel production
- **EAF Route (Secondary Steel):** Scrap → Electric Arc Furnace → Steel
  - Lower direct emissions (no ironmaking)
  - Higher indirect emissions (electricity-intensive)
  - ~30% of global steel production
- Grid carbon intensity significantly affects EAF emissions (0.2-0.8 tCO2e/ton range)

---

### 3. ALUMINUM

**Primary Source:** IAI (International Aluminum Institute) - GHG Emissions Protocol 2023

- **URL:** https://international-aluminium.org/statistics/greenhouse-gas-emissions/
- **Vintage:** 2023
- **Methodology:** Industry reporting standard (IAI GHG Protocol)
- **Scope:** Alumina refining + Aluminum smelting (global average)
- **Values Used:**
  - Primary Aluminum (Unwrought): 11.500 tCO2e/ton (1.700 direct + 9.800 indirect)
  - Aluminum Alloys (Unwrought): 11.100 tCO2e/ton (1.600 direct + 9.500 indirect)
  - Secondary Aluminum (Recycled): 0.600 tCO2e/ton (0.350 direct + 0.250 indirect)

**Notes:**
- **Highly Electricity-Intensive:** ~15,000 kWh per ton of aluminum
- **Direct Emissions:** Anode consumption (carbon) + PFC emissions (CF4, C2F6)
- **Indirect Emissions:** Electricity generation (varies by grid: 3-20 tCO2e/ton range)
  - Coal-heavy grids: ~18-20 tCO2e/ton
  - Hydro-heavy grids: ~4-6 tCO2e/ton
  - Global average: ~9.8 tCO2e/ton
- **Recycling Benefit:** Secondary aluminum saves ~95% energy vs primary

---

### 4. FERTILIZERS

**Primary Source:** IPCC 2019 Refinement, Volume 3, Chapter 3

- **URL:** https://www.ipcc-nggip.iges.or.jp/public/2019rf/
- **Vintage:** 2019
- **Methodology:** Default emission factors for fertilizer production
- **Scope:** Process emissions + fuel combustion
- **Values Used:**
  - Ammonia (Anhydrous): 2.500 tCO2e/ton (2.200 direct + 0.300 indirect)
  - Urea: 1.700 tCO2e/ton (1.500 direct + 0.200 indirect)
  - Nitric Acid: 1.050 tCO2e/ton (0.900 direct + 0.150 indirect)

**Secondary Sources:**
- FAO (Food and Agriculture Organization) - Fertilizer Use Statistics
- EFMA (European Fertilizer Manufacturers Association)
- **URL:** https://www.fertilizerseurope.com/

**Notes:**
- **Ammonia (Haber-Bosch Process):**
  - Natural gas is both feedstock (H2 source) and fuel
  - Modern plants: 1.6-2.0 tCO2/ton
  - Older plants: 2.5-3.5 tCO2/ton
  - Blue ammonia (with CCS): 0.2-0.5 tCO2/ton
- **Nitric Acid:**
  - Includes N2O emissions (GWP = 298 × CO2)
  - N2O abatement technology can reduce emissions by 70-90%
- **Urea:**
  - Includes CO2 from ammonia feedstock
  - ~0.73 tCO2 released during urea synthesis (later reabsorbed in soil - not counted in CBAM)

---

### 5. HYDROGEN

**Primary Source:** IEA - Global Hydrogen Review 2023

- **URL:** https://www.iea.org/reports/global-hydrogen-review-2023
- **Vintage:** 2023
- **Methodology:** Technology-specific emission factors
- **Scope:** Production emissions (cradle-to-gate)
- **Values Used:**
  - Grey Hydrogen (SMR without CCS): 11.000 tCO2e/ton (10.000 direct + 1.000 indirect)

**Notes:**
- **Grey Hydrogen:** Steam Methane Reforming (SMR) from natural gas without carbon capture
- **Blue Hydrogen:** SMR with CCS - 1.0-2.5 tCO2e/ton (90% lower)
- **Green Hydrogen:** Electrolysis with renewable electricity - 0.0-0.5 tCO2e/ton
- **CBAM Scope:** Currently hydrogen is included; green hydrogen has competitive advantage

---

## UNCERTAINTY & LIMITATIONS

### Sources of Uncertainty

1. **Technology Variation:**
   - Different plants use different technologies with varying efficiencies
   - Example: Steel BOF emissions range from 1.6 to 2.5 tCO2/ton

2. **Fuel Mix:**
   - Coal vs natural gas vs biomass
   - Example: Cement kilns vary by fuel type

3. **Grid Carbon Intensity:**
   - Electricity-intensive processes (aluminum, EAF steel) vary widely by location
   - Example: EU average ~0.3 tCO2/kWh, China ~0.6 tCO2/kWh, Norway ~0.01 tCO2/kWh

4. **System Boundaries:**
   - Cradle-to-gate vs cradle-to-customer
   - Upstream emissions (mining, transport) may or may not be included

5. **Vintage of Data:**
   - Older data may not reflect modern efficiency improvements
   - Our data ranges from 2006-2023 vintage

### Uncertainty Ranges (Estimated)

| Product Group | Typical Uncertainty |
|---------------|---------------------|
| Cement | ±15% |
| Steel (BOF) | ±20% |
| Steel (EAF) | ±25% (grid-dependent) |
| Aluminum | ±35% (highly grid-dependent) |
| Fertilizers | ±20-25% |
| Hydrogen | ±30% |

---

## COMPARISON WITH EXPECTED EU CBAM DEFAULTS

Based on public consultation documents and draft regulations, we expect official EU CBAM default values to be:

| Product | Our Demo Value | Expected EU Default | Difference |
|---------|----------------|---------------------|------------|
| Cement | 0.900 tCO2/ton | 0.85-0.95 tCO2/ton | Within range |
| Steel (BOF) | 2.000 tCO2/ton | 1.9-2.1 tCO2/ton | Within range |
| Steel (EAF) | 0.800 tCO2/ton | 0.7-0.9 tCO2/ton | Within range |
| Aluminum | 11.500 tCO2/ton | 10-12 tCO2/ton | Within range |
| Ammonia | 2.500 tCO2/ton | 2.3-2.7 tCO2/ton | Within range |

**Conclusion:** Our demo values are reasonable approximations for illustration purposes.

---

## METHODOLOGY NOTES

### Direct Emissions (Scope 1)
- On-site fossil fuel combustion
- Process emissions (e.g., limestone calcination in cement)
- Fugitive emissions (e.g., PFCs in aluminum smelting)

### Indirect Emissions (Scope 2)
- Purchased electricity consumption
- Based on grid average carbon intensity where data not specified
- EU grid average: ~0.3 tCO2/kWh (2023)

### Excluded from These Values
- **Scope 3 Emissions:** Upstream (mining, transport of raw materials)
- **End-of-Life:** Recycling, disposal
- **Product Use Phase:** Not applicable for materials

---

## DATA QUALITY ASSESSMENT

| Source | Authority Level | Data Currency | Coverage |
|--------|----------------|---------------|----------|
| IEA | ⭐⭐⭐⭐⭐ Very High | 2018-2023 | Comprehensive |
| IPCC | ⭐⭐⭐⭐⭐ Very High | 2006-2019 | Comprehensive |
| World Steel Assoc | ⭐⭐⭐⭐⭐ Very High | 2023 | Steel industry |
| IAI | ⭐⭐⭐⭐⭐ Very High | 2023 | Aluminum industry |
| EFMA | ⭐⭐⭐⭐ High | 2020-2023 | Fertilizer industry |

**Overall Assessment:** Data is from authoritative sources and suitable for demo/educational purposes.

---

## FUTURE IMPROVEMENTS (Production Version)

For a production version with real CBAM filings:

1. **Use Official EU Commission Defaults**
   - Published at: https://taxation-customs.ec.europa.eu/
   - Updated regularly as regulations evolve

2. **Country-Specific Grid Factors**
   - Replace generic indirect emissions with country-specific electricity carbon intensity
   - Source: IEA CO2 Emissions from Fuel Combustion database

3. **Facility-Specific Actuals**
   - Support for supplier-provided actual emissions data
   - Validation against CBAM methodology requirements

4. **Temporal Updates**
   - Emission factors should be updated as technology improves
   - Track vintage and flag outdated data

---

## REFERENCES

1. IEA (2018). "Technology Roadmap - Low-Carbon Transition in the Cement Industry." International Energy Agency.

2. IPCC (2006, 2019). "Guidelines for National Greenhouse Gas Inventories." Intergovernmental Panel on Climate Change.

3. World Steel Association (2023). "Steel Climate Impact Report." worldsteel.org

4. International Aluminum Institute (2023). "GHG Emissions Protocol and Inventory." international-aluminium.org

5. European Commission (2023). "Carbon Border Adjustment Mechanism (CBAM)." taxation-customs.ec.europa.eu

6. IEA (2023). "Global Hydrogen Review." International Energy Agency.

---

**Last Updated:** 2025-10-15
**Version:** 1.0.0-demo
**Maintained By:** GreenLang CBAM Team
