# REG-PACK-014: Regulatory Requirements -- CSRD Retail & Consumer Goods Pack

**Version**: 1.0
**Status**: DRAFT
**Author**: GL-RegulatoryIntelligence
**Date**: 2026-03-16
**Category**: Regulatory Requirements Document
**Target Pack**: PACK-014-CSRD-Retail-Consumer-Goods
**NACE Sector**: Division G (Wholesale and Retail Trade, G45-G47)

---

## 1. Executive Summary

This document consolidates the regulatory requirements that a CSRD Retail & Consumer Goods Pack must address. Retail companies face a unique regulatory profile compared to manufacturing or financial services: their Scope 1 emissions are relatively low (store/warehouse energy, refrigerants), but their Scope 3 emissions (purchased goods, transport, end-of-life of sold products) typically represent 90-95% of their total carbon footprint. They are simultaneously subject to product-level regulations (EUDR, PPWR, ESPR, Textile Strategy), consumer-facing obligations (Green Claims, Empowering Consumers Directive), and supply chain due diligence requirements (CSDDD, Forced Labour Regulation) that cut across their entire value chain.

This document identifies 12 regulatory domains, extracts specific article references, effective dates, penalties, quantitative thresholds, and maps each to the data points and calculation engines that PACK-014 must implement.

---

## 2. Regulation Index

| # | Regulation | Reference | Effective Date | Retail Relevance |
|---|-----------|-----------|----------------|------------------|
| R1 | CSRD / ESRS Set 1 | Directive 2022/2464, Del. Reg. 2023/2772 | FY2024 (Wave 1) / FY2025 (Wave 2) | PRIMARY -- core sustainability reporting |
| R2 | Omnibus I | Directive 2026/470 | 18 Mar 2026 (entry into force) | CRITICAL -- scope reduction, 61% datapoint cut |
| R3 | EUDR | Regulation 2023/1115 | 30 Dec 2026 (large operators) | HIGH -- palm oil, soy, cocoa, coffee, rubber, timber, cattle |
| R4 | PPWR | Regulation 2025/40 | 12 Aug 2026 (general application) | HIGH -- packaging recycled content, labeling, EPR |
| R5 | CSDDD / CS3D | Directive 2024/1760, amended by 2026/470 | 26 Jul 2028 (Phase 1, post-Omnibus) | HIGH -- supply chain due diligence |
| R6 | ESPR | Regulation 2024/1781 | 2027 (textiles, electronics); 2028 (furniture) | HIGH -- Digital Product Passport |
| R7 | EU Textile Strategy / WFD | Directive 2025/XX (revised WFD) | Oct 2025 (entry); transposition by mid-2027 | MEDIUM-HIGH -- textile EPR, microplastics |
| R8 | Food Waste Targets | Revised WFD (Directive 2025/XX) | 2030 target year | MEDIUM -- food retailers, grocery |
| R9 | GHG Protocol / Scope 3 | ESRS E1, GHG Protocol Scope 3 Standard | Ongoing (per CSRD timeline) | CRITICAL -- 90-95% of retail footprint |
| R10 | Green Claims / ECGT | Directive 2024/825 (ECGT); Green Claims Dir. withdrawn | 27 Sep 2026 (ECGT application) | HIGH -- product environmental claims |
| R11 | EED / F-Gas | Directive 2023/1791 (EED); Regulation 2024/573 (F-Gas) | Oct 2025 (EED transposition); Mar 2024 (F-Gas) | MEDIUM -- store energy, refrigerant management |
| R12 | Forced Labour Regulation | Regulation 2024/XX (Forced Labour Ban) | 14 Dec 2027 (application) | HIGH -- garments, food, electronics supply chains |

---

## 3. R1 -- CSRD / ESRS Set 1: Retail-Specific Requirements

### 3.1 Regulatory Reference
- **Directive**: 2022/2464 (CSRD), amending Directive 2013/34/EU (Accounting Directive)
- **Standards**: Delegated Regulation 2023/2772 (ESRS Set 1), 12 standards total
- **Sector standards**: Originally planned for NACE G by EFRAG, now permanently cancelled by Omnibus I

### 3.2 Scope (Post-Omnibus I)
Retail companies are in scope if they exceed BOTH:
- **>1,000 employees** (annual average), AND
- **>EUR 450 million net turnover** (at individual or consolidated group level)

### 3.3 ESRS Standards -- Retail Materiality Profile

The following table maps each ESRS standard to its typical materiality for retail/consumer goods companies. Research shows 78% of consumer-facing sector companies assess every ESRS chapter as material.

| ESRS Standard | Retail Materiality | Key Rationale |
|---------------|-------------------|---------------|
| **ESRS 1** (General Requirements) | MANDATORY | Cross-cutting, all in-scope companies |
| **ESRS 2** (General Disclosures) | MANDATORY | Cross-cutting, all in-scope companies |
| **E1** (Climate Change) | ALMOST ALWAYS MATERIAL | Scope 3 dominance (Cat 1, 4, 9, 11, 12); store/warehouse energy; refrigerant F-gases |
| **E2** (Pollution) | OFTEN MATERIAL | Microplastics (textiles), packaging chemicals, REACH/CLP substances in sold products |
| **E3** (Water & Marine) | SOMETIMES MATERIAL | Supply chain water (cotton, food agriculture); store water use typically immaterial |
| **E4** (Biodiversity) | OFTEN MATERIAL | EUDR-linked commodities (palm oil, soy, cocoa, coffee, cattle, timber); land-use in supply chain |
| **E5** (Resource Use & Circular Economy) | ALMOST ALWAYS MATERIAL | Packaging waste, food waste, product durability/repairability, take-back schemes, EPR |
| **S1** (Own Workforce) | ALMOST ALWAYS MATERIAL | Large retail workforces, working conditions, living wages, part-time/seasonal workers |
| **S2** (Workers in Value Chain) | ALMOST ALWAYS MATERIAL | Garment workers, agricultural workers, warehouse/logistics workers; forced labor risks |
| **S3** (Affected Communities) | OFTEN MATERIAL | Land rights (EUDR commodities), community impacts of supply chain sourcing |
| **S4** (Consumers & End-Users) | ALMOST ALWAYS MATERIAL | Product safety, health & nutrition, data privacy, responsible marketing, accessibility |
| **G1** (Business Conduct) | ALMOST ALWAYS MATERIAL | Anti-corruption in supply chain, lobbying, tax transparency, responsible sourcing codes |

### 3.4 Key ESRS Datapoints for Retail

#### 3.4.1 ESRS E1 -- Climate Change (Retail Priority Datapoints)

| Datapoint | DR Reference | Retail Specifics |
|-----------|-------------|------------------|
| Total GHG emissions (Scope 1, 2, 3) | E1-6 | Scope 3 Cat 1 (purchased goods) typically 60-80% of total |
| Scope 1 breakdown | E1-6 | Refrigerant leakage (F-gases), company vehicles, natural gas for heating |
| Scope 2 (location & market-based) | E1-6 | Electricity for stores, warehouses, data centers; green energy procurement |
| Scope 3 by category | E1-6 | Cat 1, 4, 9, 11, 12 dominant for retail (see Section 11) |
| GHG intensity per net revenue | E1-6 | tCO2e / EUR million net revenue |
| Climate transition plan | E1-1 | Decarbonization targets for store fleet, logistics, product portfolio |
| Energy consumption & mix | E1-5 | Total energy, % renewable, energy intensity per m2 of retail space |
| Internal carbon pricing | E1-8 (if applicable) | Internal carbon price applied to sourcing/investment decisions |

#### 3.4.2 ESRS E4 -- Biodiversity (Retail Priority Datapoints)

| Datapoint | DR Reference | Retail Specifics |
|-----------|-------------|------------------|
| Biodiversity transition plan | E4-1 | Deforestation-free sourcing commitments, EUDR compliance |
| Sites near biodiversity-sensitive areas | E4-5 | Store/warehouse locations; supply chain land-use hotspots |
| Land-use change (supply chain) | E4-5 | Deforestation risk commodities per EUDR |
| Impact on species | E4-5 | IUCN Red List species affected by sourcing |

#### 3.4.3 ESRS E5 -- Resource Use & Circular Economy (Retail Priority Datapoints)

| Datapoint | DR Reference | Retail Specifics |
|-----------|-------------|------------------|
| Resource inflows (materials used) | E5-4 | Packaging materials by type, recycled content % |
| Resource outflows (products & packaging) | E5-5 | Product durability, recyclability, recycled content in products |
| Waste generated | E5-5 | Food waste (tonnes), packaging waste, textile waste, e-waste |
| Waste by treatment type | E5-5 | Recycling, composting, incineration, landfill by waste stream |
| Circular economy targets | E5-3 | Packaging reduction, food waste reduction, take-back volumes |
| Financial effects of CE risks | E5-6 | EPR fee exposure, packaging tax liabilities, raw material cost volatility |

#### 3.4.4 ESRS S1 -- Own Workforce (Retail Priority Datapoints)

| Datapoint | DR Reference | Retail Specifics |
|-----------|-------------|------------------|
| Total headcount by contract type | S1-6 | Full-time, part-time, seasonal, zero-hours contracts |
| Non-employee workers | S1-7 | Contractors, temporary agency workers, gig workers |
| Gender pay gap | S1-16 | Retail-specific gender pay analysis |
| Adequate wages | S1-10 | Living wage benchmarking for retail workers by country |
| Training and skills development | S1-13 | Hours of training per employee, digital skills |
| Health and safety | S1-14 | Work-related injuries, fatalities, musculoskeletal disorders |
| Work-life balance | S1-15 | Night work, weekend shifts, scheduling predictability |

#### 3.4.5 ESRS S2 -- Workers in the Value Chain (Retail Priority Datapoints)

| Datapoint | DR Reference | Retail Specifics |
|-----------|-------------|------------------|
| Processes for identifying impacts | S2-1 | Supplier audits, risk assessment methodology |
| Material negative impacts | S2-2 | Forced labor, child labor, unsafe conditions, poverty wages |
| Due diligence actions | S2-4 | Supplier code of conduct, remediation, capacity building |
| Channels for raising concerns | S2-3 | Grievance mechanisms in supply chain |

#### 3.4.6 ESRS S4 -- Consumers & End-Users (Retail Priority Datapoints)

| Datapoint | DR Reference | Retail Specifics |
|-----------|-------------|------------------|
| Product safety incidents | S4-4 | Product recalls, safety alerts, consumer complaints |
| Health & nutrition | S4 (sector) | Nutritional profile of food products, reformulation targets |
| Data privacy | S4-4 | Customer data handling, breaches, consent management |
| Responsible marketing | S4-4 | Marketing to vulnerable groups, green claims substantiation |

### 3.5 What Makes Retail Different from Manufacturing and Financial Services

| Dimension | Retail | Manufacturing | Financial Services |
|-----------|--------|--------------|-------------------|
| **Scope 1** | Low (5-10%): refrigerants, fleet, heating | High (30-60%): process emissions, combustion | Very low (<1%) |
| **Scope 2** | Medium (5-15%): store/warehouse electricity | Medium (10-25%): factory electricity | Low (5-10%): office electricity |
| **Scope 3** | Dominant (80-95%): purchased goods, transport | Significant (30-60%): raw materials, use phase | Dominant (95%+): financed emissions |
| **Primary E standard** | E1 + E5 (circular economy) | E1 + E2 (pollution) | E1 (climate) |
| **Primary S standard** | S1 + S2 + S4 (consumers) | S1 + S2 | S1 + S4 (data privacy) |
| **Product regulations** | EUDR, PPWR, ESPR, Textile, Food | CBAM, IED, REACH, ETS | SFDR, Taxonomy, MiFID |
| **Value chain depth** | Deep (Tier 1-4+ suppliers) | Medium (Tier 1-2 suppliers) | Indirect (investee companies) |
| **Consumer-facing risk** | High (brand, greenwashing, product safety) | Medium (B2B) | Medium (retail investors) |

---

## 4. R2 -- Omnibus I (Directive 2026/470): Impact on Retail

### 4.1 Regulatory Reference
- **Directive**: 2026/470, amending Directives 2013/34/EU (Accounting), 2022/2464 (CSRD), and 2024/1760 (CSDDD)
- **Published**: 26 February 2026, Official Journal of the EU
- **Entry into force**: 18 March 2026
- **Transposition deadline**: 19 March 2027 (CSRD provisions); 26 July 2028 (CSDDD provisions)

### 4.2 Revised Scope Thresholds

| Criterion | Pre-Omnibus | Post-Omnibus (Directive 2026/470) |
|-----------|-------------|----------------------------------|
| **Employee threshold** | 250 employees (average) | **1,000 employees** (average) |
| **Turnover threshold** | EUR 50M net turnover | **EUR 450M net turnover** |
| **Logic** | OR (either threshold) | **AND (both thresholds simultaneously)** |
| **Balance sheet** | EUR 25M total assets | Removed as independent criterion |
| **Listed SMEs** | In scope (Wave 3) | **Out of scope** (may use VSME voluntarily) |

**Impact on retail**: Many mid-sized retail chains that were previously in scope under the 250-employee threshold are now excluded. Only very large retailers (e.g., Carrefour, Tesco, REWE, Ahold Delhaize, H&M, Inditex, IKEA-level) remain in scope. Smaller specialty retailers and regional chains are relieved of mandatory reporting.

### 4.3 ESRS Datapoint Reduction

| Dimension | Original ESRS | Amended ESRS (Post-Omnibus) |
|-----------|---------------|----------------------------|
| **Total datapoints** | ~1,178 | **~460** (61% reduction) |
| **Voluntary datapoints** | ~400 | **0** (all voluntary DPs eliminated) |
| **Mandatory datapoints** | Conditional on materiality | Conditional on materiality (unchanged) |
| **Sector-specific ESRS** | Planned (incl. NACE G retail) | **Permanently cancelled** |

**Key implication for PACK-014**: Since sector-specific ESRS for retail/wholesale (NACE G) have been permanently cancelled, this pack must provide the retail-sector intelligence, calculation methodologies, and reporting templates that the regulation no longer prescribes. This is the core value proposition.

### 4.4 Value Chain Cap

Omnibus I introduces a "value chain cap" limiting the data that in-scope companies can request from value chain companies with fewer than 1,000 employees. Retailers with deep supply chains (Tier 1-4) must therefore:
- Use estimation methodologies rather than requiring primary data from smaller suppliers
- Accept spend-based and average-data methods for Scope 3 calculations where supplier-specific data is unavailable
- Limit data requests to the "proportionate" information defined in the amended ESRS

### 4.5 Timeline

| Date | Milestone |
|------|-----------|
| 18 Mar 2026 | Omnibus I Directive 2026/470 enters into force |
| Q2 2026 | EC 4-week call for feedback on amended ESRS delegated act |
| Jun 2026 | VSME voluntary standard delegated act expected |
| 18 Sep 2026 | Deadline for EC to adopt revised ESRS delegated act |
| 19 Mar 2027 | Member State transposition deadline (CSRD provisions) |
| 1 Jan 2027 | Amended ESRS applicable (optional early use for FY2026) |

---

## 5. R3 -- EU Deforestation Regulation (EUDR) for Retail

### 5.1 Regulatory Reference
- **Regulation**: 2023/1115 (EUDR), as amended by Regulation 2025/XX (simplification)
- **Cutoff date**: 31 December 2020 (no deforestation/degradation after this date)
- **Application**: 30 December 2026 (large and medium operators); 30 June 2027 (micro and small enterprises)

### 5.2 Covered Commodities Relevant to Retail

| Commodity | Retail Product Examples | Risk Level |
|-----------|----------------------|------------|
| **Palm oil** | Processed foods, cosmetics, cleaning products, margarine | HIGH |
| **Soy** | Animal feed (meat supply chain), soy sauce, tofu, soy protein | HIGH |
| **Cocoa** | Chocolate, confectionery, baking products | HIGH |
| **Coffee** | Roasted coffee, instant coffee, coffee capsules | MEDIUM-HIGH |
| **Rubber** | Tires, footwear, industrial rubber products | MEDIUM |
| **Timber/Wood** | Furniture, paper products, cardboard packaging, charcoal | MEDIUM |
| **Cattle** | Beef, leather goods (shoes, bags, jackets), dairy products | HIGH |

### 5.3 Due Diligence Requirements (Articles 8-12)

Retailers who "place on the market" (first market entry) or "make available on the market" must:

1. **Information collection** (Article 9): Geolocation of production plots (lat/long coordinates), date/period of production, quantity, country of production, supplier identity
2. **Risk assessment** (Article 10): Assess risk that products are non-compliant using country benchmarking, satellite/geospatial data, supplier audit information
3. **Risk mitigation** (Article 11): If risk is not negligible -- conduct independent surveys, audits, third-party verification
4. **Due Diligence Statement** (Article 4): Submit via EU TRACES/Information System before placing products on market

### 5.4 Simplifications (December 2025 Amendment)

- Due Diligence Statements only required from the operator that first places the product on the EU market, not subsequent traders
- SME primary operators may submit one-off simplified declarations
- Country benchmarking system: "low risk", "standard risk", "high risk" countries -- simplified due diligence for low-risk origins

### 5.5 Penalties (Article 25)

| Penalty Type | Detail |
|-------------|--------|
| **Financial fines** | Up to **4% of EU-wide annual turnover** |
| **Confiscation** | Confiscation of non-compliant products and revenues |
| **Public procurement exclusion** | Temporary exclusion from public procurement |
| **Market ban** | Temporary prohibition from placing/making available products |
| **Escalation** | Fines increase with each additional infringement |

### 5.6 Retail-Specific Considerations

- **Private-label products**: Retailers with own-brand products are classified as "operators" (first placing on market), not merely "traders", triggering full due diligence obligations
- **Multi-commodity products**: Products containing multiple EUDR commodities (e.g., a chocolate bar with cocoa, palm oil, and soy lecithin) require separate due diligence for each commodity
- **E-commerce**: Online retail sales of EUDR products into the EU market are covered

---

## 6. R4 -- Packaging and Packaging Waste Regulation (PPWR)

### 6.1 Regulatory Reference
- **Regulation**: 2025/40 (PPWR), replacing Directive 94/62/EC
- **Published**: 22 January 2025, Official Journal of the EU
- **Entry into force**: 12 February 2025
- **General application date**: 12 August 2026

### 6.2 Recycled Content Targets (Article 7)

#### Plastic Packaging -- Mandatory Minimums

| Packaging Type | By 1 Jan 2030 | By 1 Jan 2040 |
|---------------|---------------|---------------|
| Contact-sensitive PET packaging (excl. beverage bottles) | **30%** | **50%** |
| Contact-sensitive non-PET plastic packaging (excl. beverage bottles) | **10%** | **25%** |
| Single-use plastic beverage bottles | **30%** | **65%** |
| Other non-contact-sensitive plastic packaging | **35%** | **65%** |

### 6.3 Packaging Reduction Targets (Article 38)

| Metric | Target |
|--------|--------|
| Overall packaging waste reduction | **5% by 2030**, **10% by 2035**, **15% by 2040** (vs. 2018 baseline) |
| Plastic packaging waste reduction | **10% by 2030**, **15% by 2035**, **20% by 2040** |

### 6.4 Recyclability Requirements (Articles 5-6)

| Milestone | Requirement |
|-----------|-------------|
| 1 Jan 2030 | All packaging placed on the EU market must be **"designed for recycling"** (per recyclability criteria in delegated acts) |
| 1 Jan 2035 | All packaging must be **"recycled at scale"** (demonstrated through established collection and recycling infrastructure) |

### 6.5 Labeling Requirements (Article 11)

| Deadline | Requirement |
|----------|-------------|
| 12 Aug 2026 | Labels must include **material composition** and **sorting instructions** (which bin) |
| 12 Aug 2028 | Harmonized labeling system with **standardized pictograms** for waste sorting |
| 2030+ | QR code linking to Digital Product Passport data for packaging |

### 6.6 Reuse Targets and Retailer Obligations

| Obligation | Detail |
|-----------|--------|
| **E-commerce reuse option** | From 2030, online retailers must offer a **reusable shipping option** at checkout, presented at least as prominently as single-use |
| **Transport packaging reuse** | From 2030, 40% reusable by weight; from 2040, 70% |
| **Beverage fill target** | From 2030, at least 10% of beverages in reusable packaging for retail |

### 6.7 Extended Producer Responsibility (Articles 40-46)

- **EPR schemes**: All Member States must have EPR for all packaging types by 2026
- **Eco-modulation**: By 2030, EPR fees must be eco-modulated based on packaging recyclability performance grades (A-E scale)
- **Fee structure**: Higher fees for non-recyclable, composite, or hard-to-recycle packaging
- **Retailer obligation**: The "final distributor" (retailer) bears EPR obligations for products sold under its own brand

### 6.8 Penalties

Penalties are determined at Member State level but must be "effective, proportionate, and dissuasive." The EC will adopt delegated acts specifying minimum enforcement standards.

---

## 7. R5 -- Corporate Sustainability Due Diligence Directive (CSDDD / CS3D)

### 7.1 Regulatory Reference
- **Directive**: 2024/1760 (CSDDD), as amended by Directive 2026/470 (Omnibus I)
- **Published**: 5 July 2024
- **Original transposition deadline**: 26 July 2026 (postponed by "stop-the-clock" directive)
- **Revised transposition deadline**: 26 July 2027 (national law); **26 July 2028** (first application, Phase 1)

### 7.2 Scope (Post-Omnibus I)

| Phase | Criterion | Application Date |
|-------|-----------|-----------------|
| Phase 1 | >5,000 employees AND >EUR 1.5 billion net turnover (worldwide) | 26 Jul 2028 |
| Phase 2 | >3,000 employees AND >EUR 900 million net turnover | 26 Jul 2029 |
| Phase 3 | >1,000 employees AND >EUR 450 million net turnover | 26 Jul 2030 |
| Non-EU companies | >EUR 1.5 billion net turnover generated in the EU (Phase 1) | 26 Jul 2028 |

### 7.3 Due Diligence Obligations (Articles 5-16)

The CSDDD requires companies to:

1. **Integrate due diligence into policies** (Article 5): Company policy with due diligence approach, code of conduct for employees and subsidiaries
2. **Identify and assess actual/potential adverse impacts** (Article 6): Mapping of own operations, subsidiaries, and business partners
3. **Prevent and mitigate potential adverse impacts** (Article 7): Prevention action plan, contractual assurances from business partners
4. **Bring actual adverse impacts to an end** (Article 8): Corrective action plans, remediation
5. **Provide remediation** (Article 12): Remediation mechanisms for those affected
6. **Establish complaints mechanism** (Article 9): Accessible grievance mechanism
7. **Monitor effectiveness** (Article 10): Periodic assessment of due diligence measures
8. **Publicly communicate** (Article 11): Annual statement on due diligence policies and outcomes

### 7.4 Climate Transition Plan (Article 15)

In-scope companies must adopt and implement a **climate transition plan** aligned with the Paris Agreement 1.5C goal, including:
- Time-bound GHG emission reduction targets (Scope 1, 2, and where relevant 3)
- Key actions to reach those targets
- Investment and financing plan
- Role of offsets (limited)

### 7.5 Retail Supply Chain Risk Areas

| Risk Area | Supply Chain Tier | Key Risks |
|-----------|------------------|-----------|
| **Garments & Textiles** | Tier 2-4 (spinners, weavers, dyers) | Forced labor, child labor, excessive overtime, building safety (Rana Plaza-type) |
| **Food & Agriculture** | Tier 2-4 (farms, plantations) | Land grabbing, deforestation, pesticide exposure, migrant worker exploitation |
| **Electronics** | Tier 3-5 (mining, component mfg) | Conflict minerals, forced labor in mining, hazardous waste, e-waste |
| **Footwear & Leather** | Tier 2-3 (tanneries, leather processing) | Chemical exposure, water pollution, child labor |
| **Home Furnishings** | Tier 2-3 (timber, textiles, ceramics) | Illegal logging, forced labor, community displacement |

### 7.6 Penalties (Article 27)

| Penalty Type | Detail |
|-------------|--------|
| **Financial penalties** | Up to **5% of worldwide net turnover** (post-Omnibus, reduced from original proposal) |
| **Injunctions** | Interim measures to cease specific practices |
| **Civil liability** | Companies liable for damages caused by failure to comply (Article 22) |
| **Public naming** | Publication of infringement decisions |

---

## 8. R6 -- Ecodesign for Sustainable Products Regulation (ESPR)

### 8.1 Regulatory Reference
- **Regulation**: 2024/1781 (ESPR)
- **Entry into force**: 18 July 2024
- **Framework**: Sets framework for delegated acts per product category

### 8.2 Digital Product Passport (DPP) -- Retail Product Categories

| Product Category | DPP Mandatory From | Key Data Fields |
|-----------------|-------------------|-----------------|
| **Batteries** | February 2027 (EU Battery Regulation 2023/1542) | Carbon footprint, recycled content, material composition, durability, state of health |
| **Textiles & Apparel** | 2027 (delegated act finalized) | Fiber composition, country of manufacturing, repairability, recyclability, environmental footprint |
| **Electronics** | 2027-2028 (delegated acts in progress) | Energy efficiency, repairability index, hazardous substances, recycled content |
| **Furniture** | 2028 | Material composition, durability, recyclability, formaldehyde emissions |
| **Construction products** | 2028 | Environmental declarations, recycled content |
| **Tyres** | 2028 | Rolling resistance, wet grip, external rolling noise, abrasion rate |

### 8.3 DPP Technical Requirements

- **Unique product identifier**: Serialized or batch-level
- **Data carrier**: QR code, RFID, NFC, or data matrix code
- **Data format**: Machine-readable, interoperable (GS1/EPCIS standards expected)
- **Accessibility**: Available to consumers, market surveillance authorities, recyclers, and customs
- **Data hosting**: Decentralized, accessible via EU Digital Product Passport registry

### 8.4 Ecodesign Requirements (Delegated Acts)

The ESPR enables the EC to set performance requirements for products via delegated acts covering:
- **Durability** (minimum product lifetime, number of use cycles)
- **Repairability** (spare parts availability, repair manuals, repairability score)
- **Recyclability** (design for disassembly, mono-material design, recyclability labeling)
- **Recycled content** (minimum % recycled material)
- **Energy efficiency** (operating energy performance)
- **Carbon footprint** (product-level CO2e per functional unit)
- **Hazardous substances** (restrictions/prohibitions per REACH/CLP)

### 8.5 Retailer Obligations

- Retailers selling products covered by ESPR delegated acts must ensure DPPs are available to consumers at point of sale (physical and online)
- E-commerce platforms must display DPP data or provide QR code link
- Retailers are responsible for ensuring products on their shelves meet ecodesign requirements (market surveillance applies to the entire distribution chain)

---

## 9. R7 -- EU Textile Strategy & Revised Waste Framework Directive

### 9.1 Regulatory References
- **EU Strategy for Sustainable and Circular Textiles**: COM(2022) 141 final (March 2022)
- **Revised Waste Framework Directive**: Entered into force 16 October 2025
- **ESPR textile delegated act**: Expected 2027
- **Microplastic measures**: Under ESPR and ECHA restriction proposals

### 9.2 Mandatory EPR for Textiles

| Requirement | Detail |
|-------------|--------|
| **Legal basis** | Revised Waste Framework Directive (2025) |
| **Transposition** | Member States: 20 months to transpose (by mid-2027) |
| **EPR scheme setup** | Member States: 30 months to establish schemes (by mid-2028) |
| **Scope** | All textile and footwear products placed on the EU market |
| **Fee structure** | Eco-modulated based on durability, recyclability, recycled content, microplastic release |
| **Collection targets** | Separate collection of textiles already mandatory from 1 Jan 2025 (existing WFD provision) |

### 9.3 Microplastic Requirements

| Requirement | Source | Timeline |
|------------|--------|----------|
| Microplastic release measurement | ESPR textile delegated act | 2027 |
| Microplastic emission reduction target | EU goal: 30% reduction by 2030 | 2030 |
| Washing machine filters | Proposed under ESPR (for new machines) | 2028 (expected) |
| Textile design for reduced shedding | ESPR ecodesign requirements | 2027+ |

### 9.4 Greenwashing Prevention for Textiles

| Measure | Status | Application |
|---------|--------|-------------|
| Ban on generic "eco" / "green" / "sustainable" claims | ECGT Directive 2024/825 | 27 Sep 2026 |
| Substantiation requirements for durability claims | ECGT Directive | 27 Sep 2026 |
| Product Environmental Footprint (PEF) for textiles | PEF Category Rules adopted | Voluntary (now uncertain after Green Claims Dir. withdrawal) |
| DPP with environmental performance data | ESPR textile delegated act | 2027 |

### 9.5 Retailer Obligations

- Retailers selling textiles under own brand are "producers" for EPR purposes
- Must register with EPR scheme in each Member State where products are sold
- Must provide consumers with information on textile care (to extend product life) and repair/take-back options
- Must ensure textile products meet ESPR ecodesign requirements from 2027

---

## 10. R8 -- Food Waste Reduction Targets

### 10.1 Regulatory Reference
- **Legal basis**: Revised Waste Framework Directive (2025 amendment), Article 9a
- **Target year**: 2030
- **Baseline**: Annual average food waste generated between 2021 and 2023

### 10.2 Binding Targets

| Stage | Target (by 2030 vs. 2021-2023 baseline) | EP Proposed (higher ambition) |
|-------|----------------------------------------|-------------------------------|
| Processing & Manufacturing | **10% reduction** | 20% reduction |
| **Retail, restaurants, food services, households** | **30% reduction** | **40% reduction** |
| Primary production | No binding target yet (review by end 2027) | To be assessed |

### 10.3 Measurement & Reporting

| Requirement | Detail |
|-------------|--------|
| **Measurement methodology** | EC Delegated Decision 2019/1597 (food waste measurement) |
| **Reporting frequency** | Annual, to national authorities |
| **ESRS E5 alignment** | Food waste tonnes reported under ESRS E5-5 (waste generated) as a sector-specific waste stream |
| **SDG alignment** | SDG Target 12.3: halve per capita global food waste at retail and consumer levels by 2030 |

### 10.4 Retail-Specific Actions

| Action Area | Detail |
|-------------|--------|
| **Food waste tracking** | Track by category: bakery, produce, dairy, meat, prepared foods |
| **Root cause analysis** | Overstock, expiry management, display practices, forecasting accuracy |
| **Redistribution** | Partnerships with food banks, charities (e.g., FareShare, Banque Alimentaire) |
| **Valorization** | Animal feed, composting, anaerobic digestion for unsalvageable food waste |
| **Markdown management** | Dynamic pricing, reduced-to-clear, app-based surplus sales (e.g., Too Good To Go) |
| **Supplier collaboration** | Flexible order quantities, date labeling alignment, cosmetic standards relaxation |

### 10.5 Connection to ESRS E5

Food waste must be disclosed under ESRS E5-5 (Resource Outflows) as a specific waste stream. Retailers should report:
- Total food waste generated (tonnes)
- Food waste by destination (redistribution, animal feed, composting, anaerobic digestion, incineration, landfill)
- Food waste as % of food sold
- Year-on-year reduction progress

---

## 11. R9 -- Scope 3 Emissions for Retail

### 11.1 Regulatory and Standards References
- **ESRS E1-6**: GHG emissions disclosure (mandatory if E1 is material)
- **GHG Protocol Corporate Value Chain (Scope 3) Standard**: Category definitions and calculation guidance
- **GHG Protocol Technical Guidance for Scope 3**: Category-specific methods

### 11.2 Retail Scope 3 Profile

For a typical retailer, Scope 3 represents 80-95% of total emissions. The dominant categories are:

| Category | % of Retail Total | Description | Key Data Sources |
|----------|-------------------|-------------|------------------|
| **Cat 1: Purchased Goods & Services** | **50-75%** | Embodied emissions in all products purchased for resale + services | Supplier-specific EFs, spend-based EFs, hybrid methods, CDP Supply Chain |
| **Cat 4: Upstream Transportation** | **5-15%** | Inbound logistics (supplier to warehouse/store) | Freight data (tonne-km), carrier emissions reports, GLEC Framework |
| **Cat 9: Downstream Transportation** | **2-8%** | Customer home delivery, last-mile logistics | Delivery fleet data, parcel carrier reports, distance-based EFs |
| **Cat 11: Use of Sold Products** | **3-15%** | Energy used by products during consumer use (electronics, appliances, lighting) | Product energy ratings, assumed use profiles, national grid EFs |
| **Cat 12: End-of-Life Treatment** | **1-5%** | Disposal/recycling of products after consumer use | Waste treatment EFs, product material composition, national recycling rates |
| **Cat 5: Waste from Operations** | **0.5-2%** | Waste from stores, warehouses, offices | Waste audit data, treatment EFs |
| **Cat 6: Business Travel** | **0.1-1%** | Employee flights, hotels, rail | Travel booking data, DEFRA/EPA EFs |
| **Cat 7: Employee Commuting** | **0.5-2%** | Daily commuting of retail workers | Employee surveys, distance/mode estimates |

### 11.3 Calculation Methodologies for Retail

#### Category 1: Purchased Goods & Services

| Method | Data Required | Accuracy | Recommended Use |
|--------|--------------|----------|-----------------|
| **Supplier-specific** | Primary data from suppliers (tCO2e per product/batch) | Highest | Key suppliers, own-brand products |
| **Hybrid** | Activity data + supplier-specific EFs for key inputs | High | Top-20 suppliers by spend |
| **Average-data** | Mass/volume of goods x product-category EFs (e.g., DEFRA, ecoinvent) | Medium | Mid-tier products |
| **Spend-based** | EUR spend x spend-based EFs (EEIO models, e.g., EXIOBASE, USEEIO) | Lowest | Long-tail of small-spend categories |

**Retail-specific guidance**: For retailers with 10,000+ SKUs, a tiered approach is essential:
- Tier 1 (top 50 products by volume/emissions): Supplier-specific data
- Tier 2 (top 500 products): Hybrid or average-data
- Tier 3 (remaining SKUs): Spend-based or average-data by product category

#### Category 4: Upstream Transportation

| Method | Detail |
|--------|--------|
| **Distance-based** | Tonne-km x mode-specific EF (road, rail, sea, air) |
| **Fuel-based** | Actual fuel consumed by logistics providers |
| **Spend-based** | EUR spend on freight x transport EF |
| **GLEC Framework** | Global Logistics Emissions Council methodology (ISO 14083 alignment) |

#### Category 9: Downstream Transportation

| Method | Detail |
|--------|--------|
| **Delivery fleet data** | Own delivery fleet: fuel consumption, EV mix, distance driven |
| **Parcel carrier data** | Carrier-reported emissions per parcel/kg |
| **Customer travel** | For in-store retail: estimated customer trips x mode x distance (often excluded or estimated) |

#### Category 11: Use of Sold Products

| Method | Detail |
|--------|--------|
| **Direct use-phase emissions** | Products that consume energy during use (electronics, appliances, lighting): assumed lifetime x annual energy x grid EF |
| **Indirect use-phase** | Products that do not directly consume energy: typically zero or negligible |
| **GHG-emitting products** | Gas appliances sold by retailers: lifetime gas consumption x gas EF |

#### Category 12: End-of-Life Treatment

| Method | Detail |
|--------|--------|
| **Waste-type-specific** | Product material composition x waste treatment EFs (landfill CH4, incineration CO2, recycling avoided) |
| **Product-category** | Average end-of-life profile per product category x national waste treatment mix |

### 11.4 Data Quality Requirements

ESRS E1-6 requires disclosure of the **data quality** of Scope 3 calculations. Recommended scoring:

| Level | Description | Scope 3 Application |
|-------|-------------|---------------------|
| **1 (Highest)** | Verified primary data from value chain | Supplier-specific, third-party verified |
| **2** | Unverified primary data from value chain | Supplier-reported, not yet verified |
| **3** | Average data from public databases | ecoinvent, DEFRA, EPA EFs by product category |
| **4** | Spend-based or highly estimated | EEIO models, spend x EF |
| **5 (Lowest)** | Expert judgment or proxy | No activity data, rough estimates |

---

## 12. R10 -- Green Claims & Empowering Consumers Directive

### 12.1 Regulatory References

| Regulation | Status | Application Date |
|-----------|--------|-----------------|
| **Empowering Consumers for Green Transition (ECGT)** -- Directive 2024/825 | ADOPTED, transposition required by 27 Mar 2026 | **27 September 2026** |
| **Green Claims Directive** -- COM(2023) 166 | **WITHDRAWN** (EC announced withdrawal June 2025) | N/A |
| **Unfair Commercial Practices Directive (UCPD)** -- 2005/29/EC | Already applicable | Ongoing |

### 12.2 ECGT Directive 2024/825 -- Key Prohibitions

The following practices are prohibited from 27 September 2026:

| Prohibition | Article | Retail Example |
|------------|---------|----------------|
| Generic environmental claims without proof | Annex I, new point 4a | "Eco-friendly product", "Green choice", "Sustainable" |
| Sustainability labels not based on certification/public authority | Annex I, new point 4b | Unverified "green" logos on own-brand products |
| Carbon neutrality claims based solely on offsets | Annex I, new point 4c | "Carbon neutral delivery", "Climate-positive shopping" |
| Durability claims without evidence | Annex I, new point 4d | "Long-lasting" without test data |
| Prompting consumers to replace consumables earlier than necessary | Annex I, new point 4e | Premature ink cartridge warnings, filter replacement alerts |
| Software updates that impair product function | Annex I, new point 4f | Updates that slow older devices |

### 12.3 Substantiation Requirements

Even with the Green Claims Directive withdrawn, the ECGT and existing UCPD require:
- All environmental claims must be **truthful, clear, unambiguous, and not misleading**
- Claims must be **substantiated** with evidence available at the time the claim is made
- Comparative claims must compare like-for-like products/services
- Claims about future environmental performance must include **time-bound commitments** and **implementation plans**

### 12.4 Penalties (ECGT via UCPD/CPC Regulation)

| Penalty Type | Detail |
|-------------|--------|
| **Financial penalties** | Up to **4% of annual turnover** or **EUR 2 million** (whichever is higher) for widespread infringements (under CPC Regulation 2017/2394) |
| **Injunctions** | Court orders to cease misleading claims |
| **Product-level remedies** | Corrective advertising, product recall in severe cases |
| **Reputational** | Public disclosure of infringement decisions |

### 12.5 Retail-Specific Impact

Retailers are particularly exposed because they:
- Make environmental claims on **own-brand/private-label products** (packaging claims, sourcing claims, carbon claims)
- Display **third-party sustainability labels** on shelves (must verify legitimacy)
- Run **marketing campaigns** with environmental messaging ("our greenest range ever")
- Operate **loyalty/rewards programs** with sustainability angles ("earn green points")

---

## 13. R11 -- Energy Efficiency Directive (EED) & F-Gas Regulation

### 13.1 EED -- Directive 2023/1791

#### Regulatory Reference
- **Directive**: 2023/1791 (recast EED)
- **Entry into force**: 10 October 2023
- **National transposition deadline**: 11 October 2025
- **EU-wide target**: 11.7% reduction in final energy consumption by 2030 vs. 2020 projections

#### Retail Energy Obligations

| Annual Energy Consumption | Obligation | Deadline |
|--------------------------|-----------|----------|
| **>85 TJ/year** | Implement certified **Energy Management System** (e.g., ISO 50001) | October 2027 |
| **10-85 TJ/year** | Conduct **energy audit** every 4 years | October 2026 (initial) |
| **<10 TJ/year** | No mandatory audit (but energy efficiency recommendations apply) | N/A |

**Retail context**: A large supermarket chain with 200+ stores and distribution centers often exceeds 85 TJ/year total energy consumption. Key energy categories:
- Store lighting (30-40% of store energy)
- Refrigeration and HVAC (40-50% of store energy)
- Warehousing and cold chain (logistics energy)
- IT and point-of-sale systems

#### Energy Audit Scope (Article 11)

Energy audits must cover:
- At least 85% of total energy consumption
- Building envelope performance (insulation, glazing)
- Heating, ventilation, air conditioning (HVAC)
- Lighting systems and controls
- Refrigeration systems (efficiency, refrigerant type)
- Transport and logistics energy
- Recommendations for cost-effective improvements

### 13.2 F-Gas Regulation -- Regulation 2024/573

#### Regulatory Reference
- **Regulation**: 2024/573, replacing Regulation 517/2014
- **Application**: 11 March 2024

#### Retail Refrigerant Requirements

| Requirement | Detail | Deadline |
|------------|--------|----------|
| **New stationary refrigeration equipment** | Must use refrigerant with GWP < 150 | Already in effect (2025) |
| **Market ban: plug-in commercial refrigerators/freezers** | GWP >= 150 banned (with temporary derogation to 30 Jun 2026 for artisan equipment) | 1 Jan 2025 / 30 Jun 2026 |
| **Centralized refrigeration systems (>40 kW)** | GWP < 150 for new systems | Already in effect |
| **Leak checks** | Systems with >= 5 tCO2e F-gas charge: leak checks every 6-12 months | Ongoing |
| **Leak detection** | Systems with >= 500 tCO2e charge: automatic leak detection required | Ongoing |
| **Record keeping** | F-gas logbook for each system (charge, leaks, recovery, disposal) | Ongoing |
| **Phase-down quota** | 42.9M tCO2e for 2025-2026; halved to ~21.7M tCO2e for 2027-2029 | Ongoing |
| **Full HFC phase-out** | Production phased to 15% of baseline by 2036; consumption phase-out by 2050 | 2036 / 2050 |

#### Retail Impact

- **Supermarkets/grocery**: Largest retail F-gas users (commercial refrigeration cabinets, cold rooms, freezer cases)
- **Transition requirement**: Move from R404A (GWP 3922) and R134a (GWP 1430) to natural refrigerants (CO2/R744, propane/R290, ammonia/R717) or low-GWP HFOs
- **Cost implication**: Refrigerant system retrofits or replacements for entire store fleets
- **ESRS E1 connection**: F-gas leakage is a material Scope 1 emission for grocery retailers

### 13.3 Energy Performance of Buildings Directive (EPBD)

| Requirement | Deadline | Retail Relevance |
|------------|----------|-----------------|
| Energy Performance Certificates (EPC) for all commercial buildings | Ongoing, updated requirements from 2025 | Stores, warehouses, offices |
| Zero-emission building standard for new public buildings | 2028 | New store construction |
| Zero-emission building standard for all new buildings | 2030 | New stores, warehouses |
| Minimum energy performance standards for worst-performing buildings | 2030 (15% worst commercial buildings) | Older stores requiring renovation |
| Solar energy obligation for new large commercial buildings (>250 m2) | By end 2026 | New large-format stores, warehouses |

---

## 14. R12 -- Forced Labour Regulation & Human Rights Due Diligence

### 14.1 Regulatory Reference
- **EU Forced Labour Regulation**: Regulation 2024/XX, adopted 23 April 2024
- **Entry into force**: 13 December 2024
- **Application date**: **14 December 2027**

### 14.2 Key Provisions

| Provision | Detail |
|-----------|--------|
| **Scope** | All products placed on, made available on, or exported from the EU market |
| **Coverage** | Entire lifecycle: raw material extraction, manufacturing, assembly |
| **No company size threshold** | Applies to companies of all sizes and all product types |
| **Forced labor definition** | ILO definition per Forced Labour Convention (Convention No. 29) |
| **Investigation** | Competent authorities conduct risk-based investigations |
| **Burden of proof** | On the economic operator to demonstrate products are forced-labor-free |

### 14.3 Enforcement Mechanism

| Step | Detail |
|------|--------|
| **Preliminary investigation** | Competent authority identifies indicators of forced labor risk |
| **Full investigation** | Evidence gathering, supply chain document requests |
| **Decision** | If forced labor is found: product ban, withdrawal from market, disposal |
| **Due diligence consideration** | Robust due diligence processes are taken into account (mitigating factor) |
| **EU-level coordination** | European Labour Authority + Member State authorities + customs |

### 14.4 Retail High-Risk Supply Chains

| Sector | Key Risks | Geographies of Concern |
|--------|-----------|----------------------|
| **Garments & Textiles** | Forced labor in cotton picking, spinning mills, garment factories | China (Xinjiang), Bangladesh, Myanmar, Turkey, India |
| **Food & Agriculture** | Bonded labor on plantations, fishing sector | Thailand, Brazil, India, West Africa |
| **Electronics** | Forced labor in mining (cobalt, lithium), factory assembly | DRC, China, Malaysia |
| **Footwear & Leather** | Tannery workers, informal economy | India, Bangladesh, Vietnam |
| **Timber & Furniture** | Illegal logging with forced labor | Brazil, Indonesia, Myanmar |
| **Cosmetics & Beauty** | Mica mining child labor, palm oil forced labor | India, Indonesia, Malaysia |

### 14.5 Complementary Frameworks

| Framework | Relationship |
|-----------|-------------|
| **CSDDD (Directive 2024/1760)** | Due diligence obligation covering forced labor (broader scope) |
| **ESRS S2** (Workers in Value Chain) | Reporting on value chain labor practices |
| **US Uyghur Forced Labor Prevention Act** | US counterpart; import ban on Xinjiang products |
| **UK Modern Slavery Act 2015** | Annual Modern Slavery Statement for UK operations >GBP 36M turnover |
| **German Supply Chain Due Diligence Act (LkSG)** | National implementation ahead of CSDDD |
| **French Duty of Vigilance Law** | National implementation since 2017 |

### 14.6 Penalties

| Penalty | Detail |
|---------|--------|
| **Product ban** | Products cannot be placed on/made available on EU market |
| **Market withdrawal** | Products already on market must be recalled |
| **Customs hold** | Products detained at EU external borders |
| **Disposal** | Forced donation or destruction of non-compliant products |
| **No turnover-based fines** | Enforcement is product-level (ban/withdrawal), not company-level fines |

---

## 15. Consolidated Timeline

| Date | Regulation | Milestone |
|------|-----------|-----------|
| **Already in force** | F-Gas 2024/573 | GWP < 150 for new commercial refrigeration equipment |
| **27 Mar 2026** | ECGT 2024/825 | Member State transposition deadline |
| **18 Mar 2026** | Omnibus I 2026/470 | Directive enters into force |
| **12 Aug 2026** | PPWR 2025/40 | General application (incl. labeling, EPR framework) |
| **18 Sep 2026** | Omnibus I / ESRS | EC must adopt revised ESRS delegated act |
| **27 Sep 2026** | ECGT 2024/825 | Anti-greenwashing prohibitions applicable |
| **Oct 2026** | EED 2023/1791 | Initial energy audit compliance (10-85 TJ/year) |
| **30 Dec 2026** | EUDR 2023/1115 | Application for large/medium operators |
| **1 Jan 2027** | ESRS (amended) | Amended ESRS applicable (optional early use for FY2026) |
| **Feb 2027** | EU Battery Reg 2023/1542 | Battery DPP mandatory |
| **mid-2027** | Revised WFD (textiles) | Textile EPR transposition deadline |
| **H2 2027** | ESPR textiles | Textile DPP and ecodesign requirements |
| **30 Jun 2027** | EUDR 2023/1115 | Application for micro/small enterprises |
| **Oct 2027** | EED 2023/1791 | ISO 50001 mandatory for >85 TJ/year energy users |
| **14 Dec 2027** | Forced Labour Reg | Application (product bans enforceable) |
| **mid-2028** | Revised WFD (textiles) | Textile EPR schemes operational |
| **26 Jul 2028** | CSDDD 2024/1760 | Phase 1 application (>5,000 employees, >EUR 1.5B) |
| **2028** | ESPR furniture/electronics | DPP for furniture, electronics |
| **26 Jul 2029** | CSDDD | Phase 2 (>3,000 employees, >EUR 900M) |
| **1 Jan 2030** | PPWR | Recycled content targets (30% PET, 10% other plastic) |
| **2030** | Food waste | 30% reduction vs. 2021-2023 baseline |
| **2030** | PPWR | Packaging waste reduction (-5%), reusable shipping for e-commerce |
| **2030** | EU Green Deal | 55% GHG reduction target vs. 1990 |
| **26 Jul 2030** | CSDDD | Phase 3 (>1,000 employees, >EUR 450M) |
| **1 Jan 2035** | PPWR | Packaging recyclability at scale |
| **1 Jan 2040** | PPWR | Recycled content targets increase (50-65%) |
| **2050** | F-Gas | Full HFC consumption phase-out |
| **2050** | EU Green Deal | Climate neutrality |

---

## 16. Penalty Summary

| Regulation | Maximum Penalty | Basis |
|-----------|----------------|-------|
| **CSRD** (non-reporting) | Varies by Member State: up to 3-6% of company assets (CZ), EUR 375K + 5 years imprisonment (FR) | Article 51 of Directive 2013/34/EU |
| **EUDR** (non-compliance) | Up to **4% of EU-wide annual turnover** + confiscation + market ban | Article 25, Regulation 2023/1115 |
| **CSDDD** (due diligence failure) | Up to **5% of worldwide net turnover** + civil liability | Article 27, Directive 2024/1760 |
| **ECGT** (greenwashing) | Up to **4% of annual turnover** or **EUR 2 million** (whichever higher) | Via CPC Regulation 2017/2394 |
| **F-Gas** (non-compliance) | Member State penalties; equipment bans, quota revocation | Regulation 2024/573, national implementation |
| **Forced Labour Reg** | Product ban, market withdrawal, customs detention, forced disposal | Product-level enforcement (no turnover-based fine) |
| **PPWR** | Member State penalties (to be defined); EPR scheme exclusion | Regulation 2025/40 |

---

## 17. Feature Mapping -- PACK-014 Engine Requirements

Based on the regulatory analysis above, PACK-014 must implement the following engines:

### Engine 1: Retail Scope 3 Emissions Engine
**Regulatory drivers**: ESRS E1-6, GHG Protocol Scope 3 Standard
- Category 1 (purchased goods) calculator with tiered methodology (supplier-specific / hybrid / average-data / spend-based)
- Category 4 (upstream transport) calculator with GLEC Framework
- Category 9 (downstream transport) calculator for home delivery / last-mile
- Category 11 (use of sold products) calculator for energy-consuming products
- Category 12 (end-of-life) calculator based on product material composition
- SKU-level emission factor mapping (10,000+ product categories)
- Data quality scoring (5-level)

### Engine 2: Packaging Compliance Engine
**Regulatory drivers**: PPWR 2025/40, ESRS E5
- Recycled content tracking by packaging material and type
- Gap-to-target calculation (2030 and 2040 targets)
- Packaging recyclability assessment (design-for-recycling grades A-E)
- EPR fee estimation with eco-modulation
- Packaging waste tonnage tracking and reduction trajectory
- Labeling compliance checker (material composition, sorting instructions)

### Engine 3: Deforestation Due Diligence Engine
**Regulatory drivers**: EUDR 2023/1115, ESRS E4
- Commodity-level risk assessment (7 EUDR commodities + derivatives)
- Geolocation data management (production plot coordinates)
- Country risk benchmarking (low/standard/high risk)
- Due Diligence Statement preparation for TRACES submission
- Private-label vs. third-party product differentiation
- Multi-commodity product handling

### Engine 4: Food Waste Tracking Engine
**Regulatory drivers**: Revised WFD, ESRS E5-5
- Food waste tracking by category (bakery, produce, dairy, meat, prepared)
- Baseline calculation (2021-2023 average)
- Reduction trajectory to 30% target (2030)
- Waste hierarchy tracking (redistribution > animal feed > composting > AD > incineration > landfill)
- Financial impact of food waste (cost of goods wasted, disposal costs, EPR fees)

### Engine 5: Store Energy & Refrigerant Engine
**Regulatory drivers**: EED 2023/1791, F-Gas 2024/573, ESRS E1
- Store-level energy consumption tracking (electricity, gas, heating oil)
- Energy intensity calculation (kWh/m2 sales floor, kWh/m2 total)
- Refrigerant inventory management (equipment, charge, GWP, leak rates)
- F-gas emissions calculation (Scope 1)
- EED energy audit compliance tracking
- Refrigerant transition planning (R404A/R134a to natural refrigerants)

### Engine 6: Supply Chain Due Diligence Engine
**Regulatory drivers**: CSDDD 2024/1760, Forced Labour Reg, ESRS S2
- Supplier risk scoring (human rights, environmental, forced labor)
- Tier 1-4 supplier mapping by risk category (garments, food, electronics)
- Due diligence process tracking (identify, prevent, mitigate, remediate)
- Grievance mechanism reporting
- Corrective action plan tracking
- Country/commodity risk matrix (ILO, ITUC, US DoL forced labor lists)

### Engine 7: Product Sustainability Engine
**Regulatory drivers**: ESPR 2024/1781, Textile Strategy, ECGT 2024/825
- Digital Product Passport data management (textiles, electronics, furniture)
- Ecodesign compliance assessment per product category
- Green claims substantiation checker (ECGT prohibitions)
- Textile EPR obligation calculator
- Microplastic measurement tracking (textiles)
- Product Environmental Footprint (PEF) calculation support

### Engine 8: Retail ESRS Disclosure Engine
**Regulatory drivers**: CSRD 2022/2464, ESRS Set 1 (amended), Omnibus I 2026/470
- Retail-specific double materiality assessment template
- Automated disclosure generation for all material ESRS standards
- Post-Omnibus datapoint mapping (amended ~460 datapoints)
- Cross-regulation data reuse (EUDR data for E4, PPWR data for E5, etc.)
- Audit trail and evidence linking
- XBRL/iXBRL tagging for digital submission (ESEF format)

---

## 18. What Makes Retail Different -- Summary

Retail CSRD reporting is fundamentally different from manufacturing and financial services in five key ways:

1. **Scope 3 dominance**: 80-95% of retail emissions are Scope 3, primarily from purchased goods (Cat 1). Manufacturing typically has 30-60% Scope 3 and significant Scope 1 (process emissions). Financial services have 95%+ Scope 3 but as "financed emissions" (PCAF methodology), not product-based.

2. **Product regulation complexity**: Retailers face simultaneous product-level regulations (EUDR for commodities, PPWR for packaging, ESPR/DPP for products, Textile Strategy, Food Waste targets) that manufacturing and financial services do not. A single retailer may sell products subject to 5-6 different product regulations.

3. **Consumer-facing risk**: Retailers are directly exposed to consumer-facing anti-greenwashing rules (ECGT Directive) in ways that B2B manufacturers and financial services are not. Every product label, marketing campaign, and sustainability claim is subject to scrutiny.

4. **Deep supply chain exposure**: Retail supply chains span Tier 1-4+ across garments, food, electronics, homeware -- creating exposure under CSDDD, Forced Labour Regulation, and ESRS S2 that is broader (more commodity types) though less deep (less process knowledge) than manufacturing.

5. **Circular economy centrality**: ESRS E5 (circular economy) is almost always material for retailers due to packaging waste, food waste, textile waste, and product end-of-life -- whereas for manufacturing E2 (pollution) and for financial services E1 (climate) tend to be the primary environmental standards.

---

## 19. Sources

- [Directive 2026/470 -- CSRD/CSDDD Reform (Omnibus I)](https://generationimpact.global/news/directive-eu-2026-470-csrd-csddd-reform/)
- [EU Omnibus I Directive Published -- Linklaters](https://sustainablefutures.linklaters.com/post/102mkjf/eu-omnibus-i-directive-published-in-the-official-journal-of-the-eu)
- [CSRD Omnibus Updates -- Deloitte Heads Up](https://dart.deloitte.com/USDART/home/publications/deloitte/heads-up/2026/eu-sustainability-reporting-omnibus-esrs-updates)
- [Amended ESRS Explained -- Coolset](https://www.coolset.com/academy/the-amended-esrs-what-has-changed-and-what-it-means-for-2026-csrd-reporting)
- [CSRD Under Omnibus -- Coolset](https://www.coolset.com/academy/csrd-under-omnibus-updated-scope-timelines-and-what-companies-should-do-in-2026)
- [EU Omnibus Sustainability Rules Reset -- ISS Corporate](https://www.iss-corporate.com/resources/blog/eu-sustainability-rules-reset-what-the-2026-changes-mean/)
- [Norton Rose Fulbright -- EU Omnibus Directive](https://www.nortonrosefulbright.com/en/knowledge/publications/1679488b/european-parliament-votes-to-adopt-omnibus-proposal-amending-csrd-and-cs3d)
- [EUDR Compliance Guide -- Coolset](https://www.coolset.com/academy/the-eu-deforestation-regulation-eudr-what-businesses-need-to-know-and-do)
- [EUDR Products Covered -- Coolset](https://www.coolset.com/academy/which-products-are-covered-under-the-eudr-sector-by-sector-overview)
- [EUDR Timeline Postponement -- HQTS](https://www.hqts.com/eudr-postponed/)
- [EUDR Penalties -- TracExTech](https://tracextech.com/eudr-penalties/)
- [EUDR 10 Key Things -- White & Case](https://www.whitecase.com/insight-alert/10-key-things-you-still-need-know-about-new-eu-deforestation-regulation)
- [PPWR Regulation -- European Commission](https://environment.ec.europa.eu/topics/waste-and-recycling/packaging-waste/packaging-packaging-waste-regulation_en)
- [PPWR at a Glance -- fkur](https://fkur.com/en/knowledgebase/ppwr-eu-packaging-waste-regulation/)
- [PPWR Compliance for E-Commerce -- Greenberg Traurig](https://www.gtlaw.com/en/insights/2025/8/eu-packaging-and-packaging-waste-regulation-new-compliance-requirements-for-e-commerce)
- [PPWR Recycled Content Guide -- Circularise](https://www.circularise.com/blogs/ppwr-guide-to-compliance-timelines-and-mass-balance-solutions)
- [CSDDD Overview -- GreenVision Solutions](https://greenvisionsolutions.de/en/csddd/)
- [CSDDD Timeline -- Clifford Chance](https://www.cliffordchance.com/insights/resources/blogs/business-and-human-rights-insights/2025/01/the-eu-corporate-sustainability-due-diligence-directive.html)
- [CSDDD Omnibus Changes -- Baldon Avocats](https://baldon-avocats.com/corporate-sustainability-due-diligence-directive-csddd-omnibus-key-changes-timelime/)
- [ESPR Digital Product Passport Guide -- Fluxy](https://fluxy.one/post/digital-product-passport-dpp-eu-guide-2025-2030)
- [ESPR DPP Requirements 2026 -- Caruma](https://dpp.caruma.io/eu-digital-product-passport-what-the-european-commission-requires-in-2026/)
- [ESPR Textiles -- GreenStitch](https://greenstitch.io/blogs/espr-fashion-textiles/)
- [Revised WFD Textile EPR -- European Commission](https://environment.ec.europa.eu/news/revised-waste-framework-directive-enters-force-2025-10-16_en)
- [Textile EPR Timeline -- H2 Compliance](https://h2compliance.com/eu-textile-epr-legislation-member-states-update/)
- [EU Textile Strategy -- European Commission](https://environment.ec.europa.eu/strategy/textiles-strategy_en)
- [Food Waste Targets -- European Commission](https://food.ec.europa.eu/food-safety/food-waste/eu-food-waste-relevant-legislation/food-waste-reduction-targets_en)
- [Food Waste in Europe -- European Parliament](https://www.europarl.europa.eu/topics/en/article/20240318STO19401/food-waste-in-europe-facts-eu-policies-and-2030-targets)
- [GHG Protocol Scope 3 Standard](https://ghgprotocol.org/corporate-value-chain-scope-3-standard)
- [GHG Protocol Scope 3 Calculation Guidance](https://ghgprotocol.org/scope-3-calculation-guidance-2)
- [ECGT Directive 2024/825 -- EUR-Lex](https://eur-lex.europa.eu/eli/dir/2024/825/oj/eng)
- [ECGT Anti-Greenwashing -- Sidley Austin](https://www.sidley.com/en/insights/newsupdates/2024/04/new-eu-directive-strengthens-consumer-protection-laws-on-greenwashing-and-circularity)
- [Green Claims Directive Withdrawal -- Latham & Watkins](https://www.lw.com/en/insights/european-commission-announces-intention-to-withdraw-eu-green-claims-directive-proposal)
- [Green Claims and Retail -- Vaayu](https://www.vaayu.tech/insights/green-claims-directive-and-retail-implications)
- [EED 2023/1791 Overview -- European Commission](https://energy.ec.europa.eu/topics/energy-efficiency/energy-efficiency-targets-directive-and-rules/energy-efficiency-directive_en)
- [EED and ISO 50001 -- DNV](https://www.dnv.com/assurance/Management-Systems/eu-directive-2023-1791-energy-efficiency-iso-50001/)
- [F-Gas Regulation 2024/573 -- Compliance & Risks](https://www.complianceandrisks.com/blog/regulation-eu-2024-573-european-commission-adopts-new-f-gas-regulation/)
- [F-Gas Phase-Down -- F-Gas Controls](https://www.fgascontrols.com/2025/eu-revised-fgas-regulations-explained/)
- [EU Forced Labour Ban -- White & Case](https://www.whitecase.com/insight-alert/eu-adopts-forced-labour-ban-8-things-know)
- [EU Forced Labour Regulation -- Human Rights Watch](https://www.hrw.org/news/2025/04/24/eu-new-law-requires-companies-tackle-forced-labor)
- [CSRD Penalties -- Seneca ESG](https://senecaesg.com/insights/csrd-penalties-for-non-compliance-understanding-the-stakes/)
- [CSRD Fines by Region -- Verv Energy](https://verv.energy/blog/csrd-fines-regional)
- [ESRS E5 Circular Economy -- Coolset](https://www.coolset.com/academy/esrs-e5-circular-economy)
- [ESRS Double Materiality -- PwC](https://www.pwc.com/us/en/services/esg/library/csrd-disclosure-requirements.html)
- [ESRS Materiality Topics -- Stratecta](https://www.stratecta.exchange/the-94-topics-of-the-materiality-analysis-of-csrd/)

---

*Document generated by GL-RegulatoryIntelligence on 2026-03-16. This regulatory analysis reflects regulations and guidance available as of this date. Regulatory requirements are subject to change through delegated acts, implementing acts, and Member State transposition. Organizations should monitor EFRAG, European Commission, and national authority publications for updates.*
