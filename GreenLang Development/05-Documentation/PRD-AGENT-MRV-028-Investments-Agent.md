# PRD: AGENT-MRV-028 -- Investments (Scope 3 Category 15) Agent

## Document Info
| Field | Value |
|-------|-------|
| PRD ID | PRD-AGENT-MRV-028 |
| Agent ID | GL-MRV-S3-015 |
| Component | AGENT-MRV-028 |
| Category | GHG Protocol Scope 3, Category 15 |
| Version | 1.0.0 |
| Status | Approved |
| Author | GL-ProductManager + GL-RegulatoryIntelligence |
| Date | 2026-02-28 |

---

## 1. Overview

### 1.1 Purpose
Build a production-grade Investments (Category 15) agent that calculates GHG emissions associated with the reporting company's investments in the reporting year not already included in Scope 1 and Scope 2. This is the **final** Scope 3 downstream category and is particularly relevant for **financial institutions** (banks, asset managers, insurance companies, pension funds, private equity, venture capital).

### 1.2 Scope Boundary
- **Category 15**: Emissions from investments not included in Scope 1/Scope 2
- **Reporter role**: INVESTOR (company making equity or debt investments)
- **Applicable to**: Financial institutions, holding companies, PE/VC firms, sovereign wealth funds, any company with significant equity investments
- **NOT in scope**: Investments already consolidated in Scope 1/2 (based on consolidation approach)
- **Boundary rule**: If the reporting company has **operational control** AND uses operational control consolidation, investee emissions are in Scope 1/2. If using **financial control** or **equity share**, minority investments are Cat 15.

### 1.3 Key Distinction: Cat 15 vs Other Categories
| Aspect | Cat 15 (Investments) | Cat 14 (Franchises) | Cat 13 (Downstream Leased) |
|--------|---------------------|---------------------|---------------------------|
| Reporter | Investor | Franchisor | Lessor |
| Relationship | Equity/debt ownership | Franchise agreement | Lease agreement |
| Emission basis | Investee's total emissions x equity share | Franchisee Scope 1+2 | Asset-level operations |
| Typical reporters | Banks, asset managers, PE/VC | Restaurant/hotel chains | REITs, fleet companies |
| Standard | PCAF, GHG Protocol | GHG Protocol | GHG Protocol |

### 1.4 Investment Types (8 asset classes)
1. **Listed equity & corporate bonds** -- Publicly traded securities
2. **Private equity (PE)** -- Direct ownership in private companies
3. **Project finance** -- Infrastructure, renewable energy, real estate development
4. **Commercial real estate (CRE)** -- Direct property investments
5. **Mortgages** -- Residential and commercial mortgage portfolios
6. **Motor vehicle loans** -- Auto lending portfolios
7. **Sovereign bonds** -- Government debt instruments
8. **Business loans & unlisted equity** -- SME lending, unlisted corporate

### 1.5 PCAF (Partnership for Carbon Accounting Financials)
PCAF is THE global standard for financial institutions measuring Cat 15. The GHG Protocol Scope 3 standard references PCAF as the preferred methodology. PCAF provides:
- **Attribution factors** by asset class
- **Data quality scoring** (1-5 scale, 1 = best)
- **Financed emissions** = SUM [ attribution_factor x investee_emissions ]
- **Carbon intensity metrics** (tCO2e per $M invested, per $M revenue)

---

## 2. Regulatory Requirements

### 2.1 GHG Protocol Scope 3 Standard (Chapter 15)
- Report emissions from equity investments where reporting company does NOT have operational control
- Attribution based on **equity share** or **investment share**
- 4 investment types explicitly covered: equity investments, debt investments, project finance, managed investments
- Data hierarchy: investee-reported > estimated > EEIO

### 2.2 PCAF Standard (Global GHG Accounting and Reporting Standard)
- **6 asset classes** with specific attribution formulas
- **Data quality score** 1-5 per asset class
- **Attribution factor** = outstanding_amount / (EVIC or total_balance_sheet or property_value)
- **Listed equity/bonds**: AF = outstanding / EVIC
- **Private equity**: AF = outstanding / (total_equity + total_debt)
- **CRE**: AF = outstanding / property_value
- **Mortgages**: AF = outstanding / property_value
- **Motor vehicle loans**: AF = outstanding / vehicle_value
- **Sovereign bonds**: AF = outstanding / adjusted_GDP (PPP)

### 2.3 Compliance Frameworks
| Framework | Requirement |
|-----------|-------------|
| GHG Protocol Scope 3 | Category 15 mandatory if material; equity share attribution |
| PCAF Global Standard | 6 asset classes, data quality 1-5, attribution factors |
| ISO 14064-1:2018 | Clause 5.2.4 indirect GHG emissions; organizational boundaries |
| CSRD / ESRS E1 | E1-6 GHG Scope 3 downstream; financed emissions for financial entities |
| CDP Climate Change | C6.5 Category 15; C-FS14.1 financed emissions |
| SBTi Financial Institutions | Sector-specific target setting; portfolio alignment |
| SB 253 (California) | All material Scope 3 categories; assurance |
| TCFD | Metrics & Targets; financed emissions disclosure |
| NZBA/NZAOA | Net-Zero Banking/Asset Owner Alliance sector targets |

---

## 3. Architecture

### 3.1 Seven-Engine Design

| # | Engine | Class | Responsibility |
|---|--------|-------|---------------|
| 1 | Investment Database | `InvestmentDatabaseEngine` | Asset class EFs, PCAF data quality scores, sector EFs, country EFs |
| 2 | Equity Investment Calculator | `EquityInvestmentCalculatorEngine` | Listed equity, private equity, unlisted equity (Tier 1) |
| 3 | Debt Investment Calculator | `DebtInvestmentCalculatorEngine` | Corporate bonds, business loans, project finance (Tier 2) |
| 4 | Real Asset Calculator | `RealAssetCalculatorEngine` | CRE, mortgages, motor vehicle loans (Tier 3) |
| 5 | Sovereign Bond Calculator | `SovereignBondCalculatorEngine` | Government bond emissions via GDP attribution (Tier 4) |
| 6 | Compliance Checker | `ComplianceCheckerEngine` | 9 frameworks, DC rules, PCAF data quality |
| 7 | Pipeline | `InvestmentsPipelineEngine` | 10-stage orchestration |

### 3.2 PCAF Attribution Formulas

#### Listed Equity & Corporate Bonds
```
attribution_factor = outstanding_amount / EVIC
financed_emissions = attribution_factor x company_emissions (Scope 1 + Scope 2)
```
Where EVIC = Enterprise Value Including Cash = market_cap + total_debt

#### Private Equity / Unlisted Equity
```
attribution_factor = outstanding_amount / (total_equity + total_debt)
financed_emissions = attribution_factor x company_emissions
```

#### Project Finance
```
attribution_factor = outstanding_amount / total_project_cost
financed_emissions = attribution_factor x project_emissions
```

#### Commercial Real Estate
```
attribution_factor = outstanding_amount / property_value_at_origination
financed_emissions = attribution_factor x building_emissions
```
Building emissions = floor_area x EUI x grid_EF (or actual energy data)

#### Mortgages
```
attribution_factor = outstanding_loan / property_value_at_origination
financed_emissions = attribution_factor x building_emissions
```

#### Motor Vehicle Loans
```
attribution_factor = outstanding_loan / vehicle_value_at_origination
financed_emissions = attribution_factor x vehicle_annual_emissions
```

#### Sovereign Bonds
```
attribution_factor = outstanding_amount / PPP_adjusted_GDP
financed_emissions = attribution_factor x country_emissions (production-based)
```
Country emissions = total GHG minus LULUCF

### 3.3 PCAF Data Quality Scores (1-5, 1=best)

| Score | Listed Equity/Bonds | Private Equity | CRE | Mortgages | Motor Vehicles | Sovereign |
|-------|-------------------|----------------|-----|-----------|----------------|-----------|
| 1 | Verified Scope 1+2 | Verified data | Actual energy | Actual energy | Actual fuel | N/A |
| 2 | Reported unverified | Reported data | Cert/rating | Cert/rating | Reported specs | N/A |
| 3 | Physical activity | Revenue + EF | Floor area EUI | Floor area EUI | Make/model/year | N/A |
| 4 | Revenue EEIO | Revenue EEIO | Avg per m2 | Avg per building | Avg per segment | Production |
| 5 | Sector average | Sector average | Avg per asset | Avg per property | Avg per vehicle | PPP GDP |

### 3.4 Double-Counting Prevention Rules

| Rule ID | Description |
|---------|-------------|
| DC-INV-001 | Investments consolidated in Scope 1/2 (operational control) are NOT Cat 15 |
| DC-INV-002 | Equity share already used for consolidation -- do not double count |
| DC-INV-003 | Fund-of-funds: look through to underlying investments, avoid counting intermediate |
| DC-INV-004 | CRE vs Cat 8/13: if same property in Cat 8 (lessee) or Cat 13 (lessor), do not also count as Cat 15 |
| DC-INV-005 | Sovereign bonds vs corporate: national emissions include corporates; do not add both |
| DC-INV-006 | Multi-asset class: same company across equity + debt -- use higher quality data, count once |
| DC-INV-007 | Managed investments: if underlying assets already counted, do not double at fund level |
| DC-INV-008 | Short positions: exclude from financed emissions (no economic exposure) |

---

## 4. Data Models

### 4.1 Enumerations (22)
1. `AssetClass` (8): listed_equity, corporate_bond, private_equity, project_finance, commercial_real_estate, mortgage, motor_vehicle_loan, sovereign_bond
2. `InvestmentType` (4): equity, debt, fund, derivative
3. `SectorClassification` (12): energy, materials, industrials, consumer_discretionary, consumer_staples, healthcare, financials, information_technology, communication_services, utilities, real_estate, other
4. `PCAFDataQuality` (5): score_1, score_2, score_3, score_4, score_5
5. `CalculationMethod` (5): reported_emissions, physical_activity, revenue_eeio, sector_average, asset_specific
6. `ConsolidationApproach` (3): operational_control, financial_control, equity_share
7. `EmissionScope` (4): scope_1, scope_2, scope_1_2, scope_1_2_3
8. `PropertyType` (6): office, retail, industrial, residential, mixed_use, hospitality
9. `VehicleCategory` (5): passenger_car, light_commercial, heavy_commercial, motorcycle, electric_vehicle
10. `FuelStandard` (4): euro_6, us_tier_3, china_6, other
11. `EFSource` (7): CDP, DEFRA_2024, EPA_2024, IEA_2024, PCAF_2024, EXIOBASE, CUSTOM
12. `CurrencyCode` (15): USD, EUR, GBP, JPY, CHF, CAD, AUD, CNY, KRW, SGD, HKD, SEK, NOK, DKK, BRL
13. `DataQualityTier` (3): tier_1, tier_2, tier_3
14. `DQIDimension` (5): temporal, geographical, technological, completeness, reliability
15. `ComplianceFramework` (9): ghg_protocol, pcaf, iso_14064, csrd_esrs, cdp, sbti_fi, sb_253, tcfd, nzba
16. `ComplianceStatus` (4): compliant, non_compliant, partial, not_applicable
17. `PipelineStage` (10): validate, classify, normalize, resolve_efs, calculate, allocate, aggregate, compliance, provenance, seal
18. `UncertaintyMethod` (3): monte_carlo, analytical, pcaf_data_quality
19. `BatchStatus` (4): pending, processing, completed, failed
20. `GWPSource` (2): AR5, AR6
21. `PortfolioAlignment` (4): aligned_1_5c, aligned_2c, not_aligned, unknown
22. `SBTiTarget` (3): near_term, long_term, net_zero

### 4.2 Constant Tables (16)
1. `PCAF_ATTRIBUTION_RULES` -- 8 asset classes with attribution formula descriptions
2. `SECTOR_EMISSION_FACTORS` -- 12 GICS sectors x average tCO2e/$M revenue
3. `COUNTRY_EMISSION_FACTORS` -- 50+ countries: total GHG (excl LULUCF), GDP PPP
4. `GRID_EMISSION_FACTORS` -- 12 countries + 26 eGRID subregions
5. `BUILDING_EUI_BENCHMARKS` -- 6 property types x 5 climate zones (kWh/m2)
6. `VEHICLE_EMISSION_FACTORS` -- 5 vehicle categories x average annual emissions
7. `EEIO_SECTOR_FACTORS` -- 12 sectors (kgCO2e/$ revenue) from EXIOBASE
8. `PCAF_DATA_QUALITY_MATRIX` -- 8 asset classes x 5 quality scores with criteria
9. `CURRENCY_CONVERSION_RATES` -- 15 currencies to USD (base year 2024)
10. `SOVEREIGN_COUNTRY_DATA` -- 50 countries: GDP PPP, total emissions, per capita
11. `CARBON_INTENSITY_BENCHMARKS` -- By sector (tCO2e/$M revenue, tCO2e/MWh, etc.)
12. `DC_RULES` -- 8 double-counting rules (DC-INV-001 through DC-INV-008)
13. `COMPLIANCE_FRAMEWORK_RULES` -- 9 frameworks with requirements
14. `DQI_SCORING` -- 5 dimensions x PCAF score mapping
15. `UNCERTAINTY_RANGES` -- 5 PCAF scores -> uncertainty percentage
16. `PORTFOLIO_ALIGNMENT_THRESHOLDS` -- Temperature alignment thresholds

### 4.3 Pydantic Models (16 frozen)
1. `EquityInvestmentInput` -- Listed/unlisted: company_name, ISIN/ticker, outstanding_amount, EVIC, company_emissions, sector, country
2. `DebtInvestmentInput` -- Bonds/loans: company_name, outstanding_amount, total_equity_plus_debt, company_emissions, sector
3. `ProjectFinanceInput` -- Projects: project_name, outstanding_amount, total_project_cost, project_emissions, project_type
4. `CREInvestmentInput` -- Real estate: property_type, outstanding_amount, property_value, floor_area_m2, energy_data, location
5. `MortgageInput` -- Mortgages: outstanding_loan, property_value, property_type, floor_area_m2, energy_cert_rating
6. `MotorVehicleLoanInput` -- Auto loans: outstanding_loan, vehicle_value, vehicle_category, make_model, annual_distance_km
7. `SovereignBondInput` -- Government bonds: country_code, outstanding_amount, GDP_PPP
8. `PortfolioInput` -- Full portfolio: list of all asset class inputs, reporting_period
9. `InvestmentCalculationResult` -- Per-investment: financed_emissions_kgco2e, attribution_factor, pcaf_data_quality, intensity
10. `PortfolioAggregationResult` -- Portfolio: total_financed_emissions, by_asset_class, by_sector, by_country, WACI
11. `ComplianceResult` -- framework, status, findings, recommendations
12. `ProvenanceRecord` -- SHA-256 hash chain record
13. `DataQualityScore` -- PCAF 1-5 + 5-dimension DQI
14. `UncertaintyResult` -- PCAF-linked uncertainty, CI bounds
15. `CarbonIntensityResult` -- WACI, revenue intensity, physical intensity
16. `PortfolioAlignmentResult` -- Temperature alignment, SBTi status

---

## 5. Database Schema (V079)

### 5.1 Tables (22)

**Reference Tables (10):**
1. `gl_inv_sector_emission_factors` -- Sector-level average EFs
2. `gl_inv_country_emission_factors` -- Country total GHG, GDP PPP
3. `gl_inv_grid_emission_factors` -- Country/region grid EFs
4. `gl_inv_building_benchmarks` -- Property type EUI by climate zone
5. `gl_inv_vehicle_emission_factors` -- Vehicle category annual EFs
6. `gl_inv_eeio_sector_factors` -- EEIO by sector
7. `gl_inv_pcaf_data_quality` -- PCAF DQ criteria per asset class
8. `gl_inv_currency_rates` -- FX rates to USD
9. `gl_inv_sovereign_data` -- Country GDP, emissions, per capita
10. `gl_inv_carbon_benchmarks` -- Sector carbon intensity benchmarks

**Operational Tables (9):**
11. `gl_inv_calculations` -- **HYPERTABLE** (7-day): Main calculation records
12. `gl_inv_investment_results` -- Per-investment calculation details
13. `gl_inv_portfolio_aggregations` -- Portfolio-level aggregations
14. `gl_inv_compliance_checks` -- **HYPERTABLE** (30-day): Compliance results
15. `gl_inv_aggregations` -- **HYPERTABLE** (30-day): Period aggregations
16. `gl_inv_provenance_records` -- SHA-256 hash chains
17. `gl_inv_audit_trail` -- Operation audit log
18. `gl_inv_batch_jobs` -- Batch processing status
19. `gl_inv_portfolio_positions` -- Investment position tracking

**Supporting Tables (3):**
20. `gl_inv_data_quality_scores` -- PCAF 1-5 + DQI
21. `gl_inv_uncertainty_results` -- Uncertainty analysis results
22. `gl_inv_carbon_intensity` -- WACI and intensity metrics

---

## 6. API Design

### 6.1 Endpoints (24 at `/api/v1/investments`)

**Calculation Endpoints (12 POST):**
1. `POST /calculate` -- Full portfolio pipeline
2. `POST /calculate/equity` -- Listed equity investment
3. `POST /calculate/private-equity` -- Private equity investment
4. `POST /calculate/corporate-bond` -- Corporate bond
5. `POST /calculate/project-finance` -- Project finance
6. `POST /calculate/commercial-real-estate` -- CRE investment
7. `POST /calculate/mortgage` -- Mortgage portfolio
8. `POST /calculate/motor-vehicle-loan` -- Auto loan portfolio
9. `POST /calculate/sovereign-bond` -- Sovereign bond
10. `POST /calculate/batch` -- Batch (up to 50,000 positions)
11. `POST /calculate/portfolio` -- Full portfolio analysis
12. `POST /compliance/check` -- Multi-framework compliance check

**Data Retrieval (12 GET + 1 DELETE):**
13. `GET /calculations/{id}` -- Get calculation detail
14. `GET /calculations` -- List calculations
15. `DELETE /calculations/{id}` -- Soft-delete
16. `GET /emission-factors/{asset_class}` -- EFs by asset class
17. `GET /sector-factors` -- Sector emission factors
18. `GET /country-factors` -- Country emission factors
19. `GET /pcaf-quality` -- PCAF data quality criteria
20. `GET /carbon-intensity` -- Portfolio carbon intensity metrics
21. `GET /portfolio-alignment` -- Temperature alignment
22. `GET /aggregations` -- Time-series aggregations
23. `GET /provenance/{id}` -- Provenance chain
24. `GET /health` -- Health check

---

## 7. File Structure

### 7.1 Source Files (15)
```
greenlang/investments/
    __init__.py                         (~140 lines)
    models.py                           (~2,500 lines)
    config.py                           (~2,400 lines)
    metrics.py                          (~1,300 lines)
    provenance.py                       (~1,600 lines)
    investment_database.py              (~2,600 lines)
    equity_investment_calculator.py     (~2,200 lines)
    debt_investment_calculator.py       (~2,100 lines)
    real_asset_calculator.py            (~2,200 lines)
    sovereign_bond_calculator.py        (~1,800 lines)
    compliance_checker.py               (~2,800 lines)
    investments_pipeline.py             (~1,800 lines)
    setup.py                            (~1,500 lines)
    api/
        __init__.py                     (~1 line)
        router.py                       (~3,200 lines)
```

### 7.2 Test Files (14)
```
tests/unit/mrv/test_investments/
    __init__.py, conftest.py, test_models.py, test_config.py,
    test_investment_database.py, test_equity_investment_calculator.py,
    test_debt_investment_calculator.py, test_real_asset_calculator.py,
    test_sovereign_bond_calculator.py, test_compliance_checker.py,
    test_investments_pipeline.py, test_provenance.py, test_setup.py, test_api.py
```

### 7.3 Migration
```
deployment/database/migrations/sql/V079__investments_service.sql (~1,200 lines)
```

---

## 8. Auth Integration

### 8.1 Permission Map (24 entries)
```
POST:/api/v1/investments/calculate                              -> investments:calculate
POST:/api/v1/investments/calculate/equity                       -> investments:calculate
POST:/api/v1/investments/calculate/private-equity               -> investments:calculate
POST:/api/v1/investments/calculate/corporate-bond               -> investments:calculate
POST:/api/v1/investments/calculate/project-finance              -> investments:calculate
POST:/api/v1/investments/calculate/commercial-real-estate       -> investments:calculate
POST:/api/v1/investments/calculate/mortgage                     -> investments:calculate
POST:/api/v1/investments/calculate/motor-vehicle-loan           -> investments:calculate
POST:/api/v1/investments/calculate/sovereign-bond               -> investments:calculate
POST:/api/v1/investments/calculate/batch                        -> investments:calculate
POST:/api/v1/investments/calculate/portfolio                    -> investments:calculate
POST:/api/v1/investments/compliance/check                       -> investments:compliance
GET:/api/v1/investments/calculations/{id}                       -> investments:read
GET:/api/v1/investments/calculations                            -> investments:read
DELETE:/api/v1/investments/calculations/{id}                    -> investments:delete
GET:/api/v1/investments/emission-factors/{asset_class}          -> investments:read
GET:/api/v1/investments/sector-factors                          -> investments:read
GET:/api/v1/investments/country-factors                         -> investments:read
GET:/api/v1/investments/pcaf-quality                            -> investments:read
GET:/api/v1/investments/carbon-intensity                        -> investments:read
GET:/api/v1/investments/portfolio-alignment                     -> investments:read
GET:/api/v1/investments/aggregations                            -> investments:read
GET:/api/v1/investments/provenance/{id}                         -> investments:read
GET:/api/v1/investments/health                                  -> investments:read
```

---

## 9. Acceptance Criteria

1. All 15 source files with production-quality code
2. All 14 test files with 650+ tests
3. V079 migration with 22 tables, 3 hypertables, 2 continuous aggregates
4. Auth integration: 24 permission entries + router registration
5. Zero-hallucination: all calculations use deterministic Decimal arithmetic
6. 8 PCAF asset classes with correct attribution formulas
7. 5-level PCAF data quality scoring
8. WACI (Weighted Average Carbon Intensity) portfolio metric
9. Temperature alignment / portfolio alignment support
10. 8 double-counting prevention rules enforced
11. 9 compliance frameworks validated
12. SHA-256 provenance chains with Merkle trees
13. Thread-safe singletons on all engines
14. Memory files updated with MRV-028 entry
