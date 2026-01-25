# EU Taxonomy Regulation - Financial Institutions Requirements
**Regulation (EU) 2020/852**
**Compliance Deadline: January 1, 2026 (Extended requirements)**

## Executive Summary
The EU Taxonomy is a classification system establishing a list of environmentally sustainable economic activities, requiring financial market participants and large companies to disclose how and to what extent their activities align with the taxonomy.

## Key Requirements and Compliance Criteria

### 1. Six Environmental Objectives
1. **Climate change mitigation**
2. **Climate change adaptation**
3. **Sustainable use of water and marine resources**
4. **Transition to circular economy**
5. **Pollution prevention and control**
6. **Protection of biodiversity and ecosystems**

### 2. Taxonomy Alignment Criteria
- **Substantial contribution**: To at least one environmental objective
- **Do No Significant Harm (DNSH)**: To other five objectives
- **Minimum safeguards**: OECD Guidelines, UN Guiding Principles
- **Technical Screening Criteria (TSC)**: Sector-specific thresholds

### 3. Reporting Obligations
- **Eligibility**: Economic activities covered by taxonomy
- **Alignment**: Activities meeting all criteria
- **KPIs**: Turnover, CapEx, OpEx percentages

## Reporting Templates and Formats

### Financial Institution Disclosure Template
```json
{
  "reporting_entity": {
    "name": "Financial Institution Name",
    "lei_code": "12345678901234567890",
    "reporting_period": "2025",
    "total_assets": 50000000000
  },
  "green_asset_ratio": {
    "total_gar": 0.35,
    "breakdown": {
      "retail_mortgages": {
        "exposure": 10000000000,
        "eligible": 0.60,
        "aligned": 0.45
      },
      "corporate_loans": {
        "exposure": 15000000000,
        "eligible": 0.40,
        "aligned": 0.30
      },
      "sovereign_exposures": {
        "exposure": 5000000000,
        "eligible": 0.20,
        "aligned": 0.15
      }
    }
  },
  "investment_kpis": {
    "equity_holdings": {
      "total_value": 8000000000,
      "taxonomy_aligned": 0.42
    },
    "debt_securities": {
      "total_value": 12000000000,
      "taxonomy_aligned": 0.38
    }
  },
  "sectoral_exposure": {
    "energy": {
      "exposure": 3000000000,
      "renewable_energy": 0.65,
      "fossil_gas": 0.20,
      "nuclear": 0.15
    },
    "buildings": {
      "exposure": 5000000000,
      "energy_efficient": 0.55,
      "renovation": 0.30
    },
    "transport": {
      "exposure": 2000000000,
      "clean_vehicles": 0.40,
      "infrastructure": 0.35
    }
  }
}
```

### Corporate Taxonomy Alignment Template
```json
{
  "taxonomy_kpis": {
    "turnover": {
      "total": 1000000000,
      "eligible": 600000000,
      "aligned": 350000000,
      "percentage_aligned": 35
    },
    "capex": {
      "total": 100000000,
      "eligible": 70000000,
      "aligned": 45000000,
      "percentage_aligned": 45
    },
    "opex": {
      "total": 200000000,
      "eligible": 120000000,
      "aligned": 60000000,
      "percentage_aligned": 30
    }
  },
  "activity_breakdown": [
    {
      "activity_code": "3.1",
      "description": "Manufacture of renewable energy technologies",
      "turnover": 150000000,
      "substantial_contribution": "climate_mitigation",
      "dnsh_compliance": true,
      "minimum_safeguards": true
    }
  ]
}
```

## Data Requirements and Sources

### For Financial Institutions

#### Portfolio Data
- Loan-by-loan mortgage data with EPC ratings
- Corporate lending NACE codes
- Counterparty taxonomy disclosures
- Project finance details
- Investment portfolio compositions

#### Assessment Data
- Building energy performance certificates
- Vehicle emission data
- Renewable energy certificates
- Corporate sustainability reports
- Third-party ESG ratings

### For Corporates

#### Activity Data
- Revenue by NACE code
- Capital expenditure projects
- Operating expense categories
- R&D investments
- Asset classifications

#### Environmental Performance Data
- GHG emissions (Scopes 1, 2, 3)
- Energy consumption and sources
- Water usage and discharge
- Waste generation and treatment
- Biodiversity impacts

## Calculation Methodologies

### Green Asset Ratio (GAR) for Banks
```
GAR = Taxonomy-aligned exposures / Total covered assets

Where covered assets exclude:
- Exposures to central banks
- Trading portfolio
- Interbank loans < 1 year
- SMEs not required to report
```

### Turnover KPI Calculation
```
Turnover KPI = Revenue from taxonomy-aligned activities / Total revenue

Alignment requires:
1. Activity in taxonomy (eligible)
2. Meets technical screening criteria
3. Complies with DNSH
4. Meets minimum safeguards
```

### CapEx KPI Calculation
```
CapEx KPI = Taxonomy-aligned CapEx / Total CapEx

Includes:
- Additions to tangible and intangible assets
- Additions from business combinations
- CapEx plans for taxonomy alignment
```

### Mandatory Look-Through for Funds
```
Fund Alignment = Σ(Investment Weight × Company Alignment)

Required for:
- Equity funds
- Bond funds
- Mixed funds
```

## Technical Screening Criteria Examples

### Energy - Solar PV (4.1)
**Substantial Contribution**: Generates electricity from solar PV
**DNSH Criteria**:
- Climate adaptation assessment
- Water stress assessment for floating solar
- Circular economy: Design for recycling
- Pollution: Comply with RoHS Directive
- Biodiversity: Environmental Impact Assessment

### Buildings - Acquisition (7.7)
**Substantial Contribution**:
- Built before 2021: EPC class A or top 15%
- Built after 2021: NZEB - 10%
**DNSH Criteria**:
- Climate adaptation plan
- Water flow devices installed
- No high-concern materials
- Biodiversity assessment for greenfield

### Transport - Passenger Cars (6.5)
**Substantial Contribution**:
- Zero emissions (electric, hydrogen)
- <50g CO2/km until 2025
**DNSH Criteria**:
- Tyre noise and rolling resistance
- Circular design principles
- Pollution standards compliance

## Penalties for Non-Compliance

### Financial Penalties
- **Administrative fines**: Up to €10 million or 2% of annual turnover
- **Disgorgement**: Of profits from non-compliance
- **Periodic penalty payments**: For continued non-compliance

### Supervisory Measures
- Public statements of non-compliance
- Temporary prohibition of activities
- Management accountability measures
- Enhanced supervision requirements

### Reputational Impacts
- Exclusion from sustainable finance indices
- Loss of green bond eligibility
- Reduced access to EU funding
- Investor divestment risk

## Scope - Which Companies Must Comply

### Financial Market Participants
- **Credit institutions** (banks)
- **Investment firms**
- **Asset managers**
- **Insurance companies**
- **Pension funds**
- **Alternative investment fund managers**

### Large Companies
- **Listed companies**: >500 employees
- **Banks and insurers**: >500 employees
- **From 2025**: All CSRD-scope companies
  - Large companies meeting 2 of 3:
    - >250 employees
    - >€50M turnover
    - >€25M balance sheet

### Product Level Requirements
- **SFDR Article 8 products**: Light green
- **SFDR Article 9 products**: Dark green
- **EU Green Bonds**: 100% alignment required
- **Banking products**: Mortgages, auto loans, corporate lending

## Implementation Requirements

### Data Management Infrastructure
1. **Data Collection Systems**
   - Automated NACE code mapping
   - EPC database integration
   - Counterparty data portals
   - Third-party data feeds

2. **Calculation Engines**
   - TSC assessment algorithms
   - DNSH verification tools
   - KPI calculation modules
   - Look-through analytics

3. **Reporting Systems**
   - XBRL taxonomy reporting
   - Regulatory submission tools
   - Public disclosure websites
   - Audit trail maintenance

### Key Implementation Steps

#### For Financial Institutions
1. Portfolio mapping to economic activities
2. Data gap analysis and remediation
3. Counterparty engagement for data
4. Proxy methodology development
5. IT system implementation
6. Calculation methodology validation
7. Internal controls establishment
8. External assurance preparation

#### For Corporates
1. Activity mapping to taxonomy
2. Technical screening assessment
3. DNSH evaluation
4. Minimum safeguards review
5. CapEx plan development
6. Data collection systems
7. KPI calculation
8. Disclosure preparation

## Related Regulations
- **CSRD**: Sustainability reporting framework
- **SFDR**: Sustainable finance disclosures
- **EU Green Bond Standard**: 100% taxonomy alignment
- **Banking Package**: Prudential treatment
- **MiFID II**: Sustainability preferences