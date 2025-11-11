# California SB 253 - Climate Corporate Data Accountability Act
**California Senate Bill 253**
**Compliance Deadline: June 30, 2026 (for 2025 data)**

## Executive Summary
SB 253 requires large companies doing business in California to publicly disclose their greenhouse gas emissions across all scopes, making California's climate disclosure requirements among the most comprehensive globally.

## Key Requirements and Compliance Criteria

### 1. Reporting Requirements
- **Scope 1 Emissions**: Direct emissions from owned/controlled sources
- **Scope 2 Emissions**: Indirect emissions from purchased energy
- **Scope 3 Emissions**: All other indirect emissions in value chain

### 2. Compliance Timeline
- **2026**: Report 2025 Scope 1 & 2 emissions
- **2027**: Report 2026 Scope 1, 2 & 3 emissions
- **Annual reporting** thereafter

### 3. Verification Requirements
- **Scope 1 & 2**: Limited assurance by 2026, reasonable assurance by 2030
- **Scope 3**: Limited assurance by 2030
- **Third-party verification** required from approved providers

## Reporting Templates and Formats

### GHG Emissions Report Structure
```json
{
  "reporting_entity": {
    "legal_name": "Company Name",
    "ca_business_id": "XXXXXXX",
    "fiscal_year": "2025",
    "revenue": 1500000000,
    "headquarters": "Address"
  },
  "emissions_data": {
    "scope_1": {
      "total_mtco2e": 50000,
      "breakdown": {
        "stationary_combustion": 30000,
        "mobile_combustion": 15000,
        "process_emissions": 3000,
        "fugitive_emissions": 2000
      }
    },
    "scope_2": {
      "total_mtco2e": 40000,
      "market_based": 35000,
      "location_based": 40000,
      "renewable_energy_credits": 5000
    },
    "scope_3": {
      "total_mtco2e": 500000,
      "categories": {
        "1_purchased_goods": 200000,
        "2_capital_goods": 50000,
        "3_fuel_energy": 30000,
        "4_upstream_transport": 40000,
        "5_waste": 10000,
        "6_business_travel": 5000,
        "7_employee_commuting": 15000,
        "8_upstream_leased": 0,
        "9_downstream_transport": 50000,
        "10_product_processing": 20000,
        "11_product_use": 60000,
        "12_product_disposal": 10000,
        "13_downstream_leased": 5000,
        "14_franchises": 0,
        "15_investments": 5000
      }
    }
  },
  "methodology": {
    "protocol": "GHG Protocol Corporate Standard",
    "boundaries": "Operational Control",
    "emission_factors": "EPA, IPCC, Industry-specific"
  }
}
```

## Data Requirements and Sources

### Required Data Points

#### Scope 1 Data
- Fuel consumption (natural gas, diesel, gasoline)
- Refrigerant leakage
- Industrial process emissions
- Company vehicle fleet data

#### Scope 2 Data
- Electricity consumption (kWh)
- Steam, heating, cooling purchases
- Renewable energy certificates
- Power purchase agreements

#### Scope 3 Data (15 Categories)
1. Purchased goods and services
2. Capital goods
3. Fuel and energy-related activities
4. Upstream transportation
5. Waste generated
6. Business travel
7. Employee commuting
8. Upstream leased assets
9. Downstream transportation
10. Processing of sold products
11. Use of sold products
12. End-of-life treatment
13. Downstream leased assets
14. Franchises
15. Investments

### Data Sources
- Utility bills and energy management systems
- Fuel purchase records
- Travel booking systems
- Supplier emissions data
- Industry emission factors (EPA, IPCC)
- Economic input-output models

## Calculation Methodologies

### Scope 1 Calculations
```
Scope 1 Emissions = Σ(Activity Data × Emission Factor × GWP)

Example:
Natural Gas = Consumption (therms) × 0.00531 mtCO2e/therm
Vehicle Fleet = Miles Driven × Fuel Economy × Emission Factor
```

### Scope 2 Calculations
```
Location-Based = Electricity (MWh) × Grid Emission Factor
Market-Based = Electricity (MWh) × Supplier Emission Factor - RECs
```

### Scope 3 Calculations
```
Category 1 (Purchased Goods):
Emissions = Σ(Spend ($) × Economic Emission Factor) OR
Emissions = Σ(Quantity × Physical Emission Factor)

Category 11 (Use of Products):
Emissions = Products Sold × Lifetime Energy Use × Grid Factor
```

## Penalties for Non-Compliance

### Administrative Penalties
- **First violation**: $500,000 per year
- **Subsequent violations**: Up to $500,000 per year
- **Late filing**: $10,000 per month (max $500,000)

### Enforcement
- California Air Resources Board (CARB) enforcement
- Civil penalties through Attorney General
- Public disclosure of non-compliance

### Safe Harbor Provisions
- Scope 3 reporting based on reasonable estimates
- Good faith effort defense
- Industry average data acceptable where specific data unavailable

## Scope - Which Companies Must Comply

### Revenue Threshold
- **Required**: Annual revenues exceeding $1 billion
- **"Doing business in California"**:
  - Sales exceeding $610,395 in California
  - Real/tangible property exceeding $61,040
  - Payroll exceeding $61,040

### Entity Types
- Corporations
- LLCs
- Partnerships
- Other business entities

### Exemptions
- Companies below $1 billion revenue threshold
- Government entities
- Non-profit organizations (unclear, pending clarification)

### Parent Company Reporting
- Consolidated reporting allowed
- Subsidiary data must be included
- Clear attribution required

## Implementation Requirements

### Technical Systems Needed
1. **Data Management Platform**
   - Automated data collection
   - API integrations with suppliers
   - Data quality validation

2. **Calculation Engine**
   - GHG Protocol compliant
   - Multiple calculation methodologies
   - Emission factor database

3. **Reporting System**
   - CARB format compliance
   - Third-party verification support
   - Public disclosure portal

### Key Compliance Steps
1. Establish GHG inventory boundaries
2. Identify all emission sources
3. Implement data collection systems
4. Calculate baseline emissions
5. Engage third-party verifier
6. Submit report to CARB
7. Public disclosure on website

## Related Regulations
- **SB 261**: Climate risk disclosure
- **SEC Climate Rules**: Federal disclosure requirements
- **EU CSRD**: For companies with EU operations
- **ISSB Standards**: International sustainability standards