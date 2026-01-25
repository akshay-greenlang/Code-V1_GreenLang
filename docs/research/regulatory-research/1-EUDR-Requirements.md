# EU Deforestation Regulation (EUDR) - Regulatory Requirements
**Regulation (EU) 2023/1115**
**Compliance Deadline: December 30, 2025**

## Executive Summary
The EU Deforestation Regulation prohibits placing on the EU market or exporting from the EU certain commodities and products associated with deforestation and forest degradation.

## Key Requirements and Compliance Criteria

### 1. Covered Commodities
- Cattle
- Cocoa
- Coffee
- Oil palm
- Rubber
- Soya
- Wood
- Derived products (leather, chocolate, furniture, paper, etc.)

### 2. Core Compliance Requirements
- **Deforestation-free**: Products must not be produced on land deforested after December 31, 2020
- **Legal compliance**: Production must comply with relevant laws of the country of production
- **Due diligence statement**: Submit before placing products on EU market

### 3. Due Diligence System Requirements
- **Information Collection**
  - Product description and quantity
  - Country of production
  - Geolocation coordinates of all plots of land (>4 hectares)
  - Date/time range of production
  - Supplier and customer information
  - Documentation proving deforestation-free and legal compliance

- **Risk Assessment**
  - Country/area deforestation prevalence
  - Presence of indigenous peoples
  - Corruption levels
  - Data reliability concerns
  - Complexity of supply chain

- **Risk Mitigation**
  - Independent audits
  - Additional documentation
  - Supplier capacity building

## Reporting Templates and Formats

### Due Diligence Statement Format
```json
{
  "reference_number": "EUDR-2025-XXXXX",
  "operator_details": {
    "name": "Company Name",
    "eori_number": "XX123456789",
    "address": "Full Address"
  },
  "product_information": {
    "commodity": "coffee",
    "cn_code": "0901",
    "quantity": 1000,
    "unit": "kg"
  },
  "geolocation_data": {
    "plots": [
      {
        "latitude": -15.7801,
        "longitude": -47.9292,
        "polygon": "POLYGON((...))",
        "area_hectares": 50
      }
    ]
  },
  "compliance_declaration": {
    "deforestation_free": true,
    "legally_produced": true,
    "risk_level": "low"
  }
}
```

## Data Requirements and Sources

### Required Data Points
1. **Product Data**
   - HS/CN codes
   - Quantity and weight
   - Production dates

2. **Geographic Data**
   - GPS coordinates (6 decimal places minimum)
   - Polygon boundaries for plots >4 hectares
   - Land use history since 2020

3. **Supply Chain Data**
   - Complete supplier list
   - Chain of custody documentation
   - Transportation records

### Data Sources
- Satellite monitoring systems (Copernicus, Planet Labs)
- National forest monitoring systems
- Certification schemes (FSC, PEFC, Rainforest Alliance)
- Supply chain management systems
- Third-party verification reports

## Calculation Methodologies

### Risk Assessment Scoring
```
Risk Score = (Country Risk × 0.4) + (Supply Chain Risk × 0.3) +
             (Product Risk × 0.2) + (Supplier Risk × 0.1)

Where:
- Country Risk: Based on FAO deforestation data
- Supply Chain Risk: Complexity and traceability
- Product Risk: Commodity-specific factors
- Supplier Risk: Historical compliance
```

### Geolocation Verification
- Minimum 6 decimal places for coordinates
- Polygon mapping for areas >4 hectares
- Time-stamped satellite imagery comparison
- Forest cover change detection algorithms

## Penalties for Non-Compliance

### Financial Penalties
- **Maximum fines**: At least 4% of annual EU turnover
- **Confiscation**: Of products and revenues
- **Temporary exclusion**: From public procurement and funding

### Administrative Measures
- Product recall and withdrawal
- Prohibition from placing products on market
- Temporary prohibition (up to 12 months)
- Public naming and shaming

### Inspection Frequency
- **High-risk countries**: 9% of operators checked
- **Standard-risk countries**: 3% of operators checked
- **Low-risk countries**: 1% of operators checked

## Scope - Which Companies Must Comply

### Operators (Primary Responsibility)
- Companies first placing products on EU market
- EU-based manufacturers using covered commodities
- Exporters from EU

### Traders
- Companies making products available on EU market
- Must ensure operators have complied
- Simplified due diligence for SME traders

### Size Categories
- **Large companies**: Full compliance by Dec 30, 2024
- **SMEs**: Extended deadline to Jun 30, 2025
- **Micro enterprises**: Simplified requirements

### Geographic Scope
- All EU member states
- EEA countries (Norway, Iceland, Liechtenstein)
- UK has similar UK Forest Risk Commodities regulations

## Implementation Timeline
- **Dec 30, 2024**: Large operators and traders must comply
- **Jun 30, 2025**: SMEs must comply
- **Dec 30, 2025**: Full enforcement begins
- **2026**: First compliance reports due

## Technical Integration Requirements
- API integration with EU Information System
- Geospatial data management capabilities
- Document management system
- Risk assessment engine
- Supplier portal for data collection