# EU Green Claims Directive - Regulatory Requirements
**Directive (EU) 2024/825**
**Compliance Deadline: September 27, 2026**

## Executive Summary
The Green Claims Directive establishes rules for substantiating and communicating environmental claims about products and traders to protect consumers from greenwashing and ensure fair competition.

## Key Requirements and Compliance Criteria

### 1. Covered Claims
- **Explicit environmental claims**: Direct statements about environmental aspects
- **Environmental labels**: Sustainability labels and certification marks
- **Comparative claims**: Environmental superiority over competitors
- **Future environmental performance**: Net-zero commitments, carbon neutral claims

### 2. Substantiation Requirements
- **Scientific evidence**: Based on recognized methods
- **Life cycle assessment**: Where relevant
- **Significant impacts**: Focus on actual environmental impacts
- **Trade-offs**: Disclose if improvement in one area causes degradation in another
- **Separate credits**: Distinguish between own performance and offsets

### 3. Communication Requirements
- **Clear and specific**: No vague terms like "eco-friendly" without specifics
- **Accessible information**: QR codes or web links to full substantiation
- **Language requirements**: In official EU languages where marketed
- **Updates**: Claims must be updated when circumstances change

## Reporting Templates and Formats

### Green Claim Substantiation Dossier
```json
{
  "claim_id": "GC-2026-XXXXX",
  "product_information": {
    "name": "Product Name",
    "ean_code": "1234567890123",
    "category": "Electronics",
    "market_countries": ["DE", "FR", "IT"]
  },
  "environmental_claim": {
    "type": "carbon_neutral",
    "specific_claim": "Carbon neutral by 2025",
    "scope": "entire_product_lifecycle",
    "aspects_covered": ["climate", "resource_use"]
  },
  "substantiation": {
    "methodology": {
      "standard": "ISO 14040/14044",
      "system_boundaries": "cradle_to_grave",
      "functional_unit": "1 product unit"
    },
    "environmental_impacts": {
      "climate_change": {
        "value": 10.5,
        "unit": "kg CO2-eq",
        "reduction_vs_baseline": "30%"
      },
      "resource_depletion": {
        "value": 0.05,
        "unit": "kg Sb-eq",
        "reduction_vs_baseline": "15%"
      }
    },
    "data_quality": {
      "primary_data_coverage": "75%",
      "data_age": "< 3 years",
      "geographic_representativeness": "European"
    },
    "offsets_used": {
      "amount": 3.5,
      "type": "verified_carbon_credits",
      "standard": "Gold Standard",
      "additionality_verified": true
    }
  },
  "verification": {
    "verifier": "Accredited Body Name",
    "verification_date": "2026-01-15",
    "certificate_number": "VER-2026-12345",
    "validity_period": "3 years"
  }
}
```

### Comparative Claim Format
```json
{
  "comparison_basis": {
    "reference_product": "Market average",
    "comparison_year": "2025",
    "geographic_scope": "EU-27"
  },
  "environmental_advantages": [
    {
      "impact_category": "carbon_footprint",
      "improvement": "40% lower",
      "evidence": "LCA study reference"
    }
  ],
  "trade_offs": [
    {
      "impact_category": "water_use",
      "degradation": "10% higher",
      "justification": "Overall environmental benefit"
    }
  ]
}
```

## Data Requirements and Sources

### Required Data Points

#### Product Level Data
- Complete bill of materials
- Manufacturing process data
- Energy consumption per unit
- Transportation distances and modes
- Use phase scenarios
- End-of-life treatment methods

#### Environmental Impact Data
- Carbon footprint (Scopes 1, 2, 3)
- Water consumption and pollution
- Resource depletion
- Biodiversity impacts
- Circularity indicators
- Chemical emissions

#### Supply Chain Data
- Supplier environmental data
- Raw material origins
- Processing locations
- Certification statuses

### Accepted Data Sources
- Primary data from operations (preferred)
- LCA databases (Ecoinvent, GaBi)
- Environmental Product Declarations (EPDs)
- Industry average data (with justification)
- Scientific literature
- Government statistics

## Calculation Methodologies

### Life Cycle Assessment Requirements
```
Environmental Impact = Σ(Activity Data × Characterization Factor)

Stages to include:
1. Raw material extraction
2. Manufacturing
3. Distribution
4. Use phase
5. End-of-life

Impact Categories (minimum):
- Climate change (GWP 100)
- Ozone depletion
- Acidification
- Eutrophication
- Resource depletion
```

### Carbon Neutrality Calculations
```
Net Emissions = Gross Emissions - Reductions - Removals - Offsets

Where:
- Gross Emissions: Total lifecycle emissions
- Reductions: Verified emission reductions
- Removals: Carbon sequestration (if applicable)
- Offsets: Must be additional, permanent, verified
```

### Circularity Metrics
```
Material Circularity = (Recycled Input + Reusable Output) / Total Material Flow

Durability Score = Actual Lifetime / Standard Lifetime × 100
```

## Penalties for Non-Compliance

### Financial Penalties
- **Minimum**: 4% of annual turnover in affected Member State
- **Repeated violations**: Up to 6% of annual turnover
- **Confiscation**: Of revenues gained from violation

### Administrative Measures
- Prohibition of the claim
- Product withdrawal from market
- Public warnings and naming
- Exclusion from public procurement
- Temporary business prohibition (up to 12 months)

### Enforcement Mechanisms
- National competent authorities
- Cross-border cooperation
- Consumer organization complaints
- Competitor complaints
- Mystery shopping and market surveillance

## Scope - Which Companies Must Comply

### Covered Entities
- All traders making environmental claims in EU
- Manufacturers
- Importers
- Distributors
- Online marketplaces (for own claims)

### Exemptions
- **Micro-enterprises**: <10 employees and ≤€2 million turnover
  - Unless making comparative claims
  - Unless using environmental labels
- **EU Ecolabel**: Products with EU Ecolabel exempt
- **Existing EU laws**: Claims covered by specific legislation

### Geographic Scope
- Claims made in EU market
- Claims on products sold in EU
- Online claims targeting EU consumers
- Cross-border sales into EU

## Implementation Requirements

### Verification System
1. **Pre-market verification required for**:
   - New environmental labels
   - Claims by large companies (>250 employees)
   - Comparative claims

2. **Verifier requirements**:
   - Accredited third-party
   - Independence from trader
   - Technical competence
   - Regular audits

### Technical Infrastructure Needed

#### Data Management System
- LCA software integration
- Supply chain data collection
- Document management
- Version control for claims

#### Public Disclosure System
- QR code generation
- Web portal for substantiation
- Multi-language support
- Update notifications

#### Monitoring System
- Claim performance tracking
- Market surveillance alerts
- Compliance dashboard
- Audit trail

## Specific Claim Requirements

### Carbon Neutral/Net Zero Claims
- Complete GHG inventory required
- Reduction targets aligned with 1.5°C
- Offset quality criteria
- Clear timeline and milestones
- Annual progress reporting

### Recyclability Claims
- Design for recycling assessment
- Available recycling infrastructure
- Actual recycling rates
- Material composition disclosure
- Contamination considerations

### Bio-based Claims
- Bio-based content percentage
- Sustainability of biomass source
- Land use change impacts
- Food competition assessment
- Biodegradability conditions

## Timeline and Transition
- **Sep 27, 2026**: Directive enters into force
- **Mar 27, 2027**: Verification bodies operational
- **Sep 27, 2027**: Full enforcement begins
- **2028**: First compliance reviews