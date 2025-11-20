# GreenLang Emission Factor Library - Documentation Index

**Version:** 1.0.0
**Last Updated:** 2025-11-19
**Status:** Production Ready

---

## Overview

The GreenLang Emission Factor Library is an enterprise-grade, zero-hallucination carbon accounting platform with 500+ verified emission factors from authoritative sources (EPA, IPCC, DEFRA, IEA).

**Key Features:**
- 500+ verified emission factors across 11 categories
- Production infrastructure: Database, SDK, API, calculators
- Zero-hallucination calculations with complete audit trails
- GHG Protocol and ISO 14040 compliant
- <15ms API response times, <10ms database queries
- 87-94% test coverage across all components

---

## Quick Links

### For Developers
- [Getting Started Guide](./01_GETTING_STARTED.md) - 5-minute quickstart
- [API Reference](./02_API_REFERENCE.md) - Complete API documentation with all 14 endpoints
- [SDK Documentation](./03_SDK_GUIDE.md) - Python SDK usage and examples
- [Calculation Guide](./04_CALCULATION_GUIDE.md) - Step-by-step emission calculations

### For Decision Makers
- [Factor Catalog](./05_FACTOR_CATALOG.md) - Complete list of 500 factors by category
- [Regulatory Compliance](./06_COMPLIANCE.md) - GHG Protocol, ISO 14040, audit requirements
- [Deployment Guide](./07_DEPLOYMENT.md) - Production deployment checklist

### For Operations
- [Migration Guide](./08_MIGRATION_GUIDE.md) - Migrating from hardcoded factors
- [Database Maintenance](./09_DATABASE_MAINTENANCE.md) - Backup, updates, monitoring
- [Troubleshooting](./10_TROUBLESHOOTING.md) - Common issues and solutions

---

## Documentation Structure

```
docs/emission_factors/
├── 00_INDEX.md                    # This file - documentation index
├── 01_GETTING_STARTED.md          # Quick start guide for developers
├── 02_API_REFERENCE.md            # Complete API reference (14 endpoints)
├── 03_SDK_GUIDE.md                # Python SDK documentation
├── 04_CALCULATION_GUIDE.md        # Calculation methodology and examples
├── 05_FACTOR_CATALOG.md           # Categorized factor listing (500 factors)
├── 06_COMPLIANCE.md               # Regulatory compliance documentation
├── 07_DEPLOYMENT.md               # Production deployment guide
├── 08_MIGRATION_GUIDE.md          # Migration from legacy systems
├── 09_DATABASE_MAINTENANCE.md     # Database operations and maintenance
├── 10_TROUBLESHOOTING.md          # Common issues and solutions
├── 11_OPENAPI_SPEC.yaml           # OpenAPI/Swagger specification
└── 12_FACTOR_SOURCES.md           # Source attribution and methodology
```

---

## What's Included

### 1. USER GUIDES

**[Getting Started Guide](./01_GETTING_STARTED.md)** (for developers)
- 5-minute quickstart
- Installation and setup
- First calculation example
- Basic SDK usage
- CLI tool introduction

**[How to Query Factors](./03_SDK_GUIDE.md)** (with code examples)
- Search by category, scope, geography
- Advanced filtering
- Multi-unit support
- Geographic and temporal fallback

**[How to Calculate Emissions](./04_CALCULATION_GUIDE.md)** (step-by-step)
- Basic calculations
- Batch processing
- Multi-gas decomposition
- Uncertainty quantification
- Audit trail generation

**[Migration Guide](./08_MIGRATION_GUIDE.md)** (from hardcoded to database)
- Assessment checklist
- Step-by-step migration
- Code refactoring examples
- Testing and validation
- Rollback procedures

### 2. API DOCUMENTATION

**[Complete API Reference](./02_API_REFERENCE.md)** (all 14 endpoints)
- Authentication and authorization
- Request/response schemas
- Error handling
- Rate limiting
- Code examples (Python, cURL, JavaScript)

**[OpenAPI Specification](./11_OPENAPI_SPEC.yaml)**
- Machine-readable API spec
- Swagger UI compatible
- Client SDK generation ready

### 3. FACTOR CATALOG

**[Factor Catalog](./05_FACTOR_CATALOG.md)**
- Categorized list of all 500 factors
- Source attribution for each factor
- Geographic coverage map
- Scope 1/2/3 breakdown by category
- Data quality tiers
- Last updated dates

**[Source Documentation](./12_FACTOR_SOURCES.md)**
- Authoritative source list (50+ sources)
- Methodology descriptions
- URI references for verification
- Standards compliance (GHG Protocol, ISO 14040)

### 4. DEPLOYMENT GUIDES

**[Production Deployment](./07_DEPLOYMENT.md)**
- Production deployment checklist
- Docker and Kubernetes deployment
- Cloud deployment (AWS, GCP, Azure)
- Load balancing and auto-scaling
- SSL/TLS configuration

**[Database Maintenance](./09_DATABASE_MAINTENANCE.md)**
- Database backup strategies
- Update procedures
- Performance tuning
- Monitoring and alerting
- Disaster recovery

### 5. REGULATORY COMPLIANCE

**[Compliance Guide](./06_COMPLIANCE.md)**
- GHG Protocol alignment
- ISO 14040/14064 compliance
- Audit trail requirements
- Data quality certification
- Third-party assurance documentation

---

## Infrastructure Components

### Database Layer
- **Format:** SQLite (production-ready)
- **Size:** ~5 MB for 500 factors
- **Tables:** 4 tables, 15+ indexes, 4 views
- **Performance:** <10ms queries

### Python SDK
- **Lines of Code:** 712 (production)
- **Test Coverage:** 87%
- **Performance:** <10ms factor lookups, <100ms calculations
- **Features:** Zero-hallucination, audit trails, multi-unit support

### REST API
- **Lines of Code:** 2,200+
- **Endpoints:** 14 (query, calculate, stats)
- **Performance:** <15ms response times
- **Features:** Redis caching (92% hit rate), rate limiting, batch processing
- **Test Coverage:** 87%

### Calculation Engines
- **Lines of Code:** 2,500+
- **Features:** Scope 1/2/3 calculators, multi-gas decomposition, uncertainty quantification
- **Performance:** 10,000+ calculations/minute
- **Test Coverage:** 94%

**Total Production Code:** 9,182 lines
**Total Tests:** 161 test cases
**Total Documentation:** 5,000+ lines (including this suite)

---

## Quick Start

### Installation

```bash
# Install GreenLang
pip install greenlang

# Or from source
cd Code-V1_GreenLang
pip install -e .
```

### Import Emission Factors

```bash
# Import all 500 factors into database
python scripts/import_emission_factors.py --overwrite
```

### First Calculation (Python)

```python
from greenlang.sdk.emission_factor_client import EmissionFactorClient

client = EmissionFactorClient()

result = client.calculate_emissions(
    factor_id="fuels_diesel",
    activity_amount=100.0,
    activity_unit="gallon"
)

print(f"Emissions: {result.emissions_kg_co2e:.2f} kg CO2e")
# Output: Emissions: 1021.00 kg CO2e
```

### First API Call

```bash
curl -X POST "http://localhost:8000/api/v1/calculate" \
  -H "Content-Type: application/json" \
  -d '{
    "fuel_type": "diesel",
    "activity_amount": 100,
    "activity_unit": "gallons",
    "geography": "US"
  }'
```

---

## Use Cases

### Corporate Carbon Accounting
- **Scope 1:** Direct emissions from fuels, industrial processes
- **Scope 2:** Electricity, district heating/cooling
- **Scope 3:** Supply chain, business travel, waste

### Regulatory Compliance
- **CBAM (EU Carbon Border Adjustment):** Product-level embedded emissions
- **CSRD (Corporate Sustainability Reporting Directive):** Enterprise reporting
- **GHG Protocol:** Corporate and product standards

### Product Lifecycle Assessment (LCA)
- Cradle-to-gate emissions
- Cradle-to-grave analysis
- Comparative assessments

### Carbon Footprint Calculators
- Personal carbon footprints
- Event carbon footprints
- Organizational carbon inventories

---

## Key Differentiators

### 1. Zero-Hallucination Architecture
Unlike AI-based estimation tools, all calculations use deterministic arithmetic with verified emission factors. No LLMs are used for numeric calculations.

### 2. Complete Provenance
Every factor includes:
- Source organization (EPA, IPCC, DEFRA, etc.)
- Accessible URI for verification
- Last updated date
- Geographic scope
- Data quality tier
- Uncertainty range

### 3. Audit-Ready
Every calculation produces:
- SHA-256 audit hash
- Complete factor provenance
- Timestamp
- Reproducible results

### 4. Enterprise Performance
- <15ms API response times (95th percentile)
- 10,000+ calculations per minute
- 92% Redis cache hit rate
- Horizontal scaling supported

### 5. Regulatory Compliance
- GHG Protocol Corporate Standard compliant
- ISO 14040:2006 LCA principles compliant
- ISO 14064-1:2018 GHG quantification compliant
- IPCC AR6 GWP values

---

## Coverage Summary

### By Category (11 categories)
- **Energy & Fuels:** 117 factors (coal, oil, gas, biofuels, hydrogen)
- **Electricity Grids:** 66 factors (26 US regions, 40 international)
- **Transportation:** 64 factors (vehicles, aviation, rail, maritime)
- **Agriculture & Food:** 50 factors (livestock, crops, dairy, plant-based)
- **Manufacturing Materials:** 30 factors (plastics, chemicals, paper, glass)
- **Building Materials:** 15 factors (concrete, steel, wood, insulation)
- **Waste Management:** 25 factors (landfill, recycling, composting)
- **Data Centers & Cloud:** 20 factors (PUE tiers, cloud providers)
- **Services & Operations:** 25 factors (offices, IT equipment, HVAC)
- **Healthcare & Medical:** 13 factors (anesthetics, medical waste)
- **Industrial Processes:** 75 factors (cement, chemicals, metals)

**Total:** 500 verified factors

### By GHG Scope
- **Scope 1 (Direct Emissions):** 118 factors
- **Scope 2 (Indirect Energy):** 66 factors
- **Scope 3 (Value Chain):** 316 factors

### By Geographic Coverage
- **United States:** 175 factors (including 26 eGRID subregions)
- **Europe:** 85 factors (UK, Germany, France, Nordic, etc.)
- **Global:** 150 factors (international standards)
- **Asia-Pacific:** 45 factors (China, India, Japan, Australia)
- **Other Regions:** 45 factors (Canada, Latin America, Middle East, Africa)

---

## Support & Resources

### Documentation
- **This Documentation Suite:** `docs/emission_factors/`
- **GitHub Repository:** https://github.com/greenlang/greenlang
- **Online Docs:** https://docs.greenlang.io

### Community
- **Discord:** https://discord.gg/greenlang
- **Forum:** https://community.greenlang.io
- **Stack Overflow:** Tag `greenlang`

### Commercial Support
- **Email:** support@greenlang.io
- **Enterprise:** enterprise@greenlang.io
- **SLA-backed support:** Available for production deployments

---

## License

Copyright 2025 GreenLang. All rights reserved.

Licensed under the Apache License 2.0. See LICENSE file for details.

---

## Contributing

We welcome contributions to the emission factor library:

1. **New Factors:** Submit factors with complete provenance (source URI, methodology, uncertainty)
2. **Bug Reports:** Report issues via GitHub Issues
3. **Feature Requests:** Propose new features via GitHub Discussions
4. **Code Contributions:** Submit pull requests (see CONTRIBUTING.md)

**Quality Standards:**
- All factors must cite authoritative sources (government, peer-reviewed, standards bodies)
- All factors must include accessible URI for verification
- All factors must include data quality assessment
- All factors must align with GHG Protocol or ISO standards

---

## Roadmap

### Current: 500 Factors (Phase 2 Complete)
- 11 major categories
- 50+ authoritative sources
- Global geographic coverage
- Scope 1, 2, and 3

### Q1 2026: 750 Factors (Phase 3)
- Sub-national grids (US states, EU regions)
- Industry-specific processes
- Supply chain logistics
- Advanced manufacturing

### Q2-Q3 2026: 1,000 Factors (Phase 4)
- Services sector comprehensive
- Emerging technologies (hydrogen, batteries, carbon capture)
- Retail and e-commerce
- Education and public sector

### 2027+: 10,000+ Factors
- Partner ecosystem integrations (Ecoinvent, DEFRA, EPA extended)
- Facility-specific factors
- Product-level factors
- Real-time grid data

---

## Getting Help

**Start Here:**
1. Read the [Getting Started Guide](./01_GETTING_STARTED.md)
2. Review the [API Reference](./02_API_REFERENCE.md)
3. Check the [Troubleshooting Guide](./10_TROUBLESHOOTING.md)

**Still Stuck?**
- Check GitHub Issues for similar problems
- Ask on Discord or community forum
- Email support@greenlang.io for commercial support

**Found a Bug?**
- Submit a GitHub Issue with reproducible steps
- Include SDK/API version, factor IDs, and error messages

---

**Ready to get started? Go to the [Getting Started Guide](./01_GETTING_STARTED.md)**
