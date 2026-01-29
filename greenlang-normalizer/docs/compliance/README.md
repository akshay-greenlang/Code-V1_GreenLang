# Compliance Documentation

This directory contains compliance documentation for regulatory requirements and audit support.

## Contents

- `ghg-protocol.md` - GHG Protocol compliance
- `iso-14064.md` - ISO 14064 compliance
- `csrd-esrs.md` - EU CSRD/ESRS compliance
- `sec-climate.md` - SEC Climate Disclosure compliance
- `audit-guide.md` - Audit preparation guide

## Supported Frameworks

### GHG Protocol

The normalizer supports GHG Protocol Corporate Standard requirements:

- **Scope 1**: Direct emissions from owned sources
- **Scope 2**: Indirect emissions from purchased energy
- **Scope 3**: Other indirect emissions in value chain

Key features:
- GWP values (AR4, AR5, AR6)
- Standard emission factors
- Activity data validation
- Organizational boundary tracking

### ISO 14064

ISO 14064-1:2018 compliance features:

- Mass balance calculations
- Uncertainty assessment
- Data quality indicators
- Verification support

### EU CSRD/ESRS

European Sustainability Reporting Standards compliance:

- ESRS E1: Climate change
- Double materiality assessment
- Forward-looking targets
- Transition plans

### SEC Climate Disclosure

US SEC climate disclosure rule support:

- GHG emissions (Scope 1, 2, 3)
- Climate-related risks
- Financial impact metrics
- Safe harbor provisions

## Audit Trail

All normalization operations create audit records with:

- Input data hash
- Output data hash
- Conversion steps with factors
- Policy applied
- Timestamp and user context

### Retention

Audit records are retained for:
- 7 years (default)
- 10 years (financial reporting)
- Indefinite (configurable)

### Export Formats

- JSON (API)
- Parquet (Data warehouse)
- CSV (Spreadsheet)
- XBRL (Regulatory filing)
