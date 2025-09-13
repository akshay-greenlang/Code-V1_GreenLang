# Enterprise Carbon Suite - Advanced Pack Example

This advanced pack demonstrates enterprise-grade carbon accounting capabilities with comprehensive features including multi-scope emissions tracking, renewable energy analysis, compliance validation, and detailed reporting.

## Pack Overview

**Name:** `enterprise-carbon-suite`
**Version:** 2.1.0
**License:** Apache-2.0
**Compatibility:** GreenLang >=0.3,<0.5, Python >=3.10

## Key Features Demonstrated

### 1. **Advanced Pack Configuration**
- **Complex Dependencies**: Multiple Python packages with version constraints
- **Policy Enforcement**: Network restrictions, data residency requirements, compliance frameworks
- **Security Controls**: SBOM, digital signatures, encryption requirements, access control
- **Rich Metadata**: Industry focus, certification status, support channels

### 2. **Multi-Pipeline Architecture**
- **Main Analysis Pipeline** (`main-analysis.yaml`): Comprehensive carbon accounting
- **Backup Pipeline** (`backup-pipeline.yaml`): Simplified fallback calculations
- **Validation Pipeline** (`validation-pipeline.yaml`): Data quality assurance

### 3. **Enterprise Security Features**
```yaml
security:
  sbom: security/sbom.spdx.json
  signatures: [security/pack.sig, security/release.sig]
  encryption:
    data_at_rest: true
    transit_required: true
  access_control:
    min_clearance_level: "restricted"
    required_certifications: ["ISO-27001", "SOC2-Type2"]
```

### 4. **Policy-Driven Operations**
```yaml
policy:
  network: ["era5:*", "ghg-protocol.org:*", "iea.org:api/*"]
  data_residency: ["IN", "EU", "US"]
  ef_vintage_min: 2023
  compliance_frameworks: ["GHG-Protocol", "ISO-14064", "CDP"]
  audit_retention_years: 7
```

## Pipeline Details

### Main Analysis Pipeline

A comprehensive multi-step pipeline featuring:

#### **Input Schema**
- Structured input validation with JSON Schema-like definitions
- Support for complex nested objects (facility data, energy consumption)
- Type validation, ranges, and default values

#### **Advanced Step Features**
1. **Conditional Execution**
   ```yaml
   conditional:
     condition: "${inputs.energy_consumption.renewable_kwh} > 0"
     on_skip:
       renewable_offset_tons: 0
   ```

2. **Parallel Processing**
   ```yaml
   parallel: true  # Steps run simultaneously
   ```

3. **Error Handling**
   ```yaml
   error_handling:
     on_failure: stop
     retry_count: 3
     retry_delay: 5
   ```

4. **Dependency Management**
   ```yaml
   depends_on: [step1, step2, step3]
   ```

5. **Output Caching**
   ```yaml
   cache_duration: "24h"
   ```

#### **Multi-Scope Emissions Calculation**
- **Scope 1**: Direct emissions from natural gas combustion
- **Scope 2**: Indirect emissions from electricity consumption
  - Market-based method (with renewable certificates)
  - Location-based method (grid average)
- **Renewable Energy Integration**: Solar offset calculations

#### **Advanced Analytics**
- Intensity metric calculations (per sqft, per employee, per revenue)
- Industry benchmarking and percentile analysis
- Compliance validation against multiple standards

### Backup Pipeline

A simplified fallback pipeline that demonstrates:
- Conservative emission factor defaults
- Minimal dependency requirements
- Basic validation and reporting
- Fast execution (5-minute timeout)

### Validation Pipeline

Comprehensive data quality assurance featuring:
- **Data Completeness Assessment**
- **Type and Range Validation**
- **Consistency Rule Checking**
- **Statistical Outlier Detection**
- **Historical Comparison** (optional)
- **Quality Score Calculation**
- **Quality Gate Enforcement**

## Usage Examples

### 1. Basic Execution
```bash
# Run main analysis
gl pipeline run examples/packs/advanced/main-analysis.yaml \
  --input facility_data.name="Headquarters" \
  --input facility_data.location="IN-North" \
  --input facility_data.building_area_sqft=50000 \
  --input energy_consumption.electricity_kwh=500000 \
  --input energy_consumption.natural_gas_mmbtu=1000

# Validate data first
gl pipeline run examples/packs/advanced/validation-pipeline.yaml \
  --input raw_data=@facility_data.json
```

### 2. Pack Installation
```bash
# Install the pack
gl pack install examples/packs/advanced/

# List pack contents
gl pack inspect enterprise-carbon-suite

# Validate pack configuration
gl pack validate examples/packs/advanced/pack.yaml --strict
```

### 3. Configuration Examples

**Sample Input File** (`facility_data.json`):
```json
{
  "facility_data": {
    "name": "Manufacturing Plant A",
    "location": "IN-North",
    "building_area_sqft": 150000,
    "employee_count": 250,
    "industry_type": "manufacturing",
    "annual_revenue_usd": 50000000
  },
  "energy_consumption": {
    "electricity_kwh": 2500000,
    "natural_gas_mmbtu": 15000,
    "renewable_kwh": 500000
  }
}
```

## Output Examples

### Comprehensive Results
```yaml
# Primary emissions data
total_emissions_scope1: 849.9  # tons CO2e
total_emissions_scope2_market: 1850.0  # tons CO2e
net_emissions_total: 2289.9  # tons CO2e
renewable_offset: 410.1  # tons CO2e

# Intensity metrics
emissions_per_sqft: 0.0153  # tons CO2e per sqft
emissions_per_employee: 9.16  # tons CO2e per employee

# Compliance status
compliance_status: "COMPLIANT"
compliance_gaps: []

# Generated reports
executive_report_path: "out/analysis/executive_summary_2024.html"
audit_report_path: "out/analysis/detailed_audit_2024.pdf"
```

## Best Practices Demonstrated

### 1. **Input Validation**
- Comprehensive schema definitions
- Type safety and range validation
- Required vs. optional fields
- Default value handling

### 2. **Error Resilience**
- Multiple error handling strategies
- Retry mechanisms with backoff
- Graceful degradation (backup pipeline)
- Conditional execution for optional steps

### 3. **Security & Compliance**
- Policy-driven network access
- Data encryption requirements
- Audit trail generation
- Access control integration

### 4. **Modularity & Reusability**
- Separate pipelines for different use cases
- Parameterized configurations
- Agent-based architecture
- Clear separation of concerns

### 5. **Enterprise Operations**
- Comprehensive logging and auditing
- Quality gates and validation
- Industry benchmarking
- Multi-format reporting

## File Structure

```
examples/packs/advanced/
├── pack.yaml                    # Main pack manifest
├── main-analysis.yaml          # Primary carbon analysis pipeline
├── backup-pipeline.yaml        # Fallback calculation pipeline
├── validation-pipeline.yaml    # Data quality assurance pipeline
├── README.md                   # This documentation
├── CARD.md                     # Model/pack card (referenced)
├── datasets/                   # Sample emission factors data
│   ├── emission_factors_2024.csv
│   ├── renewable_certificates.csv
│   ├── grid_intensity_regions.csv
│   └── compliance_standards.json
├── reports/                    # Report templates
│   ├── executive_summary.html.j2
│   ├── detailed_audit.pdf.j2
│   └── compliance_report.xlsx.j2
└── security/                   # Security artifacts
    ├── sbom.spdx.json         # Software bill of materials
    ├── pack.sig               # Pack signature
    └── release.sig            # Release signature
```

## Learning Objectives

This example teaches:

1. **Advanced YAML Configuration**: Complex pack.yaml with all optional fields
2. **Multi-Pipeline Design**: Different pipelines for different purposes
3. **Error Handling Patterns**: Various strategies for dealing with failures
4. **Security Best Practices**: Enterprise-grade security controls
5. **Compliance Integration**: Multi-standard compliance validation
6. **Data Quality Management**: Comprehensive validation workflows
7. **Scalable Architecture**: Patterns for large-scale deployments

## Related Examples

- [Minimal Pack](../minimal/) - Basic pack structure
- [Complex Pipeline](../../pipelines/complex/) - Advanced pipeline features
- [SDK Examples](../../sdk/) - Programmatic pack usage

## Support

For questions about this example:
- Review the [Pack Specification](../../../docs/specs/PACK_SCHEMA_V1.md)
- Check the [Pipeline Specification](../../../docs/specs/GL_PIPELINE_SPEC_V1.md)
- Consult the [Enterprise Documentation](https://docs.greenlang.io/enterprise/)