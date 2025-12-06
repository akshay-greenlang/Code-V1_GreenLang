# CBAM Reporting Guide

## Overview

The EU Carbon Border Adjustment Mechanism (CBAM) is a policy tool designed to prevent carbon leakage by putting a carbon price on imports of certain goods. This guide explains how to use GreenLang to calculate embedded emissions and generate CBAM quarterly reports.

**Key Dates:**

| Date | Milestone |
|------|-----------|
| October 1, 2023 | Transitional period begins |
| January 31, 2024 | First quarterly report due (Q4 2023) |
| December 31, 2025 | Transitional period ends |
| January 1, 2026 | Definitive period begins (certificates required) |

---

## CBAM Requirements Summary

### Covered Products

CBAM applies to imports in these sectors:

| Sector | CN Codes (Examples) | Products |
|--------|---------------------|----------|
| **Cement** | 2523 | Cement clinker, Portland cement |
| **Iron and Steel** | 7206-7229, 7301-7311 | Pig iron, steel products, tubes |
| **Aluminum** | 7601-7609 | Unwrought aluminum, aluminum products |
| **Fertilizers** | 2808, 2814, 3102-3105 | Ammonia, nitric acid, fertilizers |
| **Electricity** | 2716 | Electrical energy |
| **Hydrogen** | 2804 | Hydrogen |

### Reporting Obligations

**During Transitional Period (2023-2025):**
- Submit quarterly CBAM reports
- Report embedded emissions (no payment required)
- Use actual or default emission values

**During Definitive Period (from 2026):**
- Purchase CBAM certificates
- Surrender certificates based on embedded emissions
- Deduct foreign carbon prices paid

---

## Quarterly Reporting Requirements

### Report Contents

Each quarterly report must include:

1. **Importer Information** - EORI number, name, address
2. **Product Details** - CN code, quantity, country of origin
3. **Installation Information** - Producer details (if actual data)
4. **Embedded Emissions** - Direct and indirect emissions
5. **Emission Source** - Actual values or default values used

### Reporting Deadlines

| Quarter | Period | Report Due |
|---------|--------|------------|
| Q1 | January - March | April 30 |
| Q2 | April - June | July 31 |
| Q3 | July - September | October 31 |
| Q4 | October - December | January 31 (next year) |

---

## Embedded Emissions Calculation

### Understanding Embedded Emissions

Embedded emissions are the greenhouse gases released during production:

- **Direct Emissions**: Emissions from the production process itself
- **Indirect Emissions**: Emissions from electricity used in production

**Total Embedded Emissions = Direct + Indirect**

### Emission Data Sources

| Source | Description | Accuracy | Use Case |
|--------|-------------|----------|----------|
| **Actual (Installation)** | From specific production facility | Highest | Preferred when available |
| **Actual (Operator Average)** | Average across operator facilities | High | Multiple facilities |
| **Default (Country)** | EU-provided country defaults | Medium | When actual unavailable |
| **Default (EU Average)** | EU average values | Lower | Fallback option |

---

## Step-by-Step CBAM Workflow

### Step 1: Set Up Your Environment

**Python SDK:**

```python
from greenlang import Client

client = Client(
    client_id="your_client_id",
    client_secret="your_client_secret"
)
```

### Step 2: Prepare Import Data

```python
# Example: Steel imports for Q1 2025
imports = [
    {
        "product_id": "STEEL-001",
        "cn_code": "7208510091",
        "product_category": "iron_steel",
        "description": "Hot-rolled steel coils, width >= 600mm, thickness < 3mm",
        "quantity": 500,
        "unit": "tonnes",
        "country_of_origin": "TR",
        "installation_id": "TR-INST-2024-12345",
        "production_route": "basic_oxygen_furnace",
        "emission_data": {
            "type": "actual",
            "direct_emissions": 1.85,
            "indirect_emissions": 0.42,
            "unit": "tCO2e/tonne",
            "verification_status": "verified",
            "verifier": "TUV Rheinland"
        }
    },
    {
        "product_id": "STEEL-002",
        "cn_code": "7209160000",
        "product_category": "iron_steel",
        "description": "Cold-rolled steel coils, width >= 600mm, thickness 1-3mm",
        "quantity": 300,
        "unit": "tonnes",
        "country_of_origin": "CN",
        "emission_data": {
            "type": "default",
            "use_country_default": True
        }
    },
    {
        "product_id": "ALUM-001",
        "cn_code": "7601100000",
        "product_category": "aluminum",
        "description": "Unwrought aluminum, not alloyed",
        "quantity": 200,
        "unit": "tonnes",
        "country_of_origin": "RU",
        "production_route": "primary_smelting",
        "emission_data": {
            "type": "actual",
            "direct_emissions": 1.65,
            "indirect_emissions": 8.50,
            "unit": "tCO2e/tonne",
            "electricity_source": "grid_mix"
        }
    }
]
```

### Step 3: Calculate Embedded Emissions

```python
result = client.calculate.cbam(
    reporting_period={
        "year": 2025,
        "quarter": 1
    },
    imports=imports,
    options={
        "include_certificate_estimate": True,
        "carbon_price_eur": 85.50,
        "generate_xml": True,
        "include_breakdown": True
    }
)

print(f"Calculation ID: {result.calculation_id}")
print(f"\nReporting Period: Q{result.reporting_period.quarter} {result.reporting_period.year}")
print(f"Period: {result.reporting_period.start_date} to {result.reporting_period.end_date}")
```

### Step 4: Review Results

```python
# Summary
print("\n=== CBAM Report Summary ===")
print(f"Total Imports: {result.summary.total_imports}")
print(f"Total Quantity: {result.summary.total_quantity} {result.summary.quantity_unit}")
print(f"\nTotal Embedded Emissions:")
print(f"  Direct: {result.summary.total_embedded_emissions.direct:,.2f} tCO2e")
print(f"  Indirect: {result.summary.total_embedded_emissions.indirect:,.2f} tCO2e")
print(f"  Total: {result.summary.total_embedded_emissions.total:,.2f} tCO2e")

# Per-product breakdown
print("\n=== Product Details ===")
for product in result.products:
    print(f"\n{product.product_id}: {product.description[:50]}...")
    print(f"  CN Code: {product.cn_code}")
    print(f"  Quantity: {product.quantity} {product.unit}")
    print(f"  Origin: {product.country_of_origin}")
    print(f"  Emission Source: {product.emission_source}")
    print(f"  Specific Emissions: {product.embedded_emissions.specific_emissions.total:.2f} tCO2e/tonne")
    print(f"  Total Emissions: {product.embedded_emissions.total:.2f} tCO2e")
```

**Example Output:**

```
=== CBAM Report Summary ===
Total Imports: 3
Total Quantity: 1000 tonnes

Total Embedded Emissions:
  Direct: 1,805.00 tCO2e
  Indirect: 1,910.00 tCO2e
  Total: 3,715.00 tCO2e

=== Product Details ===

STEEL-001: Hot-rolled steel coils, width >= 600mm, thic...
  CN Code: 7208510091
  Quantity: 500 tonnes
  Origin: TR
  Emission Source: actual
  Specific Emissions: 2.27 tCO2e/tonne
  Total Emissions: 1,135.00 tCO2e

STEEL-002: Cold-rolled steel coils, width >= 600mm, thi...
  CN Code: 7209160000
  Quantity: 300 tonnes
  Origin: CN
  Emission Source: default_country
  Specific Emissions: 2.45 tCO2e/tonne
  Total Emissions: 735.00 tCO2e

ALUM-001: Unwrought aluminum, not alloyed...
  CN Code: 7601100000
  Quantity: 200 tonnes
  Origin: RU
  Emission Source: actual
  Specific Emissions: 10.15 tCO2e/tonne
  Total Emissions: 2,030.00 tCO2e
```

### Step 5: Estimate Certificate Costs (Definitive Period)

```python
# Certificate estimate (for planning purposes)
cert = result.certificate_estimate

print("\n=== CBAM Certificate Estimate ===")
print(f"Certificates Required: {cert.certificates_required:,}")
print(f"EU ETS Carbon Price: EUR {cert.carbon_price_eur:.2f}/tCO2e")
print(f"Gross Cost: EUR {cert.estimated_cost_eur:,.2f}")

# Foreign carbon price deductions
if cert.foreign_carbon_price_deduction:
    print("\nForeign Carbon Price Deductions:")
    for country, deduction in cert.foreign_carbon_price_deduction.items():
        print(f"  {country}:")
        print(f"    Carbon Price: EUR {deduction.price_eur_per_tco2:.2f}/tCO2e")
        print(f"    Emissions Covered: {deduction.emissions_covered:,.2f} tCO2e")
        print(f"    Deduction: EUR {deduction.deduction_eur:,.2f}")

print(f"\nNet Certificate Cost: EUR {cert.net_cost_eur:,.2f}")
print(f"\nNote: {cert.disclaimer}")
```

**Example Output:**

```
=== CBAM Certificate Estimate ===
Certificates Required: 3,715
EU ETS Carbon Price: EUR 85.50/tCO2e
Gross Cost: EUR 317,632.50

Foreign Carbon Price Deductions:
  TR:
    Carbon Price: EUR 12.50/tCO2e
    Emissions Covered: 1,135.00 tCO2e
    Deduction: EUR 14,187.50

Net Certificate Cost: EUR 303,445.00

Note: Estimate only. Actual certificate requirements determined at time of import.
```

### Step 6: Generate XML Report

```python
# Download XML report for submission
if result.xml_report.available:
    print(f"\n=== XML Report ===")
    print(f"Download URL: {result.xml_report.download_url}")
    print(f"Expires: {result.xml_report.expires_at}")

    # Download and save
    xml_content = client.download(result.xml_report.download_url)

    filename = f"cbam_q{result.reporting_period.quarter}_{result.reporting_period.year}.xml"
    with open(filename, "wb") as f:
        f.write(xml_content)

    print(f"Saved to: {filename}")
```

---

## Production Routes and Emission Factors

### Iron and Steel Production Routes

| Production Route | Typical Direct Emissions | Description |
|-----------------|--------------------------|-------------|
| `basic_oxygen_furnace` | 1.8 - 2.2 tCO2e/t | Primary production from iron ore |
| `electric_arc_furnace` | 0.3 - 0.6 tCO2e/t | Secondary production from scrap |
| `direct_reduced_iron` | 1.0 - 1.5 tCO2e/t | Gas-based reduction |

```python
# Specify production route for accurate calculations
import_with_route = {
    "cn_code": "7208510091",
    "product_category": "iron_steel",
    "quantity": 500,
    "unit": "tonnes",
    "country_of_origin": "TR",
    "production_route": "basic_oxygen_furnace",  # Specify route
    "emission_data": {
        "type": "actual",
        "direct_emissions": 1.85,
        "indirect_emissions": 0.42,
        "unit": "tCO2e/tonne"
    }
}
```

### Aluminum Production Routes

| Production Route | Typical Direct Emissions | Typical Indirect Emissions |
|-----------------|--------------------------|---------------------------|
| `primary_smelting` | 1.5 - 2.0 tCO2e/t | 5.0 - 15.0 tCO2e/t |
| `secondary_recycling` | 0.2 - 0.5 tCO2e/t | 0.3 - 0.8 tCO2e/t |

```python
# Aluminum with electricity source specification
aluminum_import = {
    "cn_code": "7601100000",
    "product_category": "aluminum",
    "quantity": 200,
    "unit": "tonnes",
    "country_of_origin": "IS",
    "production_route": "primary_smelting",
    "emission_data": {
        "type": "actual",
        "direct_emissions": 1.65,
        "indirect_emissions": 0.10,  # Low due to hydro power
        "unit": "tCO2e/tonne",
        "electricity_source": "renewable_hydro"
    }
}
```

### Cement Production

| Type | Typical Emissions | Notes |
|------|-------------------|-------|
| `clinker` | 0.8 - 0.9 tCO2e/t | Main emission source |
| `portland_cement` | 0.6 - 0.7 tCO2e/t | Includes clinker factor |
| `blended_cement` | 0.4 - 0.5 tCO2e/t | Lower clinker ratio |

---

## Using Default Values

### When to Use Default Values

Use default values when:
- Actual installation data is not available
- Supplier cannot provide verified emissions data
- Data verification is pending

### Default Value Hierarchy

```python
# Option 1: Country-specific default (preferred)
emission_data = {
    "type": "default",
    "use_country_default": True
}

# Option 2: EU average default
emission_data = {
    "type": "default",
    "use_eu_default": True
}

# Option 3: Product-specific default
emission_data = {
    "type": "default",
    "default_value_type": "product_benchmark"
}
```

### Transitional Period Flexibility

During the transitional period, reporting flexibility is allowed:

```python
# Transitional period - flexible emission sources allowed
import_transitional = {
    "cn_code": "7208510091",
    "product_category": "iron_steel",
    "quantity": 500,
    "unit": "tonnes",
    "country_of_origin": "IN",
    "emission_data": {
        "type": "transitional_estimate",
        "estimation_method": "comparable_installation",
        "comparable_country": "TR",
        "comparable_production_route": "basic_oxygen_furnace",
        "justification": "Based on similar BOF facility in Turkey due to unavailable Indian data"
    }
}
```

---

## XML Export Format

### CBAM XML Structure

GreenLang generates CBAM-compliant XML:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<CBAMReport xmlns="urn:ec.europa.eu:cbam:1.0">
  <ReportingPeriod>
    <Year>2025</Year>
    <Quarter>1</Quarter>
  </ReportingPeriod>
  <Declarant>
    <EORI>DE123456789012</EORI>
    <Name>German Steel Imports GmbH</Name>
  </Declarant>
  <Goods>
    <Good id="1">
      <CNCode>7208510091</CNCode>
      <Description>Hot-rolled steel coils</Description>
      <Quantity unit="tonnes">500</Quantity>
      <OriginCountry>TR</OriginCountry>
      <InstallationID>TR-INST-2024-12345</InstallationID>
      <EmbeddedEmissions>
        <Direct unit="tCO2e">925.00</Direct>
        <Indirect unit="tCO2e">210.00</Indirect>
        <Total unit="tCO2e">1135.00</Total>
        <SpecificEmissions unit="tCO2e/t">2.27</SpecificEmissions>
      </EmbeddedEmissions>
      <DataSource>ACTUAL_INSTALLATION</DataSource>
    </Good>
  </Goods>
  <Summary>
    <TotalQuantity unit="tonnes">1000</TotalQuantity>
    <TotalEmbeddedEmissions unit="tCO2e">3715.00</TotalEmbeddedEmissions>
  </Summary>
</CBAMReport>
```

### Validating XML Output

```python
# Validate XML before submission
validation = client.cbam.validate_xml(result.xml_report.download_url)

if validation.is_valid:
    print("XML validation passed!")
else:
    print("XML validation failed:")
    for error in validation.errors:
        print(f"  - Line {error.line}: {error.message}")
```

---

## Supplier Communication

### Requesting Emission Data from Suppliers

GreenLang can generate supplier communication templates:

```python
# Generate supplier questionnaire
questionnaire = client.cbam.generate_supplier_questionnaire(
    supplier={
        "name": "Turkish Steel Works",
        "contact_email": "exports@turkishsteel.com",
        "country": "TR"
    },
    products=[
        {"cn_code": "7208510091", "description": "Hot-rolled steel coils"}
    ],
    reporting_period={"year": 2025, "quarter": 1}
)

print(f"Questionnaire URL: {questionnaire.url}")
print(f"Token (for supplier): {questionnaire.supplier_token}")

# Send to supplier
client.cbam.send_questionnaire(
    questionnaire_id=questionnaire.id,
    message="Please complete by March 15, 2025"
)
```

### Tracking Supplier Responses

```python
# Check supplier response status
responses = client.cbam.get_questionnaire_responses(
    reporting_period={"year": 2025, "quarter": 1}
)

for response in responses:
    print(f"\n{response.supplier_name}:")
    print(f"  Status: {response.status}")
    print(f"  Sent: {response.sent_date}")

    if response.status == "completed":
        print(f"  Received: {response.completed_date}")
        print(f"  Direct Emissions: {response.data.direct_emissions} tCO2e/t")
        print(f"  Indirect Emissions: {response.data.indirect_emissions} tCO2e/t")
    elif response.status == "pending":
        print(f"  Reminder sent: {response.reminder_count} times")
```

---

## Certificate Estimation and Planning

### Estimating Annual Certificate Needs

```python
# Estimate certificates for budget planning
annual_imports = [
    # Q1 imports
    {"quarter": 1, "category": "iron_steel", "quantity": 1000, "avg_emissions": 2.3},
    {"quarter": 1, "category": "aluminum", "quantity": 200, "avg_emissions": 10.0},
    # Q2 estimates
    {"quarter": 2, "category": "iron_steel", "quantity": 1200, "avg_emissions": 2.3},
    {"quarter": 2, "category": "aluminum", "quantity": 250, "avg_emissions": 10.0},
    # Q3 estimates
    {"quarter": 3, "category": "iron_steel", "quantity": 1100, "avg_emissions": 2.3},
    {"quarter": 3, "category": "aluminum", "quantity": 220, "avg_emissions": 10.0},
    # Q4 estimates
    {"quarter": 4, "category": "iron_steel", "quantity": 1300, "avg_emissions": 2.3},
    {"quarter": 4, "category": "aluminum", "quantity": 280, "avg_emissions": 10.0},
]

total_emissions = sum(
    imp["quantity"] * imp["avg_emissions"]
    for imp in annual_imports
)

carbon_price = 85.00  # EUR per tCO2e (estimate)

print(f"Annual Estimated Emissions: {total_emissions:,.0f} tCO2e")
print(f"Estimated Certificate Cost: EUR {total_emissions * carbon_price:,.0f}")
print(f"\n(Based on EU ETS price of EUR {carbon_price}/tCO2e)")
```

### Scenario Analysis

```python
# Analyze cost under different carbon price scenarios
scenarios = [
    {"name": "Low", "price": 70.00},
    {"name": "Base", "price": 85.00},
    {"name": "High", "price": 120.00},
]

total_emissions = 5000  # tCO2e example

print("CBAM Cost Scenarios")
print("=" * 50)
for scenario in scenarios:
    cost = total_emissions * scenario["price"]
    print(f"{scenario['name']:10} (EUR {scenario['price']:6.2f}/tCO2e): EUR {cost:>12,.2f}")
```

---

## Integration Examples

### SAP Integration

```python
from sap_connector import SAPClient
from greenlang import Client

sap = SAPClient(...)
greenlang = Client(...)

def process_quarterly_cbam_report(year, quarter):
    """Generate CBAM report from SAP import data."""

    # Fetch imports from SAP
    sap_imports = sap.get_imports_by_period(
        start_date=f"{year}-{(quarter-1)*3+1:02d}-01",
        end_date=get_quarter_end_date(year, quarter),
        filter_cbam_relevant=True
    )

    # Transform to CBAM format
    imports = []
    for sap_import in sap_imports:
        imports.append({
            "product_id": sap_import.document_number,
            "cn_code": sap_import.customs_tariff,
            "product_category": map_cn_to_cbam_category(sap_import.customs_tariff),
            "description": sap_import.material_description,
            "quantity": sap_import.quantity,
            "unit": "tonnes",
            "country_of_origin": sap_import.country_of_origin,
            "emission_data": get_emission_data(sap_import)
        })

    # Calculate CBAM
    result = greenlang.calculate.cbam(
        reporting_period={"year": year, "quarter": quarter},
        imports=imports,
        options={"generate_xml": True}
    )

    # Update SAP with results
    for product in result.products:
        sap.update_cbam_emissions(
            document_number=product.product_id,
            embedded_emissions=product.embedded_emissions.total,
            cbam_calculation_id=result.calculation_id
        )

    return result
```

### Automated Quarterly Reporting

```python
from datetime import datetime
from greenlang import Client

def submit_cbam_report():
    """Automated quarterly CBAM report submission."""

    client = Client(...)

    # Determine reporting period
    now = datetime.now()
    if now.month <= 3:
        year, quarter = now.year - 1, 4
    elif now.month <= 6:
        year, quarter = now.year, 1
    elif now.month <= 9:
        year, quarter = now.year, 2
    else:
        year, quarter = now.year, 3

    # Load import data
    imports = load_quarter_imports(year, quarter)

    # Generate report
    result = client.calculate.cbam(
        reporting_period={"year": year, "quarter": quarter},
        imports=imports,
        options={
            "generate_xml": True,
            "include_certificate_estimate": True
        }
    )

    # Validate before submission
    if result.summary.non_compliant > 0:
        notify_compliance_team(result)
        return {"status": "review_required", "result": result}

    # Download XML
    xml_path = download_xml_report(result)

    # Submit to EU CBAM Portal (when API available)
    # submission = submit_to_eu_portal(xml_path)

    return {
        "status": "success",
        "calculation_id": result.calculation_id,
        "xml_path": xml_path,
        "summary": result.summary
    }
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Invalid CN code" | Code not CBAM-covered | Verify CN code is in CBAM scope |
| "Missing installation ID" | Required for actual data | Provide installation identifier |
| "Country not recognized" | Invalid country code | Use ISO 3166-1 alpha-2 codes |
| "Emissions out of range" | Implausible values | Check units (tCO2e/tonne) |
| "XML validation failed" | Schema mismatch | Review error details |

### Error Handling

```python
from greenlang.exceptions import CBAMValidationError, CBAMCalculationError

try:
    result = client.calculate.cbam(...)

except CBAMValidationError as e:
    print(f"Validation Error: {e.message}")
    for detail in e.details:
        print(f"  Product {detail.product_id}: {detail.message}")

except CBAMCalculationError as e:
    print(f"Calculation Error: {e.message}")
    print(f"Affected CN codes: {e.affected_cn_codes}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Resources

### Official EU Resources

- [CBAM Regulation Text](https://eur-lex.europa.eu/eli/reg/2023/956)
- [CBAM Implementing Regulation](https://eur-lex.europa.eu/eli/reg_impl/2023/1773)
- [EU CBAM Portal](https://cbam.ec.europa.eu) (for submissions)
- [CBAM Default Values](https://taxation-customs.ec.europa.eu/cbam-default-values)

### GreenLang Resources

- [CBAM API Reference](../api/calculations.md#cbam-carbon-border-adjustment-mechanism)
- [CN Code Lookup Tool](https://app.greenlang.io/tools/cn-lookup)
- [CBAM Webinars](https://greenlang.io/webinars/cbam)

### Support

- **Email:** cbam-support@greenlang.io
- **Community Forum:** https://community.greenlang.io/c/cbam
- **Office Hours:** Tuesdays 10-11am CET

---

## Next Steps

- [EUDR Compliance Guide](./eudr_compliance.md) - EU Deforestation Regulation
- [Scope 3 Calculations](../api/calculations.md#scope-3-emissions) - Value chain emissions
- [API Reference](../api/README.md) - Complete API documentation
