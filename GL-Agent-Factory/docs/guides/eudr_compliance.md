# EUDR Compliance Guide

## Overview

The EU Deforestation Regulation (EUDR - Regulation 2023/1115) requires companies placing certain commodities on the EU market to prove that their products are "deforestation-free" and legally produced. This guide explains how to use GreenLang to verify EUDR compliance.

**Key Dates:**

| Date | Milestone |
|------|-----------|
| June 29, 2023 | EUDR entered into force |
| December 30, 2024 | Large operators compliance deadline |
| June 30, 2025 | SME operators compliance deadline |
| December 30, 2025 | Country benchmarking system operational |

---

## EUDR Requirements Summary

### Covered Commodities

The EUDR applies to seven commodities and their derived products:

| Commodity | Example Products |
|-----------|------------------|
| **Cattle** | Beef, leather, hides |
| **Cocoa** | Chocolate, cocoa butter, cocoa powder |
| **Coffee** | Roasted coffee, coffee extracts |
| **Palm Oil** | Crude palm oil, palm kernel oil, derivatives |
| **Rubber** | Natural rubber, tires, rubber products |
| **Soya** | Soybeans, soybean oil, soybean meal |
| **Wood** | Timber, furniture, paper, charcoal |

### Compliance Requirements

Operators must:

1. **Collect Geolocation Data** - Precise coordinates of production plots
2. **Verify Deforestation-Free** - No deforestation after December 31, 2020
3. **Ensure Legality** - Compliance with local laws
4. **Conduct Due Diligence** - Risk assessment and mitigation
5. **Submit Due Diligence Statement (DDS)** - Before placing on market

---

## Data Preparation

### Required Data Fields

For each commodity shipment, you must provide:

| Field | Required | Description |
|-------|----------|-------------|
| `type` | Yes | Commodity type |
| `hs_code` | Yes | Harmonized System code |
| `quantity` | Yes | Quantity with unit |
| `country_of_production` | Yes | ISO 3166-1 alpha-2 code |
| `geolocation` | Yes | GeoJSON coordinates |
| `production_date` | Yes | Date of harvest/production |
| `supplier.name` | Yes | Supplier name |
| `supplier.registration_number` | Recommended | Supplier registration |
| `certifications` | Optional | Relevant certifications |

### Geolocation Data Formats

GreenLang accepts geolocation data in GeoJSON format:

**Point (for plots < 4 hectares):**

```json
{
  "type": "point",
  "coordinates": [-1.5, 6.2]
}
```

**Polygon (for larger plots):**

```json
{
  "type": "polygon",
  "coordinates": [
    [
      [-1.5, 6.2],
      [-1.4, 6.2],
      [-1.4, 6.3],
      [-1.5, 6.3],
      [-1.5, 6.2]
    ]
  ]
}
```

**MultiPolygon (for multiple non-contiguous plots):**

```json
{
  "type": "multipolygon",
  "coordinates": [
    [[[-1.5, 6.2], [-1.4, 6.2], [-1.4, 6.3], [-1.5, 6.3], [-1.5, 6.2]]],
    [[[-1.3, 6.1], [-1.2, 6.1], [-1.2, 6.2], [-1.3, 6.2], [-1.3, 6.1]]]
  ]
}
```

### Data Template

Download our data templates:

- [EUDR Data Template (Excel)](https://docs.greenlang.io/templates/eudr_template.xlsx)
- [EUDR Data Template (CSV)](https://docs.greenlang.io/templates/eudr_template.csv)
- [EUDR Data Template (JSON)](https://docs.greenlang.io/templates/eudr_template.json)

---

## Step-by-Step Workflow

### Step 1: Set Up Your Environment

**Python SDK:**

```python
from greenlang import Client

client = Client(
    client_id="your_client_id",
    client_secret="your_client_secret"
)
```

### Step 2: Prepare Commodity Data

```python
# Example: Cocoa import from Ghana
commodities = [
    {
        "id": "shipment_001",
        "type": "cocoa",
        "product_description": "Cocoa beans, raw, fermented",
        "hs_code": "1801.00.00",
        "quantity": 50,
        "unit": "tonnes",
        "country_of_production": "GH",
        "geolocation": {
            "type": "polygon",
            "coordinates": [
                [
                    [-1.6234, 6.1456],
                    [-1.5987, 6.1456],
                    [-1.5987, 6.1789],
                    [-1.6234, 6.1789],
                    [-1.6234, 6.1456]
                ]
            ]
        },
        "production_date": "2024-10-15",
        "supplier": {
            "name": "Ashanti Cocoa Farmers Cooperative",
            "registration_number": "GH-COC-2023-12345",
            "address": "Kumasi, Ashanti Region, Ghana"
        }
    },
    {
        "id": "shipment_002",
        "type": "cocoa",
        "product_description": "Cocoa beans, raw, fermented",
        "hs_code": "1801.00.00",
        "quantity": 30,
        "unit": "tonnes",
        "country_of_production": "CI",
        "geolocation": {
            "type": "point",
            "coordinates": [-5.3456, 7.5678]
        },
        "production_date": "2024-10-20",
        "supplier": {
            "name": "Ivory Coast Cocoa Ltd",
            "registration_number": "CI-COC-2023-67890"
        },
        "certifications": ["Rainforest Alliance", "UTZ"]
    }
]
```

### Step 3: Submit for Compliance Verification

```python
result = client.calculate.eudr(
    operator={
        "name": "European Chocolate Manufacturing GmbH",
        "eori_number": "DE123456789012",
        "country": "DE",
        "contact": {
            "name": "Maria Schmidt",
            "email": "compliance@euro-choc.de"
        }
    },
    commodities=commodities,
    options={
        "deforestation_cutoff_date": "2020-12-31",
        "include_satellite_analysis": True,
        "generate_dds": True,
        "risk_mitigation_required": True
    }
)

print(f"Assessment ID: {result.calculation_id}")
print(f"Overall Risk Level: {result.summary.overall_risk_level}")
print(f"Compliant: {result.summary.compliant}/{result.summary.total_commodities}")
```

### Step 4: Review Results

```python
# Check each commodity's compliance status
for commodity in result.commodities:
    print(f"\n--- {commodity.id} ---")
    print(f"Type: {commodity.type}")
    print(f"Country: {commodity.country_of_production}")
    print(f"Status: {commodity.compliance_status}")
    print(f"Risk Level: {commodity.risk_level}")

    # Geolocation validation
    geo = commodity.geolocation_validation
    print(f"Geolocation Valid: {geo.status}")
    if geo.status != "valid":
        print(f"  Issue: {geo.error_message}")

    # Deforestation analysis
    defor = commodity.deforestation_analysis
    print(f"Deforestation Analysis: {defor.status}")
    print(f"  Forest Cover 2020: {defor.forest_cover_2020}%")
    print(f"  Forest Cover Current: {defor.forest_cover_current}%")
    print(f"  Deforestation Detected: {defor.deforestation_detected}")

    # Legality check
    legal = commodity.legality_check
    print(f"Legality: {legal.status}")
```

**Example Output:**

```
--- shipment_001 ---
Type: cocoa
Country: GH
Status: compliant
Risk Level: low
Geolocation Valid: valid
Deforestation Analysis: clear
  Forest Cover 2020: 0.0%
  Forest Cover Current: 0.0%
  Deforestation Detected: False
Legality: verified

--- shipment_002 ---
Type: cocoa
Country: CI
Status: requires_review
Risk Level: standard
Geolocation Valid: valid
Deforestation Analysis: review_required
  Forest Cover 2020: 12.5%
  Forest Cover Current: 11.8%
  Deforestation Detected: True
Legality: pending_verification
```

### Step 5: Handle Risk Mitigation

When a commodity requires review, implement risk mitigation:

```python
# Get detailed risk information
for commodity in result.commodities:
    if commodity.compliance_status == "requires_review":
        print(f"\nRisk Mitigation Required for {commodity.id}:")

        # Check specific risks
        if commodity.deforestation_analysis.deforestation_detected:
            print("  - Deforestation detected in production area")
            print(f"    Area affected: {commodity.deforestation_analysis.deforestation_area_hectares} ha")
            print(f"    Confidence: {commodity.deforestation_analysis.confidence * 100}%")
            print("\n  Recommended Actions:")
            print("    1. Request detailed plot boundaries from supplier")
            print("    2. Verify production plot does not overlap deforested area")
            print("    3. Obtain satellite imagery for specific plot")

        if commodity.legality_check.status == "pending_verification":
            print("  - Legal documentation incomplete")
            print("\n  Required Documents:")
            for doc in commodity.legality_check.required_documents:
                print(f"    - {doc.name}: {doc.status}")
```

### Step 6: Generate Due Diligence Statement

```python
# Generate DDS once compliance is verified
if result.summary.non_compliant == 0:
    dds = result.due_diligence_statement

    print(f"\nDue Diligence Statement Generated:")
    print(f"Reference Number: {dds.reference_number}")
    print(f"Status: {dds.status}")
    print(f"Download URL: {dds.download_url}")

    # Download the DDS
    dds_content = client.download(dds.download_url)
    with open(f"dds_{dds.reference_number}.pdf", "wb") as f:
        f.write(dds_content)

    print(f"DDS saved to: dds_{dds.reference_number}.pdf")
else:
    print("\nCannot generate DDS - non-compliant commodities detected")
    print("Resolve all issues before generating Due Diligence Statement")
```

---

## Geolocation Validation

### Validation Checks

GreenLang performs these geolocation validations:

| Check | Description |
|-------|-------------|
| Format Validity | GeoJSON format is correct |
| Coordinate Range | Coordinates are within valid ranges |
| Country Boundary | Coordinates fall within declared country |
| Area Calculation | Polygon area is calculated correctly |
| Plot Size | Verifies plot size against thresholds |

### Common Validation Errors

```python
# Handle geolocation validation errors
for commodity in result.commodities:
    geo = commodity.geolocation_validation

    if geo.status == "invalid":
        print(f"\nGeolocation Error for {commodity.id}:")
        print(f"Error Code: {geo.error_code}")
        print(f"Message: {geo.error_message}")

        if geo.error_code == "outside_country_boundary":
            print("  Fix: Verify coordinates match declared country of production")

        elif geo.error_code == "invalid_format":
            print("  Fix: Ensure coordinates are in [longitude, latitude] format")

        elif geo.error_code == "self_intersecting_polygon":
            print("  Fix: Polygon boundaries should not cross themselves")

        elif geo.error_code == "insufficient_precision":
            print("  Fix: Coordinates require at least 4 decimal places")
```

### Best Practices for Geolocation Data

1. **Use WGS84 coordinate system** (EPSG:4326)
2. **Format as [longitude, latitude]** (not lat/long)
3. **Minimum 4 decimal places** for precision (~11m accuracy)
4. **Close polygons** - first and last coordinate must match
5. **Verify against maps** before submission

---

## Satellite Analysis

### How It Works

GreenLang's satellite analysis:

1. **Retrieves historical imagery** from Global Forest Watch, Sentinel-2, and Landsat
2. **Analyzes forest cover** for the cutoff date (December 31, 2020)
3. **Compares with current imagery** to detect changes
4. **Calculates deforestation** if any occurred

### Understanding Results

```python
defor = commodity.deforestation_analysis

print(f"Analysis Status: {defor.status}")
# Possible values: clear, review_required, non_compliant

print(f"Forest Cover on Cutoff Date: {defor.forest_cover_2020}%")
print(f"Current Forest Cover: {defor.forest_cover_current}%")

print(f"Deforestation Detected: {defor.deforestation_detected}")
if defor.deforestation_detected:
    print(f"Deforested Area: {defor.deforestation_area_hectares} hectares")

print(f"Analysis Source: {defor.analysis_source}")
print(f"Confidence Level: {defor.confidence * 100}%")
```

### Interpretation Guide

| Status | Meaning | Action Required |
|--------|---------|-----------------|
| `clear` | No deforestation detected | Proceed with DDS |
| `review_required` | Possible deforestation, needs review | Manual verification |
| `non_compliant` | Confirmed deforestation after cutoff | Cannot proceed |

---

## Country Risk Classification

### Risk Levels

The EU classifies countries into three risk levels:

| Risk Level | Description | Due Diligence Requirements |
|------------|-------------|---------------------------|
| **Low** | Minimal deforestation risk | Simplified due diligence |
| **Standard** | Moderate risk | Standard due diligence |
| **High** | Significant deforestation risk | Enhanced due diligence |

### Checking Country Risk

```python
# Get country risk information
for country_code, risk_info in result.country_risk_classification.items():
    print(f"\n{risk_info.country} ({country_code}):")
    print(f"  Risk Level: {risk_info.risk_level}")
    print(f"  Source: {risk_info.source}")

    if risk_info.risk_level == "high":
        print("  Note: Enhanced due diligence required for this origin")
```

### Enhanced Due Diligence for High-Risk Countries

```python
# For high-risk countries, additional checks are required
high_risk_commodities = [
    c for c in result.commodities
    if result.country_risk_classification[c.country_of_production].risk_level == "high"
]

for commodity in high_risk_commodities:
    print(f"\nEnhanced Due Diligence for {commodity.id}:")
    print("Required additional measures:")
    print("  1. Independent third-party verification")
    print("  2. On-site audits or inspections")
    print("  3. Additional satellite imagery analysis")
    print("  4. Detailed supplier questionnaires")
```

---

## Due Diligence Statement (DDS)

### DDS Contents

The Due Diligence Statement includes:

1. **Operator Information** - Name, EORI number, address
2. **Commodity Details** - HS code, description, quantity
3. **Geolocation Data** - Coordinates of production plots
4. **Supplier Information** - Name, registration, country
5. **Risk Assessment** - Risk level and mitigation measures
6. **Compliance Declaration** - Deforestation-free and legal

### Generating the DDS

```python
# Verify all commodities are compliant first
if result.summary.non_compliant == 0 and result.summary.requires_review == 0:
    # Generate final DDS
    dds = client.eudr.generate_dds(
        assessment_id=result.calculation_id,
        declaration={
            "deforestation_free": True,
            "legally_produced": True,
            "authorized_signatory": {
                "name": "Dr. Hans Mueller",
                "title": "Chief Compliance Officer",
                "date": "2025-01-15"
            }
        }
    )

    print(f"DDS Reference: {dds.reference_number}")
    print(f"Valid Until: {dds.valid_until}")
    print(f"Download: {dds.download_url}")

    # Submit to EU Information System (when available)
    if dds.can_submit_to_eu:
        submission = client.eudr.submit_dds(dds.reference_number)
        print(f"Submitted to EU: {submission.eu_reference_number}")
```

### DDS Validity

- DDS is valid for **single import** of specified commodities
- Must be generated **before** placing goods on the EU market
- Reference number must be included in **customs declaration**
- Keep records for **minimum 5 years**

---

## Batch Processing

### Processing Multiple Shipments

```python
# Load commodities from file
import json

with open("commodities_batch.json", "r") as f:
    commodities_batch = json.load(f)

print(f"Processing {len(commodities_batch)} commodities...")

# Submit batch for processing
result = client.calculate.eudr(
    operator={
        "name": "European Chocolate Manufacturing GmbH",
        "eori_number": "DE123456789012",
        "country": "DE"
    },
    commodities=commodities_batch,
    options={
        "include_satellite_analysis": True,
        "generate_dds": True,
        "async": True  # Use async for large batches
    }
)

# For async processing, poll for results
if result.status == "processing":
    print(f"Job ID: {result.job_id}")
    print("Processing in background...")

    # Wait for completion
    final_result = client.wait_for_job(result.job_id, timeout=3600)
    print(f"Processing complete!")
    print(f"Compliant: {final_result.summary.compliant}")
    print(f"Requires Review: {final_result.summary.requires_review}")
    print(f"Non-Compliant: {final_result.summary.non_compliant}")
```

### Handling Large Datasets

For datasets with thousands of commodities:

```python
import time

# Split into batches of 100
BATCH_SIZE = 100
batches = [
    commodities_all[i:i + BATCH_SIZE]
    for i in range(0, len(commodities_all), BATCH_SIZE)
]

all_results = []

for i, batch in enumerate(batches):
    print(f"Processing batch {i + 1}/{len(batches)}...")

    result = client.calculate.eudr(
        operator=operator_info,
        commodities=batch,
        options={
            "include_satellite_analysis": True,
            "generate_dds": False  # Generate DDS separately after all batches
        }
    )

    all_results.extend(result.commodities)

    # Respect rate limits
    time.sleep(1)

# Summarize results
compliant = sum(1 for r in all_results if r.compliance_status == "compliant")
review = sum(1 for r in all_results if r.compliance_status == "requires_review")
non_compliant = sum(1 for r in all_results if r.compliance_status == "non_compliant")

print(f"\nTotal Results:")
print(f"  Compliant: {compliant}")
print(f"  Requires Review: {review}")
print(f"  Non-Compliant: {non_compliant}")
```

---

## Integration with Supply Chain Systems

### ERP Integration Example

```python
# Example: SAP Integration
from sap_connector import SAPClient

sap = SAPClient(...)
greenlang = Client(...)

# Fetch pending imports from SAP
pending_imports = sap.get_materials_by_status("pending_eudr_check")

for material in pending_imports:
    # Convert SAP data to EUDR format
    commodity = {
        "id": material.material_number,
        "type": map_material_to_eudr_type(material.material_group),
        "hs_code": material.customs_tariff_number,
        "quantity": material.quantity,
        "unit": material.unit_of_measure,
        "country_of_production": material.country_of_origin,
        "geolocation": {
            "type": "point",
            "coordinates": [material.origin_longitude, material.origin_latitude]
        },
        "production_date": material.production_date,
        "supplier": {
            "name": material.vendor_name,
            "registration_number": material.vendor_number
        }
    }

    # Verify compliance
    result = greenlang.calculate.eudr(
        operator=operator_info,
        commodities=[commodity],
        options={"include_satellite_analysis": True}
    )

    # Update SAP with results
    sap.update_material_eudr_status(
        material_number=material.material_number,
        status=result.commodities[0].compliance_status,
        risk_level=result.commodities[0].risk_level,
        dds_reference=result.due_diligence_statement.reference_number if result.due_diligence_statement else None
    )
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Coordinates outside country boundary" | Wrong country code or coordinates | Verify country matches actual production location |
| "Invalid HS code" | HS code not covered by EUDR | Check HS code is for covered commodity |
| "Production date before cutoff" | Date parsing error | Use ISO 8601 format (YYYY-MM-DD) |
| "Supplier not found" | Missing or invalid registration | Add complete supplier information |
| "Satellite data unavailable" | Cloud cover or data gaps | Request alternative date range analysis |

### Error Handling

```python
from greenlang.exceptions import EUDRValidationError, GeoLocationError

try:
    result = client.calculate.eudr(...)

except GeoLocationError as e:
    print(f"Geolocation Error: {e.message}")
    print(f"Affected Commodities: {e.affected_ids}")
    print(f"Suggestion: {e.suggestion}")

except EUDRValidationError as e:
    print(f"EUDR Validation Error: {e.message}")
    for detail in e.details:
        print(f"  - {detail.field}: {detail.message}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Resources

### Official Resources

- [EU EUDR Regulation Text](https://eur-lex.europa.eu/eli/reg/2023/1115)
- [EU EUDR FAQ](https://environment.ec.europa.eu/topics/forests/deforestation/regulation-deforestation-free-products_en)
- [EU EUDR Information System](https://eudr.ec.europa.eu) (when available)

### GreenLang Resources

- [EUDR API Reference](../api/calculations.md#eudr-eu-deforestation-regulation)
- [Data Templates](https://docs.greenlang.io/templates/)
- [Webinar: EUDR Compliance](https://greenlang.io/webinars/eudr)

### Support

- **Email:** eudr-support@greenlang.io
- **Community Forum:** https://community.greenlang.io/c/eudr
- **Office Hours:** Thursdays 2-3pm CET

---

## Next Steps

- [CBAM Reporting Guide](./cbam_reporting.md) - Carbon Border Adjustment Mechanism
- [Scope 3 Calculations](../api/calculations.md#scope-3-emissions) - Value chain emissions
- [API Reference](../api/README.md) - Complete API documentation
