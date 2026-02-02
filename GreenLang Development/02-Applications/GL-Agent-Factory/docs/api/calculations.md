# Calculation Endpoints

## Overview

The Calculation API provides direct access to GreenLang's emissions calculation engines. These endpoints offer a streamlined interface for common calculation tasks without the overhead of agent configuration.

**Base URL:** `https://api.greenlang.io/v1`

---

## Fuel Emissions

Calculate greenhouse gas emissions from fuel combustion.

### POST /v1/calculate/fuel

**Description:**

Calculate CO2, CH4, N2O, and total CO2e emissions from burning various fuel types. Supports multiple emission factor databases including IPCC, EPA, and DEFRA.

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | Bearer token or API key |
| `Content-Type` | Yes | `application/json` |

**Request Body:**

```json
{
  "fuel_type": "diesel",
  "quantity": 1000,
  "unit": "liters",
  "emission_factor_source": "ipcc_2023",
  "options": {
    "include_breakdown": true,
    "output_unit": "kg",
    "gwp_version": "ar6"
  }
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `fuel_type` | string | Yes | Type of fuel |
| `quantity` | number | Yes | Amount of fuel consumed (must be > 0) |
| `unit` | string | Yes | Unit of measurement |
| `emission_factor_source` | string | No | Emission factor database (default: `ipcc_2023`) |
| `options.include_breakdown` | boolean | No | Include calculation details (default: false) |
| `options.output_unit` | string | No | Output unit: `kg`, `tonnes`, `lbs` (default: `kg`) |
| `options.gwp_version` | string | No | GWP version: `ar5`, `ar6` (default: `ar6`) |

**Supported Fuel Types:**

| Fuel Type | Description |
|-----------|-------------|
| `diesel` | Diesel fuel (road vehicles) |
| `gasoline` | Gasoline/petrol |
| `natural_gas` | Natural gas |
| `lpg` | Liquefied petroleum gas |
| `coal` | Coal (various grades) |
| `fuel_oil` | Heavy fuel oil |
| `kerosene` | Kerosene/jet fuel |
| `biodiesel` | Biodiesel blends |
| `ethanol` | Ethanol fuel |
| `marine_diesel` | Marine diesel oil |
| `aviation_gasoline` | Aviation gasoline |

**Supported Units:**

| Unit | Description |
|------|-------------|
| `liters` | Liters (L) |
| `gallons` | US gallons |
| `kg` | Kilograms |
| `tonnes` | Metric tonnes |
| `cubic_meters` | Cubic meters (m3) |
| `mmbtu` | Million BTU |
| `therms` | Therms |
| `kwh` | Kilowatt-hours |

**Response (200 OK):**

```json
{
  "data": {
    "calculation_id": "calc_fuel_abc123",
    "inputs": {
      "fuel_type": "diesel",
      "quantity": 1000,
      "unit": "liters"
    },
    "emissions": {
      "co2": 2680.50,
      "ch4": 0.12,
      "n2o": 0.08,
      "co2e": 2705.20,
      "unit": "kg"
    },
    "breakdown": {
      "emission_factors": {
        "co2": 2.6805,
        "ch4": 0.00012,
        "n2o": 0.00008,
        "unit": "kg/liter",
        "source": "IPCC 2023 Guidelines for National Greenhouse Gas Inventories"
      },
      "gwp_values": {
        "ch4": 28,
        "n2o": 265,
        "source": "IPCC AR6 (2021)"
      },
      "calculation_steps": [
        {
          "gas": "CO2",
          "formula": "quantity x emission_factor",
          "calculation": "1000 x 2.6805",
          "result": 2680.50,
          "unit": "kg"
        },
        {
          "gas": "CH4",
          "formula": "quantity x emission_factor",
          "calculation": "1000 x 0.00012",
          "result": 0.12,
          "unit": "kg"
        },
        {
          "gas": "N2O",
          "formula": "quantity x emission_factor",
          "calculation": "1000 x 0.00008",
          "result": 0.08,
          "unit": "kg"
        },
        {
          "gas": "CO2e",
          "formula": "CO2 + (CH4 x GWP_CH4) + (N2O x GWP_N2O)",
          "calculation": "2680.50 + (0.12 x 28) + (0.08 x 265)",
          "result": 2705.20,
          "unit": "kg"
        }
      ]
    },
    "methodology": {
      "name": "GHG Protocol",
      "version": "2023",
      "scope": "Scope 1 - Direct Emissions"
    },
    "created_at": "2025-01-15T10:30:00Z"
  },
  "meta": {
    "request_id": "req_xyz789abc123",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

**Code Examples:**

**Python:**

```python
import requests

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

data = {
    "fuel_type": "diesel",
    "quantity": 1000,
    "unit": "liters",
    "emission_factor_source": "ipcc_2023",
    "options": {
        "include_breakdown": True,
        "output_unit": "kg"
    }
}

response = requests.post(
    "https://api.greenlang.io/v1/calculate/fuel",
    headers=headers,
    json=data
)

result = response.json()["data"]
print(f"CO2e Emissions: {result['emissions']['co2e']:,.2f} {result['emissions']['unit']}")
```

**cURL:**

```bash
curl -X POST "https://api.greenlang.io/v1/calculate/fuel" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "fuel_type": "diesel",
    "quantity": 1000,
    "unit": "liters",
    "emission_factor_source": "ipcc_2023",
    "options": {
      "include_breakdown": true
    }
  }'
```

**JavaScript:**

```javascript
const response = await fetch('https://api.greenlang.io/v1/calculate/fuel', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${accessToken}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    fuel_type: 'diesel',
    quantity: 1000,
    unit: 'liters',
    emission_factor_source: 'ipcc_2023',
    options: {
      include_breakdown: true
    }
  })
});

const { data } = await response.json();
console.log(`CO2e Emissions: ${data.emissions.co2e.toLocaleString()} ${data.emissions.unit}`);
```

---

## CBAM (Carbon Border Adjustment Mechanism)

Calculate embedded emissions for EU CBAM reporting.

### POST /v1/calculate/cbam

**Description:**

Calculate embedded emissions for goods imported into the EU, following CBAM methodology. Supports all CBAM-covered product categories including cement, iron/steel, aluminum, fertilizers, electricity, and hydrogen.

**Request Body:**

```json
{
  "reporting_period": {
    "year": 2025,
    "quarter": 1
  },
  "imports": [
    {
      "product_id": "prod_001",
      "cn_code": "7208510091",
      "product_category": "iron_steel",
      "description": "Hot-rolled steel coils",
      "quantity": 500,
      "unit": "tonnes",
      "country_of_origin": "TR",
      "installation_id": "TR-INST-12345",
      "production_route": "basic_oxygen_furnace",
      "emission_data": {
        "type": "actual",
        "direct_emissions": 1.85,
        "indirect_emissions": 0.42,
        "unit": "tCO2e/tonne"
      }
    },
    {
      "product_id": "prod_002",
      "cn_code": "7601100000",
      "product_category": "aluminum",
      "description": "Unwrought aluminum",
      "quantity": 200,
      "unit": "tonnes",
      "country_of_origin": "IN",
      "emission_data": {
        "type": "default",
        "use_country_default": true
      }
    }
  ],
  "options": {
    "include_certificate_estimate": true,
    "carbon_price_eur": 85.50,
    "generate_xml": true
  }
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `reporting_period.year` | integer | Yes | Reporting year |
| `reporting_period.quarter` | integer | Yes | Reporting quarter (1-4) |
| `imports` | array | Yes | Array of imported products |
| `imports[].cn_code` | string | Yes | CN (Combined Nomenclature) code |
| `imports[].product_category` | string | Yes | CBAM product category |
| `imports[].quantity` | number | Yes | Import quantity |
| `imports[].unit` | string | Yes | Unit: `tonnes`, `mwh` (electricity) |
| `imports[].country_of_origin` | string | Yes | ISO 3166-1 alpha-2 country code |
| `imports[].emission_data.type` | string | Yes | `actual` or `default` |
| `options.include_certificate_estimate` | boolean | No | Estimate CBAM certificates needed |
| `options.carbon_price_eur` | number | No | EU ETS carbon price for estimate |
| `options.generate_xml` | boolean | No | Generate CBAM XML report |

**CBAM Product Categories:**

| Category | Description |
|----------|-------------|
| `cement` | Cement and clinker |
| `iron_steel` | Iron and steel products |
| `aluminum` | Aluminum and aluminum products |
| `fertilizers` | Nitrogen-based fertilizers |
| `electricity` | Electricity |
| `hydrogen` | Hydrogen |

**Response (200 OK):**

```json
{
  "data": {
    "calculation_id": "calc_cbam_def456",
    "reporting_period": {
      "year": 2025,
      "quarter": 1,
      "start_date": "2025-01-01",
      "end_date": "2025-03-31"
    },
    "summary": {
      "total_imports": 2,
      "total_quantity": 700,
      "quantity_unit": "tonnes",
      "total_embedded_emissions": {
        "direct": 1185.00,
        "indirect": 310.00,
        "total": 1495.00,
        "unit": "tCO2e"
      }
    },
    "products": [
      {
        "product_id": "prod_001",
        "cn_code": "7208510091",
        "product_category": "iron_steel",
        "quantity": 500,
        "unit": "tonnes",
        "country_of_origin": "TR",
        "embedded_emissions": {
          "direct": 925.00,
          "indirect": 210.00,
          "total": 1135.00,
          "unit": "tCO2e",
          "specific_emissions": {
            "direct": 1.85,
            "indirect": 0.42,
            "total": 2.27,
            "unit": "tCO2e/tonne"
          }
        },
        "emission_source": "actual",
        "installation_id": "TR-INST-12345"
      },
      {
        "product_id": "prod_002",
        "cn_code": "7601100000",
        "product_category": "aluminum",
        "quantity": 200,
        "unit": "tonnes",
        "country_of_origin": "IN",
        "embedded_emissions": {
          "direct": 260.00,
          "indirect": 100.00,
          "total": 360.00,
          "unit": "tCO2e",
          "specific_emissions": {
            "direct": 1.30,
            "indirect": 0.50,
            "total": 1.80,
            "unit": "tCO2e/tonne"
          }
        },
        "emission_source": "default_country",
        "default_value_source": "EU CBAM Default Values - India (2024)"
      }
    ],
    "certificate_estimate": {
      "certificates_required": 1495,
      "carbon_price_eur": 85.50,
      "estimated_cost_eur": 127822.50,
      "foreign_carbon_price_deduction": {
        "TR": {
          "price_eur_per_tco2": 12.50,
          "emissions_covered": 1135.00,
          "deduction_eur": 14187.50
        }
      },
      "net_cost_eur": 113635.00,
      "disclaimer": "Estimate only. Actual certificate requirements determined at time of import."
    },
    "xml_report": {
      "available": true,
      "download_url": "/v1/downloads/cbam_report_def456.xml",
      "expires_at": "2025-01-16T10:30:00Z"
    },
    "methodology": {
      "regulation": "EU Regulation 2023/956 (CBAM)",
      "default_values_source": "EU CBAM Implementing Regulation",
      "version": "2024.1"
    },
    "created_at": "2025-01-15T10:30:00Z"
  },
  "meta": {
    "request_id": "req_cbam789xyz",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

**Code Examples:**

**Python:**

```python
import requests

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

data = {
    "reporting_period": {
        "year": 2025,
        "quarter": 1
    },
    "imports": [
        {
            "product_id": "steel_coils_001",
            "cn_code": "7208510091",
            "product_category": "iron_steel",
            "description": "Hot-rolled steel coils",
            "quantity": 500,
            "unit": "tonnes",
            "country_of_origin": "TR",
            "installation_id": "TR-INST-12345",
            "emission_data": {
                "type": "actual",
                "direct_emissions": 1.85,
                "indirect_emissions": 0.42,
                "unit": "tCO2e/tonne"
            }
        }
    ],
    "options": {
        "include_certificate_estimate": True,
        "carbon_price_eur": 85.50,
        "generate_xml": True
    }
}

response = requests.post(
    "https://api.greenlang.io/v1/calculate/cbam",
    headers=headers,
    json=data
)

result = response.json()["data"]

print(f"Total Embedded Emissions: {result['summary']['total_embedded_emissions']['total']} tCO2e")
print(f"Estimated Certificate Cost: EUR {result['certificate_estimate']['net_cost_eur']:,.2f}")

# Download XML report
if result.get("xml_report", {}).get("available"):
    xml_url = result["xml_report"]["download_url"]
    print(f"XML Report available at: {xml_url}")
```

**cURL:**

```bash
curl -X POST "https://api.greenlang.io/v1/calculate/cbam" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "reporting_period": {"year": 2025, "quarter": 1},
    "imports": [{
      "cn_code": "7208510091",
      "product_category": "iron_steel",
      "quantity": 500,
      "unit": "tonnes",
      "country_of_origin": "TR",
      "emission_data": {
        "type": "actual",
        "direct_emissions": 1.85,
        "indirect_emissions": 0.42,
        "unit": "tCO2e/tonne"
      }
    }],
    "options": {"include_certificate_estimate": true}
  }'
```

---

## Building Energy

Calculate emissions from building energy consumption.

### POST /v1/calculate/building

**Description:**

Calculate Scope 1 and Scope 2 emissions from building energy consumption including electricity, natural gas, heating oil, and district heating/cooling.

**Request Body:**

```json
{
  "building": {
    "name": "Corporate HQ",
    "type": "office",
    "location": {
      "country": "US",
      "state": "CA",
      "zip_code": "94105",
      "grid_region": "CAMX"
    },
    "area": {
      "value": 50000,
      "unit": "sqft"
    }
  },
  "period": {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  },
  "energy_consumption": {
    "electricity": {
      "quantity": 1200000,
      "unit": "kwh",
      "renewable_percentage": 25
    },
    "natural_gas": {
      "quantity": 50000,
      "unit": "therms"
    },
    "district_heating": {
      "quantity": 100000,
      "unit": "kwh"
    }
  },
  "options": {
    "include_intensity_metrics": true,
    "emission_factor_source": "epa_egrid_2024",
    "market_based_accounting": true
  }
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `building.type` | string | Yes | Building type: `office`, `retail`, `warehouse`, `residential`, `industrial`, `mixed` |
| `building.location.country` | string | Yes | ISO 3166-1 alpha-2 country code |
| `building.location.grid_region` | string | No | Electricity grid region (US: eGRID subregion) |
| `building.area.value` | number | No | Building floor area |
| `building.area.unit` | string | No | Area unit: `sqft`, `sqm` |
| `period.start_date` | string | Yes | Period start (ISO 8601 date) |
| `period.end_date` | string | Yes | Period end (ISO 8601 date) |
| `energy_consumption` | object | Yes | Energy consumption by type |
| `options.include_intensity_metrics` | boolean | No | Include per-sqft metrics |
| `options.market_based_accounting` | boolean | No | Use market-based emission factors |

**Response (200 OK):**

```json
{
  "data": {
    "calculation_id": "calc_bldg_ghi789",
    "building": {
      "name": "Corporate HQ",
      "type": "office",
      "location": "San Francisco, CA, US",
      "area": {
        "value": 50000,
        "unit": "sqft"
      }
    },
    "period": {
      "start_date": "2024-01-01",
      "end_date": "2024-12-31",
      "days": 366
    },
    "emissions": {
      "scope1": {
        "natural_gas": 265.50,
        "total": 265.50,
        "unit": "tCO2e"
      },
      "scope2": {
        "location_based": {
          "electricity": 342.00,
          "district_heating": 21.50,
          "total": 363.50,
          "unit": "tCO2e"
        },
        "market_based": {
          "electricity": 256.50,
          "district_heating": 21.50,
          "total": 278.00,
          "unit": "tCO2e",
          "renewable_offset": 85.50
        }
      },
      "total": {
        "location_based": 629.00,
        "market_based": 543.50,
        "unit": "tCO2e"
      }
    },
    "intensity_metrics": {
      "emissions_per_sqft": {
        "location_based": 12.58,
        "market_based": 10.87,
        "unit": "kgCO2e/sqft"
      },
      "emissions_per_sqm": {
        "location_based": 135.36,
        "market_based": 117.00,
        "unit": "kgCO2e/sqm"
      },
      "energy_use_intensity": {
        "value": 85.2,
        "unit": "kBtu/sqft"
      }
    },
    "breakdown": {
      "electricity": {
        "consumption": 1200000,
        "unit": "kwh",
        "emission_factor": {
          "location_based": 0.285,
          "market_based": 0.214,
          "unit": "kgCO2e/kwh",
          "source": "EPA eGRID 2024 - CAMX"
        },
        "renewable_offset": {
          "percentage": 25,
          "kwh_offset": 300000
        }
      },
      "natural_gas": {
        "consumption": 50000,
        "unit": "therms",
        "emission_factor": {
          "value": 5.31,
          "unit": "kgCO2e/therm",
          "source": "EPA GHG Emission Factors Hub 2024"
        }
      },
      "district_heating": {
        "consumption": 100000,
        "unit": "kwh",
        "emission_factor": {
          "value": 0.215,
          "unit": "kgCO2e/kwh",
          "source": "Local utility data"
        }
      }
    },
    "benchmarks": {
      "energy_star_score_estimate": 72,
      "comparison_to_median": {
        "percentage": -15,
        "description": "15% below median for office buildings in this region"
      }
    },
    "methodology": {
      "name": "GHG Protocol - Corporate Standard",
      "scope2_guidance": "GHG Protocol Scope 2 Guidance (2015)",
      "version": "2024"
    },
    "created_at": "2025-01-15T10:30:00Z"
  },
  "meta": {
    "request_id": "req_bldg123abc",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

**Code Examples:**

**Python:**

```python
import requests

data = {
    "building": {
        "name": "Corporate HQ",
        "type": "office",
        "location": {
            "country": "US",
            "state": "CA",
            "grid_region": "CAMX"
        },
        "area": {"value": 50000, "unit": "sqft"}
    },
    "period": {
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    },
    "energy_consumption": {
        "electricity": {"quantity": 1200000, "unit": "kwh", "renewable_percentage": 25},
        "natural_gas": {"quantity": 50000, "unit": "therms"}
    },
    "options": {
        "include_intensity_metrics": True,
        "market_based_accounting": True
    }
}

response = requests.post(
    "https://api.greenlang.io/v1/calculate/building",
    headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
    json=data
)

result = response.json()["data"]
print(f"Total Emissions (Market-Based): {result['emissions']['total']['market_based']} tCO2e")
print(f"Emissions Intensity: {result['intensity_metrics']['emissions_per_sqft']['market_based']} kgCO2e/sqft")
```

---

## EUDR (EU Deforestation Regulation)

Verify compliance with EU Deforestation Regulation.

### POST /v1/calculate/eudr

**Description:**

Validate commodity sourcing against EU Deforestation Regulation requirements. Perform geolocation verification, deforestation risk assessment, and generate due diligence statements.

**Request Body:**

```json
{
  "operator": {
    "name": "Acme Foods Ltd",
    "eori_number": "GB123456789000",
    "country": "GB"
  },
  "commodities": [
    {
      "id": "shipment_001",
      "type": "cocoa",
      "product_description": "Cocoa beans, raw",
      "hs_code": "1801.00",
      "quantity": 50,
      "unit": "tonnes",
      "country_of_production": "GH",
      "geolocation": {
        "type": "polygon",
        "coordinates": [
          [[-1.5, 6.2], [-1.4, 6.2], [-1.4, 6.3], [-1.5, 6.3], [-1.5, 6.2]]
        ]
      },
      "production_date": "2024-10-15",
      "supplier": {
        "name": "Ghana Cocoa Cooperative",
        "registration_number": "GH-COC-12345"
      }
    },
    {
      "id": "shipment_002",
      "type": "palm_oil",
      "product_description": "Crude palm oil",
      "hs_code": "1511.10",
      "quantity": 100,
      "unit": "tonnes",
      "country_of_production": "ID",
      "geolocation": {
        "type": "point",
        "coordinates": [101.5, 0.5]
      },
      "production_date": "2024-11-01",
      "certifications": ["RSPO", "ISCC"]
    }
  ],
  "options": {
    "deforestation_cutoff_date": "2020-12-31",
    "include_satellite_analysis": true,
    "generate_dds": true
  }
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `operator.name` | string | Yes | Operator/importer name |
| `operator.eori_number` | string | Yes | EORI number |
| `commodities` | array | Yes | Array of commodity shipments |
| `commodities[].type` | string | Yes | Commodity type: `cattle`, `cocoa`, `coffee`, `palm_oil`, `rubber`, `soya`, `wood` |
| `commodities[].hs_code` | string | Yes | Harmonized System code |
| `commodities[].geolocation` | object | Yes | Production location (GeoJSON) |
| `commodities[].production_date` | string | Yes | Date of production (ISO 8601) |
| `options.deforestation_cutoff_date` | string | No | Cutoff date (default: 2020-12-31) |
| `options.include_satellite_analysis` | boolean | No | Include satellite forest cover analysis |
| `options.generate_dds` | boolean | No | Generate due diligence statement |

**Response (200 OK):**

```json
{
  "data": {
    "calculation_id": "calc_eudr_jkl012",
    "operator": {
      "name": "Acme Foods Ltd",
      "eori_number": "GB123456789000"
    },
    "assessment_date": "2025-01-15",
    "summary": {
      "total_commodities": 2,
      "compliant": 1,
      "non_compliant": 0,
      "requires_review": 1,
      "overall_risk_level": "standard"
    },
    "commodities": [
      {
        "id": "shipment_001",
        "type": "cocoa",
        "quantity": 50,
        "unit": "tonnes",
        "country_of_production": "GH",
        "compliance_status": "compliant",
        "risk_level": "low",
        "geolocation_validation": {
          "status": "valid",
          "area_hectares": 125.5,
          "within_country_boundary": true
        },
        "deforestation_analysis": {
          "status": "clear",
          "forest_cover_2020": 0.0,
          "forest_cover_current": 0.0,
          "deforestation_detected": false,
          "analysis_source": "Global Forest Watch + Sentinel-2",
          "confidence": 0.95
        },
        "legality_check": {
          "status": "verified",
          "documents_verified": ["phytosanitary_certificate", "export_permit"]
        }
      },
      {
        "id": "shipment_002",
        "type": "palm_oil",
        "quantity": 100,
        "unit": "tonnes",
        "country_of_production": "ID",
        "compliance_status": "requires_review",
        "risk_level": "standard",
        "geolocation_validation": {
          "status": "valid",
          "point_coordinates": [101.5, 0.5],
          "within_country_boundary": true
        },
        "deforestation_analysis": {
          "status": "review_required",
          "forest_cover_2020": 45.2,
          "forest_cover_current": 42.8,
          "deforestation_detected": true,
          "deforestation_area_hectares": 2.4,
          "analysis_source": "Global Forest Watch + Sentinel-2",
          "confidence": 0.88,
          "note": "Minor forest cover change detected. Manual review recommended to confirm production plot boundaries."
        },
        "certifications": {
          "rspo": {
            "valid": true,
            "certificate_number": "RSPO-123456",
            "expiry": "2025-12-31"
          },
          "iscc": {
            "valid": true,
            "certificate_number": "ISCC-EU-123456"
          }
        }
      }
    ],
    "due_diligence_statement": {
      "available": true,
      "reference_number": "DDS-2025-001234",
      "download_url": "/v1/downloads/dds_jkl012.pdf",
      "expires_at": "2025-01-16T10:30:00Z",
      "status": "draft",
      "note": "Statement pending review for shipment_002"
    },
    "country_risk_classification": {
      "GH": {
        "country": "Ghana",
        "risk_level": "standard",
        "source": "EU EUDR Country Benchmarking"
      },
      "ID": {
        "country": "Indonesia",
        "risk_level": "high",
        "source": "EU EUDR Country Benchmarking"
      }
    },
    "methodology": {
      "regulation": "EU Regulation 2023/1115 (EUDR)",
      "deforestation_definition": "FAO Forest Definition",
      "satellite_data_sources": ["Global Forest Watch", "Sentinel-2", "Landsat"],
      "version": "2024.2"
    },
    "created_at": "2025-01-15T10:30:00Z"
  },
  "meta": {
    "request_id": "req_eudr456def",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

**Code Examples:**

**Python:**

```python
import requests

data = {
    "operator": {
        "name": "Acme Foods Ltd",
        "eori_number": "GB123456789000",
        "country": "GB"
    },
    "commodities": [
        {
            "id": "cocoa_shipment_001",
            "type": "cocoa",
            "product_description": "Cocoa beans, raw",
            "hs_code": "1801.00",
            "quantity": 50,
            "unit": "tonnes",
            "country_of_production": "GH",
            "geolocation": {
                "type": "polygon",
                "coordinates": [[[-1.5, 6.2], [-1.4, 6.2], [-1.4, 6.3], [-1.5, 6.3], [-1.5, 6.2]]]
            },
            "production_date": "2024-10-15"
        }
    ],
    "options": {
        "include_satellite_analysis": True,
        "generate_dds": True
    }
}

response = requests.post(
    "https://api.greenlang.io/v1/calculate/eudr",
    headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
    json=data
)

result = response.json()["data"]
print(f"Overall Risk Level: {result['summary']['overall_risk_level']}")
print(f"Compliant Shipments: {result['summary']['compliant']}/{result['summary']['total_commodities']}")
```

---

## Scope 3 Emissions

Calculate value chain (Scope 3) emissions.

### POST /v1/calculate/scope3

**Description:**

Calculate Scope 3 emissions across all 15 GHG Protocol categories including purchased goods, transportation, business travel, employee commuting, and investments.

**Request Body:**

```json
{
  "reporting_year": 2024,
  "organization": {
    "name": "Acme Corporation",
    "industry": "manufacturing",
    "employees": 5000
  },
  "categories": {
    "cat1_purchased_goods": {
      "method": "spend_based",
      "data": [
        {"category": "raw_materials", "spend_usd": 50000000, "sector": "metals"},
        {"category": "packaging", "spend_usd": 5000000, "sector": "plastics"},
        {"category": "office_supplies", "spend_usd": 500000, "sector": "paper"}
      ]
    },
    "cat4_upstream_transport": {
      "method": "distance_based",
      "data": [
        {"mode": "road_freight", "distance_km": 500000, "weight_tonnes": 10000},
        {"mode": "ocean_freight", "distance_km": 15000000, "weight_tonnes": 50000},
        {"mode": "air_freight", "distance_km": 200000, "weight_tonnes": 500}
      ]
    },
    "cat6_business_travel": {
      "method": "distance_based",
      "data": [
        {"mode": "short_haul_flight", "distance_km": 500000, "class": "economy"},
        {"mode": "long_haul_flight", "distance_km": 2000000, "class": "business"},
        {"mode": "rail", "distance_km": 100000},
        {"mode": "car_rental", "distance_km": 200000}
      ]
    },
    "cat7_employee_commuting": {
      "method": "average_data",
      "data": {
        "employees": 5000,
        "work_days_per_year": 230,
        "remote_work_percentage": 40,
        "commute_profile": {
          "car": 50,
          "public_transit": 30,
          "cycling_walking": 15,
          "carpool": 5
        },
        "average_distance_km": 25
      }
    }
  },
  "options": {
    "include_breakdown": true,
    "emission_factor_source": "defra_2024"
  }
}
```

**Scope 3 Categories:**

| Category | Name | Methods Supported |
|----------|------|-------------------|
| 1 | Purchased Goods & Services | spend_based, hybrid, supplier_specific |
| 2 | Capital Goods | spend_based, supplier_specific |
| 3 | Fuel & Energy Activities | average_data |
| 4 | Upstream Transportation | distance_based, spend_based |
| 5 | Waste Generated | waste_type_based |
| 6 | Business Travel | distance_based, spend_based |
| 7 | Employee Commuting | distance_based, average_data |
| 8 | Upstream Leased Assets | asset_specific |
| 9 | Downstream Transportation | distance_based |
| 10 | Processing of Sold Products | average_data |
| 11 | Use of Sold Products | product_specific |
| 12 | End-of-Life Treatment | average_data |
| 13 | Downstream Leased Assets | asset_specific |
| 14 | Franchises | average_data |
| 15 | Investments | investment_based |

**Response (200 OK):**

```json
{
  "data": {
    "calculation_id": "calc_scope3_mno345",
    "reporting_year": 2024,
    "organization": {
      "name": "Acme Corporation",
      "industry": "manufacturing"
    },
    "summary": {
      "total_scope3_emissions": 125420.5,
      "unit": "tCO2e",
      "categories_calculated": 4,
      "data_quality_score": 72
    },
    "emissions_by_category": {
      "cat1_purchased_goods": {
        "emissions": 85000.0,
        "unit": "tCO2e",
        "percentage_of_total": 67.8,
        "method": "spend_based",
        "data_quality": "medium"
      },
      "cat4_upstream_transport": {
        "emissions": 28500.5,
        "unit": "tCO2e",
        "percentage_of_total": 22.7,
        "method": "distance_based",
        "data_quality": "high"
      },
      "cat6_business_travel": {
        "emissions": 8420.0,
        "unit": "tCO2e",
        "percentage_of_total": 6.7,
        "method": "distance_based",
        "data_quality": "high"
      },
      "cat7_employee_commuting": {
        "emissions": 3500.0,
        "unit": "tCO2e",
        "percentage_of_total": 2.8,
        "method": "average_data",
        "data_quality": "medium"
      }
    },
    "breakdown": {
      "cat1_purchased_goods": {
        "details": [
          {"category": "raw_materials", "spend_usd": 50000000, "emission_factor": 0.0015, "emissions": 75000.0},
          {"category": "packaging", "spend_usd": 5000000, "emission_factor": 0.0018, "emissions": 9000.0},
          {"category": "office_supplies", "spend_usd": 500000, "emission_factor": 0.002, "emissions": 1000.0}
        ]
      },
      "cat4_upstream_transport": {
        "details": [
          {"mode": "road_freight", "tonne_km": 5000000000, "emission_factor": 0.0001, "emissions": 5000.0},
          {"mode": "ocean_freight", "tonne_km": 750000000000, "emission_factor": 0.00003, "emissions": 22500.0},
          {"mode": "air_freight", "tonne_km": 100000000, "emission_factor": 0.01, "emissions": 1000.5}
        ]
      }
    },
    "recommendations": [
      {
        "category": "cat1_purchased_goods",
        "recommendation": "Consider supplier engagement to obtain primary emissions data",
        "potential_reduction": "10-30% data quality improvement"
      },
      {
        "category": "cat6_business_travel",
        "recommendation": "Implement video conferencing policy to reduce air travel",
        "potential_reduction": "20-40% emissions reduction"
      }
    ],
    "methodology": {
      "name": "GHG Protocol Corporate Value Chain (Scope 3) Standard",
      "emission_factors": "DEFRA 2024 Conversion Factors",
      "version": "2024"
    },
    "created_at": "2025-01-15T10:30:00Z"
  },
  "meta": {
    "request_id": "req_scope3789ghi",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

---

## PCF (Product Carbon Footprint)

Calculate product-level carbon footprint.

### POST /v1/calculate/pcf

**Description:**

Calculate cradle-to-gate or cradle-to-grave carbon footprint for individual products following ISO 14067 and GHG Protocol Product Standard.

**Request Body:**

```json
{
  "product": {
    "name": "Organic Cotton T-Shirt",
    "sku": "TSHIRT-ORG-001",
    "category": "apparel",
    "functional_unit": {
      "description": "One medium-sized t-shirt (200g)",
      "quantity": 1,
      "unit": "piece"
    }
  },
  "boundary": "cradle_to_gate",
  "lifecycle_stages": {
    "raw_materials": {
      "items": [
        {
          "material": "organic_cotton",
          "quantity": 0.25,
          "unit": "kg",
          "origin_country": "IN",
          "emission_factor_source": "ecoinvent"
        },
        {
          "material": "polyester_thread",
          "quantity": 0.01,
          "unit": "kg"
        }
      ]
    },
    "manufacturing": {
      "location": {
        "country": "BD",
        "grid_region": "bangladesh_national"
      },
      "processes": [
        {
          "process": "spinning",
          "electricity_kwh": 0.5
        },
        {
          "process": "weaving",
          "electricity_kwh": 0.3
        },
        {
          "process": "dyeing",
          "electricity_kwh": 0.8,
          "water_liters": 50,
          "chemicals_kg": 0.05
        },
        {
          "process": "cutting_sewing",
          "electricity_kwh": 0.2
        }
      ]
    },
    "packaging": {
      "items": [
        {"material": "recycled_cardboard", "quantity": 0.05, "unit": "kg"},
        {"material": "ldpe_bag", "quantity": 0.01, "unit": "kg"}
      ]
    },
    "distribution": {
      "segments": [
        {"mode": "ocean_freight", "distance_km": 15000, "weight_kg": 0.3},
        {"mode": "road_freight", "distance_km": 500, "weight_kg": 0.3}
      ]
    }
  },
  "options": {
    "include_uncertainty": true,
    "include_breakdown": true,
    "data_quality_assessment": true
  }
}
```

**PCF Boundary Options:**

| Boundary | Description |
|----------|-------------|
| `cradle_to_gate` | Raw materials through manufacturing |
| `cradle_to_grave` | Full lifecycle including use and end-of-life |
| `gate_to_gate` | Manufacturing processes only |
| `cradle_to_cradle` | Circular economy, including recycling |

**Response (200 OK):**

```json
{
  "data": {
    "calculation_id": "calc_pcf_pqr678",
    "product": {
      "name": "Organic Cotton T-Shirt",
      "sku": "TSHIRT-ORG-001",
      "functional_unit": "One medium-sized t-shirt (200g)"
    },
    "boundary": "cradle_to_gate",
    "pcf_result": {
      "total": 5.85,
      "unit": "kgCO2e",
      "uncertainty": {
        "lower_bound": 4.68,
        "upper_bound": 7.31,
        "confidence_level": 0.95
      }
    },
    "breakdown_by_stage": {
      "raw_materials": {
        "emissions": 2.10,
        "unit": "kgCO2e",
        "percentage": 35.9,
        "details": [
          {"item": "organic_cotton", "quantity": "0.25 kg", "emissions": 2.00},
          {"item": "polyester_thread", "quantity": "0.01 kg", "emissions": 0.10}
        ]
      },
      "manufacturing": {
        "emissions": 2.85,
        "unit": "kgCO2e",
        "percentage": 48.7,
        "details": [
          {"process": "spinning", "electricity": "0.5 kWh", "emissions": 0.35},
          {"process": "weaving", "electricity": "0.3 kWh", "emissions": 0.21},
          {"process": "dyeing", "electricity": "0.8 kWh", "water": "50 L", "emissions": 1.89},
          {"process": "cutting_sewing", "electricity": "0.2 kWh", "emissions": 0.40}
        ]
      },
      "packaging": {
        "emissions": 0.15,
        "unit": "kgCO2e",
        "percentage": 2.6,
        "details": [
          {"item": "recycled_cardboard", "quantity": "0.05 kg", "emissions": 0.05},
          {"item": "ldpe_bag", "quantity": "0.01 kg", "emissions": 0.10}
        ]
      },
      "distribution": {
        "emissions": 0.75,
        "unit": "kgCO2e",
        "percentage": 12.8,
        "details": [
          {"segment": "ocean_freight", "distance": "15000 km", "emissions": 0.45},
          {"segment": "road_freight", "distance": "500 km", "emissions": 0.30}
        ]
      }
    },
    "hotspot_analysis": {
      "top_contributors": [
        {"source": "Dyeing process", "emissions": 1.89, "percentage": 32.3},
        {"source": "Organic cotton cultivation", "emissions": 2.00, "percentage": 34.2},
        {"source": "Ocean freight", "emissions": 0.45, "percentage": 7.7}
      ],
      "recommendations": [
        "Switch to renewable energy in manufacturing to reduce dyeing emissions by up to 60%",
        "Consider nearshoring to reduce distribution emissions"
      ]
    },
    "data_quality": {
      "overall_score": 3.2,
      "scale": "1 (lowest) to 5 (highest)",
      "by_stage": {
        "raw_materials": {"score": 4, "source": "ecoinvent database"},
        "manufacturing": {"score": 3, "source": "industry averages"},
        "packaging": {"score": 3, "source": "industry averages"},
        "distribution": {"score": 3, "source": "modeled data"}
      }
    },
    "comparison": {
      "industry_average": 8.50,
      "difference_percentage": -31.2,
      "benchmark_source": "HIGG MSI 2024"
    },
    "certifications_applicable": [
      {"name": "Climate Neutral", "status": "eligible", "offset_required": "5.85 kgCO2e"},
      {"name": "Carbon Trust Footprint Label", "status": "eligible"}
    ],
    "methodology": {
      "standard": "ISO 14067:2018",
      "secondary_standard": "GHG Protocol Product Standard",
      "lca_database": "ecoinvent 3.9",
      "gwp_source": "IPCC AR6",
      "version": "2024.1"
    },
    "created_at": "2025-01-15T10:30:00Z"
  },
  "meta": {
    "request_id": "req_pcf123xyz",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

**Code Examples:**

**Python:**

```python
import requests

data = {
    "product": {
        "name": "Organic Cotton T-Shirt",
        "sku": "TSHIRT-ORG-001",
        "category": "apparel",
        "functional_unit": {
            "description": "One medium-sized t-shirt (200g)",
            "quantity": 1,
            "unit": "piece"
        }
    },
    "boundary": "cradle_to_gate",
    "lifecycle_stages": {
        "raw_materials": {
            "items": [
                {"material": "organic_cotton", "quantity": 0.25, "unit": "kg", "origin_country": "IN"}
            ]
        },
        "manufacturing": {
            "location": {"country": "BD"},
            "processes": [
                {"process": "dyeing", "electricity_kwh": 0.8, "water_liters": 50}
            ]
        }
    },
    "options": {
        "include_breakdown": True,
        "include_uncertainty": True
    }
}

response = requests.post(
    "https://api.greenlang.io/v1/calculate/pcf",
    headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
    json=data
)

result = response.json()["data"]
print(f"Product Carbon Footprint: {result['pcf_result']['total']} {result['pcf_result']['unit']}")
print(f"Top Hotspot: {result['hotspot_analysis']['top_contributors'][0]['source']}")
```

---

## Batch Calculations

Submit multiple calculations in a single request.

### POST /v1/calculate/batch

**Request Body:**

```json
{
  "calculations": [
    {
      "id": "calc_001",
      "type": "fuel",
      "inputs": {"fuel_type": "diesel", "quantity": 1000, "unit": "liters"}
    },
    {
      "id": "calc_002",
      "type": "fuel",
      "inputs": {"fuel_type": "natural_gas", "quantity": 500, "unit": "cubic_meters"}
    },
    {
      "id": "calc_003",
      "type": "building",
      "inputs": {
        "building": {"type": "office", "location": {"country": "US", "state": "NY"}},
        "period": {"start_date": "2024-01-01", "end_date": "2024-12-31"},
        "energy_consumption": {"electricity": {"quantity": 500000, "unit": "kwh"}}
      }
    }
  ],
  "options": {
    "continue_on_error": true
  }
}
```

**Response (200 OK):**

```json
{
  "data": {
    "batch_id": "batch_abc123",
    "status": "completed",
    "summary": {
      "total": 3,
      "successful": 3,
      "failed": 0
    },
    "results": [
      {"id": "calc_001", "status": "completed", "emissions": {"co2e": 2705.2, "unit": "kg"}},
      {"id": "calc_002", "status": "completed", "emissions": {"co2e": 990.5, "unit": "kg"}},
      {"id": "calc_003", "status": "completed", "emissions": {"co2e": 142.5, "unit": "tCO2e"}}
    ],
    "processing_time_ms": 250
  }
}
```

---

## Error Responses

All calculation endpoints return consistent error responses:

**Validation Error (422):**

```json
{
  "error": {
    "code": "validation_error",
    "message": "Input validation failed",
    "details": [
      {
        "field": "fuel_type",
        "message": "Invalid fuel type 'petrol'. Did you mean 'gasoline'?",
        "suggestion": "gasoline"
      }
    ]
  }
}
```

**Calculation Error (400):**

```json
{
  "error": {
    "code": "calculation_error",
    "message": "Unable to calculate emissions",
    "details": {
      "reason": "No emission factor available for specified fuel type and region combination",
      "fuel_type": "biofuel_blend",
      "region": "XX"
    }
  }
}
```

---

## Next Steps

- [Agents API](./agents.md) - Use pre-configured calculation agents
- [Quick Start Guide](../guides/quickstart.md) - Get started in 5 minutes
- [EUDR Compliance Guide](../guides/eudr_compliance.md) - EU Deforestation Regulation
- [CBAM Reporting Guide](../guides/cbam_reporting.md) - Carbon Border Adjustment Mechanism
