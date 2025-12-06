# Emissions Calculation APIs

The Emissions APIs provide comprehensive carbon and greenhouse gas emissions calculations for industrial process heat equipment. These APIs support EPA Method 19, GHG Protocol, and EU ETS methodologies.

---

## Table of Contents

1. [Calculate Carbon Emissions](#calculate-carbon-emissions)
2. [Calculate Scope 3 Emissions](#calculate-scope-3-emissions)
3. [Batch Emissions Calculation](#batch-emissions-calculation)
4. [Get Emission Factors](#get-emission-factors)
5. [Get Source Emissions](#get-source-emissions)
6. [Emissions Monitoring](#emissions-monitoring)
7. [Error Codes](#error-codes)

---

## Calculate Carbon Emissions

Calculate direct (Scope 1) carbon emissions from fuel combustion.

```http
POST /api/v1/emissions/carbon
```

### Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | Bearer token |
| `Content-Type` | Yes | `application/json` |
| `X-Request-ID` | Recommended | Unique request identifier |

### Request Body

```json
{
  "source_id": "BLR-001",
  "calculation_method": "EPA_PART_98",
  "fuel_data": {
    "fuel_type": "natural_gas",
    "fuel_consumption": 150.0,
    "fuel_unit": "MMBTU/hr",
    "fuel_hhv": 1020.0,
    "fuel_carbon_content_pct": null
  },
  "operating_conditions": {
    "load_pct": 85.0,
    "operating_hours": 8760,
    "stack_o2_pct": 3.2,
    "stack_temperature_f": 380
  },
  "reporting_period": {
    "start_date": "2025-01-01",
    "end_date": "2025-12-31"
  },
  "include_uncertainty": true,
  "metadata": {
    "facility_id": "FACILITY-001",
    "unit_id": "UNIT-001"
  }
}
```

### Request Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_id` | string | Yes | Emission source identifier |
| `calculation_method` | string | No | Calculation methodology: `EPA_PART_98`, `GHG_PROTOCOL`, `EU_ETS`, `ISO_14064`. Default: `EPA_PART_98` |
| `fuel_data` | object | Yes | Fuel data specification |
| `operating_conditions` | object | No | Operating conditions |
| `reporting_period` | object | No | Reporting period for annualized emissions |
| `include_uncertainty` | boolean | No | Include uncertainty analysis. Default: true |
| `metadata` | object | No | Additional metadata |

### Fuel Data Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `fuel_type` | string | Yes | Fuel type: `natural_gas`, `fuel_oil_no2`, `fuel_oil_no6`, `diesel`, `propane`, `coal_bituminous`, `coal_subbituminous`, `biomass_wood`, `biogas` |
| `fuel_consumption` | float | Yes | Fuel consumption rate or total |
| `fuel_unit` | string | Yes | Fuel unit: `MMBTU/hr`, `MMBTU`, `therms`, `MCF`, `gallons`, `kg`, `tonnes`, `MWh` |
| `fuel_hhv` | float | No | Higher heating value (BTU/unit). Uses default if not provided |
| `fuel_carbon_content_pct` | float | No | Carbon content percentage. Uses default if not provided |

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "source_id": "BLR-001",
    "calculation_id": "calc_a1b2c3d4",
    "timestamp": "2025-12-06T10:30:00Z",
    "emissions": {
      "co2": {
        "value": 8567.2,
        "unit": "kg/hr",
        "annual_value": 75048792.0,
        "annual_unit": "kg/yr",
        "annual_tonnes": 75048.8
      },
      "ch4": {
        "value": 0.158,
        "unit": "kg/hr",
        "co2e": 4.43,
        "co2e_unit": "kg/hr"
      },
      "n2o": {
        "value": 0.016,
        "unit": "kg/hr",
        "co2e": 4.77,
        "co2e_unit": "kg/hr"
      },
      "total_co2e": {
        "value": 8576.4,
        "unit": "kg/hr",
        "annual_value": 75129264.0,
        "annual_unit": "kg/yr",
        "annual_tonnes": 75129.3
      }
    },
    "emission_rates": {
      "co2_per_mmbtu": {
        "value": 53.06,
        "unit": "kg/MMBTU"
      },
      "co2_per_mwh": {
        "value": 181.2,
        "unit": "kg/MWh"
      }
    },
    "emission_factors_used": {
      "co2_factor": {
        "value": 53.06,
        "unit": "kg/MMBTU",
        "source": "40 CFR Part 98 Table C-1"
      },
      "ch4_factor": {
        "value": 0.001,
        "unit": "kg/MMBTU",
        "source": "40 CFR Part 98 Table C-2"
      },
      "n2o_factor": {
        "value": 0.0001,
        "unit": "kg/MMBTU",
        "source": "40 CFR Part 98 Table C-2"
      }
    },
    "uncertainty": {
      "co2_lower_bound_pct": -2.5,
      "co2_upper_bound_pct": 2.5,
      "confidence_level": 0.95,
      "primary_uncertainty_sources": [
        "Fuel flow measurement: +/- 1.5%",
        "Emission factor uncertainty: +/- 1.8%"
      ]
    },
    "calculation_details": {
      "method": "EPA_PART_98",
      "formula": "CO2 = Fuel x EF",
      "formula_reference": "40 CFR Part 98 Subpart C",
      "heat_input_mmbtu_hr": 161.5,
      "fuel_hhv_used": 1020.0
    },
    "provenance_hash": "sha256:a1b2c3d4e5f6..."
  },
  "metadata": {
    "request_id": "req_xyz789",
    "processing_time_ms": 45
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

### Example (cURL)

```bash
curl -X POST "https://api.greenlang.io/v1/emissions/carbon" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "BLR-001",
    "calculation_method": "EPA_PART_98",
    "fuel_data": {
      "fuel_type": "natural_gas",
      "fuel_consumption": 150.0,
      "fuel_unit": "MMBTU/hr"
    },
    "include_uncertainty": true
  }'
```

### Example (Python)

```python
import requests

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

payload = {
    "source_id": "BLR-001",
    "calculation_method": "EPA_PART_98",
    "fuel_data": {
        "fuel_type": "natural_gas",
        "fuel_consumption": 150.0,
        "fuel_unit": "MMBTU/hr"
    },
    "operating_conditions": {
        "load_pct": 85.0,
        "operating_hours": 8760
    },
    "include_uncertainty": True
}

response = requests.post(
    "https://api.greenlang.io/v1/emissions/carbon",
    headers=headers,
    json=payload
)

result = response.json()
emissions = result["data"]["emissions"]

print(f"CO2 Emissions: {emissions['co2']['value']:.1f} kg/hr")
print(f"Annual CO2e: {emissions['total_co2e']['annual_tonnes']:,.1f} tonnes/yr")
```

---

## Calculate Scope 3 Emissions

Calculate indirect (Scope 3) emissions across the value chain.

```http
POST /api/v1/emissions/scope3
```

### Request Body

```json
{
  "organization_id": "ORG-001",
  "reporting_year": 2025,
  "categories": {
    "purchased_goods": {
      "enabled": true,
      "data": [
        {
          "category": "raw_materials",
          "description": "Steel inputs",
          "spend_usd": 5000000,
          "emission_factor_source": "EPA_EEIO"
        },
        {
          "category": "chemicals",
          "description": "Process chemicals",
          "quantity": 1000,
          "quantity_unit": "tonnes",
          "emission_factor_kg_co2e": 2.5
        }
      ]
    },
    "fuel_and_energy": {
      "enabled": true,
      "data": [
        {
          "energy_type": "electricity",
          "consumption": 50000,
          "consumption_unit": "MWh",
          "grid_region": "US_RFCE"
        }
      ]
    },
    "transportation": {
      "enabled": true,
      "upstream": {
        "mode": "truck",
        "distance_km": 500,
        "weight_tonnes": 10000
      },
      "downstream": {
        "mode": "rail",
        "distance_km": 1500,
        "weight_tonnes": 8000
      }
    },
    "waste": {
      "enabled": true,
      "data": [
        {
          "waste_type": "industrial_waste",
          "quantity_tonnes": 500,
          "treatment": "landfill"
        }
      ]
    },
    "business_travel": {
      "enabled": false
    },
    "employee_commuting": {
      "enabled": false
    }
  },
  "calculation_approach": "hybrid",
  "include_uncertainty": true
}
```

### Scope 3 Categories

| Category | Description |
|----------|-------------|
| `purchased_goods` | Category 1: Purchased goods and services |
| `capital_goods` | Category 2: Capital goods |
| `fuel_and_energy` | Category 3: Fuel- and energy-related activities |
| `transportation` | Category 4 & 9: Upstream/downstream transportation |
| `waste` | Category 5: Waste generated in operations |
| `business_travel` | Category 6: Business travel |
| `employee_commuting` | Category 7: Employee commuting |
| `leased_assets` | Category 8 & 13: Upstream/downstream leased assets |
| `processing` | Category 10: Processing of sold products |
| `use_of_products` | Category 11: Use of sold products |
| `end_of_life` | Category 12: End-of-life treatment |
| `franchises` | Category 14: Franchises |
| `investments` | Category 15: Investments |

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "organization_id": "ORG-001",
    "reporting_year": 2025,
    "calculation_id": "scope3_calc_xyz789",
    "emissions_by_category": {
      "category_1_purchased_goods": {
        "co2e_tonnes": 12500,
        "percentage_of_total": 35.2,
        "data_quality_score": 0.72,
        "calculation_approach": "spend_based"
      },
      "category_3_fuel_and_energy": {
        "co2e_tonnes": 8200,
        "percentage_of_total": 23.1,
        "data_quality_score": 0.85,
        "calculation_approach": "activity_based"
      },
      "category_4_upstream_transportation": {
        "co2e_tonnes": 3400,
        "percentage_of_total": 9.6,
        "data_quality_score": 0.68,
        "calculation_approach": "distance_based"
      },
      "category_5_waste": {
        "co2e_tonnes": 1200,
        "percentage_of_total": 3.4,
        "data_quality_score": 0.75,
        "calculation_approach": "waste_type_specific"
      },
      "category_9_downstream_transportation": {
        "co2e_tonnes": 10200,
        "percentage_of_total": 28.7,
        "data_quality_score": 0.65,
        "calculation_approach": "distance_based"
      }
    },
    "total_scope3_emissions": {
      "co2e_tonnes": 35500,
      "uncertainty_lower_tonnes": 31950,
      "uncertainty_upper_tonnes": 39050,
      "confidence_level": 0.90
    },
    "data_quality_summary": {
      "overall_score": 0.73,
      "coverage_pct": 85,
      "categories_included": 5,
      "categories_excluded": 10,
      "recommendations": [
        "Improve data collection for upstream transportation",
        "Consider including business travel emissions",
        "Obtain supplier-specific emission factors for key materials"
      ]
    },
    "methodology": {
      "standard": "GHG_PROTOCOL_SCOPE3",
      "reference": "GHG Protocol Corporate Value Chain (Scope 3) Standard",
      "version": "2011"
    },
    "provenance_hash": "sha256:b2c3d4e5f6g7..."
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

---

## Batch Emissions Calculation

Calculate emissions for multiple sources in a single request.

```http
POST /api/v1/emissions/batch
```

### Request Body

```json
{
  "batch_id": "batch_daily_2025-12-06",
  "calculation_method": "EPA_PART_98",
  "sources": [
    {
      "source_id": "BLR-001",
      "fuel_data": {
        "fuel_type": "natural_gas",
        "fuel_consumption": 150.0,
        "fuel_unit": "MMBTU/hr"
      }
    },
    {
      "source_id": "BLR-002",
      "fuel_data": {
        "fuel_type": "natural_gas",
        "fuel_consumption": 200.0,
        "fuel_unit": "MMBTU/hr"
      }
    },
    {
      "source_id": "FUR-001",
      "fuel_data": {
        "fuel_type": "fuel_oil_no2",
        "fuel_consumption": 500.0,
        "fuel_unit": "gallons/hr"
      }
    }
  ],
  "aggregation": {
    "enabled": true,
    "group_by": ["facility_id", "fuel_type"]
  },
  "include_uncertainty": true
}
```

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "batch_id": "batch_daily_2025-12-06",
    "calculation_id": "batch_calc_abc123",
    "processed_at": "2025-12-06T10:30:00Z",
    "sources_processed": 3,
    "sources_failed": 0,
    "results": [
      {
        "source_id": "BLR-001",
        "status": "success",
        "emissions": {
          "co2_kg_hr": 8567.2,
          "co2e_kg_hr": 8576.4
        }
      },
      {
        "source_id": "BLR-002",
        "status": "success",
        "emissions": {
          "co2_kg_hr": 11422.9,
          "co2e_kg_hr": 11435.2
        }
      },
      {
        "source_id": "FUR-001",
        "status": "success",
        "emissions": {
          "co2_kg_hr": 5234.1,
          "co2e_kg_hr": 5241.8
        }
      }
    ],
    "aggregation": {
      "total_co2e_kg_hr": 25253.4,
      "by_fuel_type": {
        "natural_gas": {
          "sources": 2,
          "co2e_kg_hr": 20011.6
        },
        "fuel_oil_no2": {
          "sources": 1,
          "co2e_kg_hr": 5241.8
        }
      }
    },
    "provenance_hash": "sha256:c3d4e5f6g7h8..."
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

### Example (Python)

```python
import requests

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

# Prepare batch data from operational sources
sources = [
    {
        "source_id": f"BLR-{i:03d}",
        "fuel_data": {
            "fuel_type": "natural_gas",
            "fuel_consumption": consumption,
            "fuel_unit": "MMBTU/hr"
        }
    }
    for i, consumption in enumerate([150, 200, 180, 120, 250], start=1)
]

payload = {
    "batch_id": f"batch_{datetime.now().strftime('%Y-%m-%d')}",
    "calculation_method": "EPA_PART_98",
    "sources": sources,
    "aggregation": {
        "enabled": True,
        "group_by": ["fuel_type"]
    }
}

response = requests.post(
    "https://api.greenlang.io/v1/emissions/batch",
    headers=headers,
    json=payload
)

result = response.json()
print(f"Total CO2e: {result['data']['aggregation']['total_co2e_kg_hr']:,.1f} kg/hr")
```

---

## Get Emission Factors

Retrieve emission factors for various fuel types and methodologies.

```http
GET /api/v1/emissions/factors
```

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `fuel_type` | string | No | Filter by fuel type |
| `methodology` | string | No | Filter by methodology: `EPA_PART_98`, `GHG_PROTOCOL`, `EU_ETS`, `IPCC` |
| `region` | string | No | Filter by region: `US`, `EU`, `GLOBAL` |
| `year` | integer | No | Emission factor year. Default: current year |

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "emission_factors": [
      {
        "fuel_type": "natural_gas",
        "methodology": "EPA_PART_98",
        "region": "US",
        "year": 2025,
        "factors": {
          "co2": {
            "value": 53.06,
            "unit": "kg/MMBTU",
            "uncertainty_pct": 1.8
          },
          "ch4": {
            "value": 0.001,
            "unit": "kg/MMBTU",
            "uncertainty_pct": 15.0
          },
          "n2o": {
            "value": 0.0001,
            "unit": "kg/MMBTU",
            "uncertainty_pct": 50.0
          }
        },
        "source": {
          "reference": "40 CFR Part 98 Table C-1",
          "publication_date": "2024-01-01",
          "last_updated": "2024-06-15"
        },
        "default_hhv": {
          "value": 1020,
          "unit": "BTU/SCF"
        }
      },
      {
        "fuel_type": "fuel_oil_no2",
        "methodology": "EPA_PART_98",
        "region": "US",
        "year": 2025,
        "factors": {
          "co2": {
            "value": 73.96,
            "unit": "kg/MMBTU",
            "uncertainty_pct": 2.0
          },
          "ch4": {
            "value": 0.003,
            "unit": "kg/MMBTU",
            "uncertainty_pct": 15.0
          },
          "n2o": {
            "value": 0.0006,
            "unit": "kg/MMBTU",
            "uncertainty_pct": 50.0
          }
        },
        "source": {
          "reference": "40 CFR Part 98 Table C-1",
          "publication_date": "2024-01-01"
        },
        "default_hhv": {
          "value": 137000,
          "unit": "BTU/gallon"
        }
      }
    ],
    "supported_fuel_types": [
      "natural_gas",
      "fuel_oil_no2",
      "fuel_oil_no6",
      "diesel",
      "propane",
      "butane",
      "coal_bituminous",
      "coal_subbituminous",
      "coal_anthracite",
      "coal_lignite",
      "biomass_wood",
      "biomass_agricultural",
      "biogas",
      "landfill_gas"
    ]
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

---

## Get Source Emissions

Retrieve historical emissions data for a specific source.

```http
GET /api/v1/emissions/sources/{source_id}
```

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `source_id` | string | Emission source identifier |

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `start_date` | string | No | Start date (ISO 8601) |
| `end_date` | string | No | End date (ISO 8601) |
| `interval` | string | No | Aggregation interval: `hourly`, `daily`, `monthly`, `annual`. Default: `daily` |
| `include_permit_limits` | boolean | No | Include permit limit comparison. Default: true |

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "source_id": "BLR-001",
    "source_name": "Boiler 001",
    "facility_id": "FACILITY-001",
    "period": {
      "start_date": "2025-11-01",
      "end_date": "2025-11-30",
      "interval": "daily"
    },
    "summary": {
      "total_co2e_tonnes": 6125.4,
      "avg_co2e_kg_hr": 8507.5,
      "max_co2e_kg_hr": 12450.2,
      "min_co2e_kg_hr": 2150.8,
      "operating_hours": 720,
      "capacity_factor_pct": 78.5
    },
    "permit_compliance": {
      "permit_limit_co2e_tonnes_yr": 85000,
      "ytd_emissions_tonnes": 68542,
      "remaining_allowance_tonnes": 16458,
      "projected_annual_tonnes": 82500,
      "compliance_status": "on_track",
      "exceedance_events": 0
    },
    "time_series": [
      {
        "date": "2025-11-01",
        "co2e_tonnes": 205.2,
        "co2e_kg_hr": 8550.0,
        "operating_hours": 24,
        "load_pct": 82.5
      },
      {
        "date": "2025-11-02",
        "co2e_tonnes": 198.7,
        "co2e_kg_hr": 8279.2,
        "operating_hours": 24,
        "load_pct": 79.8
      }
    ]
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

---

## Emissions Monitoring

Real-time emissions monitoring endpoint for continuous compliance tracking.

```http
POST /api/v1/emissions/monitor
```

### Request Body

```json
{
  "source_id": "BLR-001",
  "timestamp": "2025-12-06T10:30:00Z",
  "fuel_data": {
    "fuel_type": "natural_gas",
    "fuel_flow_rate": 150.0,
    "fuel_unit": "MMBTU/hr"
  },
  "stack_measurements": {
    "stack_o2_pct": 3.2,
    "stack_co_ppm": 45,
    "stack_nox_ppm": 25,
    "stack_so2_ppm": null,
    "stack_pm_mg_m3": null,
    "stack_temperature_f": 380,
    "stack_flow_rate_acfm": 15000
  },
  "operating_conditions": {
    "load_pct": 85.0,
    "operating_mode": "normal"
  },
  "permit_limits": {
    "co2_lb_hr": 25000,
    "nox_lb_hr": 5.0,
    "co_ppm": 400
  }
}
```

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "source_id": "BLR-001",
    "timestamp": "2025-12-06T10:30:00Z",
    "status": "compliant",
    "emissions": {
      "co2_lb_hr": 18875.4,
      "co2_kg_hr": 8563.2,
      "co2_ton_yr": 82675.8,
      "nox_lb_hr": 2.35,
      "co2_lb_mmbtu": 125.8,
      "nox_lb_mmbtu": 0.0157
    },
    "compliance": {
      "permit_limits": {
        "co2_lb_hr": 25000,
        "nox_lb_hr": 5.0,
        "co_ppm": 400
      },
      "current_values": {
        "co2_lb_hr": 18875.4,
        "nox_lb_hr": 2.35,
        "co_ppm": 45
      },
      "margin_pct": {
        "co2_lb_hr": 24.5,
        "nox_lb_hr": 53.0,
        "co_ppm": 88.75
      },
      "exceedances": [],
      "warnings": []
    },
    "predictions": {
      "exceedance_risk": 0.05,
      "predicted_exceedance_time_hr": null,
      "trend": "stable"
    },
    "provenance_hash": "sha256:d4e5f6g7h8i9..."
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

### Response with Warning

```json
{
  "success": true,
  "data": {
    "source_id": "BLR-001",
    "status": "warning",
    "compliance": {
      "exceedances": [],
      "warnings": [
        {
          "parameter": "co2_lb_hr",
          "message": "CO2 emissions at 92% of permit limit",
          "current_value": 23000,
          "limit": 25000,
          "margin_pct": 8.0
        },
        {
          "parameter": "trend",
          "message": "Exceedance predicted in ~2.5 hours based on trend"
        }
      ]
    },
    "predictions": {
      "exceedance_risk": 0.72,
      "predicted_exceedance_time_hr": 2.5,
      "trend": "increasing"
    }
  }
}
```

---

## Error Codes

| Code | HTTP Status | Description | Resolution |
|------|-------------|-------------|------------|
| `INVALID_FUEL_TYPE` | 400 | Unsupported fuel type | Use supported fuel type from `/factors` |
| `INVALID_FUEL_UNIT` | 400 | Invalid fuel unit | Check unit compatibility |
| `MISSING_FUEL_DATA` | 400 | Required fuel data missing | Provide fuel_type and consumption |
| `EMISSION_FACTOR_NOT_FOUND` | 404 | No emission factor available | Check methodology and region |
| `SOURCE_NOT_FOUND` | 404 | Emission source not found | Verify source_id |
| `CALCULATION_ERROR` | 500 | Calculation failed | Check input values |
| `PERMIT_EXCEEDANCE` | 200 | Permit limit exceeded | Review emissions and take action |
| `DATA_QUALITY_LOW` | 200 | Low data quality score | Improve input data quality |

---

## Rate Limiting

Emissions endpoints have the following rate limits:

| Endpoint | Rate Limit |
|----------|------------|
| `POST /emissions/carbon` | 200/minute |
| `POST /emissions/scope3` | 50/minute |
| `POST /emissions/batch` | 20/minute |
| `GET /emissions/factors` | 300/minute |
| `GET /emissions/sources/{id}` | 200/minute |
| `POST /emissions/monitor` | 600/minute |

---

## Best Practices

1. **Use appropriate calculation methodology** for your regulatory requirements
2. **Validate fuel types and units** before submitting
3. **Include uncertainty analysis** for compliance reporting
4. **Use batch endpoint** for multiple sources to reduce API calls
5. **Store provenance hashes** for audit trail verification
6. **Set up monitoring webhooks** for exceedance alerts
7. **Cache emission factors** to reduce lookups

---

## See Also

- [Main API Reference](../process_heat_api_reference.md)
- [Orchestrator API](orchestrator.md)
- [Compliance API](compliance.md)
- [Common Data Models](../schemas/common_models.md)
