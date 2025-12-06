# Compliance APIs

The Compliance APIs provide endpoints for generating regulatory compliance reports, calculating obligations, and tracking regulatory deadlines. These APIs support CSRD (Corporate Sustainability Reporting Directive), CBAM (Carbon Border Adjustment Mechanism), and EUDR (EU Deforestation Regulation) compliance.

---

## Table of Contents

1. [CSRD Reporting](#csrd-reporting)
2. [CBAM Calculations](#cbam-calculations)
3. [EUDR Compliance](#eudr-compliance)
4. [Report Management](#report-management)
5. [Regulatory Deadlines](#regulatory-deadlines)
6. [Audit Trail](#audit-trail)
7. [Error Codes](#error-codes)

---

## CSRD Reporting

Generate Corporate Sustainability Reporting Directive (CSRD) compliant reports.

### Generate CSRD Report

```http
POST /api/v1/compliance/csrd
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
  "organization_id": "ORG-001",
  "reporting_year": 2025,
  "reporting_standards": ["ESRS_E1", "ESRS_E2", "ESRS_S1"],
  "report_type": "annual",
  "data_sources": {
    "emissions_data": {
      "source": "api",
      "endpoint": "/emissions/sources",
      "filters": {
        "facility_ids": ["FACILITY-001", "FACILITY-002"]
      }
    },
    "energy_data": {
      "source": "upload",
      "file_id": "file_abc123"
    },
    "workforce_data": {
      "source": "manual",
      "data": {
        "total_employees": 5000,
        "by_region": {
          "EU": 3500,
          "US": 1200,
          "APAC": 300
        }
      }
    }
  },
  "materiality_assessment": {
    "double_materiality": true,
    "climate_change": {
      "financial_impact": "high",
      "environmental_impact": "high"
    },
    "pollution": {
      "financial_impact": "medium",
      "environmental_impact": "high"
    }
  },
  "assurance_level": "limited",
  "output_format": "XBRL",
  "language": "en",
  "include_audit_trail": true,
  "metadata": {
    "prepared_by": "sustainability@example.com",
    "approved_by": "cfo@example.com"
  }
}
```

### Request Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `organization_id` | string | Yes | Organization identifier |
| `reporting_year` | integer | Yes | Reporting year |
| `reporting_standards` | array | Yes | ESRS standards to include |
| `report_type` | string | No | Report type: `annual`, `interim`, `amendment`. Default: `annual` |
| `data_sources` | object | Yes | Data source specifications |
| `materiality_assessment` | object | No | Double materiality assessment |
| `assurance_level` | string | No | Assurance level: `none`, `limited`, `reasonable`. Default: `limited` |
| `output_format` | string | No | Output format: `XBRL`, `PDF`, `JSON`, `HTML`. Default: `XBRL` |
| `language` | string | No | Report language (ISO 639-1). Default: `en` |
| `include_audit_trail` | boolean | No | Include full audit trail. Default: true |

### ESRS Standards

| Standard | Description |
|----------|-------------|
| `ESRS_E1` | Climate change |
| `ESRS_E2` | Pollution |
| `ESRS_E3` | Water and marine resources |
| `ESRS_E4` | Biodiversity and ecosystems |
| `ESRS_E5` | Resource use and circular economy |
| `ESRS_S1` | Own workforce |
| `ESRS_S2` | Workers in the value chain |
| `ESRS_S3` | Affected communities |
| `ESRS_S4` | Consumers and end-users |
| `ESRS_G1` | Business conduct |

### Response (202 Accepted)

```json
{
  "success": true,
  "data": {
    "report_id": "csrd_rpt_a1b2c3d4",
    "status": "processing",
    "organization_id": "ORG-001",
    "reporting_year": 2025,
    "reporting_standards": ["ESRS_E1", "ESRS_E2", "ESRS_S1"],
    "created_at": "2025-12-06T10:30:00Z",
    "estimated_completion_s": 300,
    "webhook_url": "https://api.greenlang.io/v1/webhooks/reports/csrd_rpt_a1b2c3d4"
  },
  "metadata": {
    "request_id": "req_xyz789"
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

### Response (200 OK - Completed)

```json
{
  "success": true,
  "data": {
    "report_id": "csrd_rpt_a1b2c3d4",
    "status": "completed",
    "organization_id": "ORG-001",
    "reporting_year": 2025,
    "report_summary": {
      "standards_covered": ["ESRS_E1", "ESRS_E2", "ESRS_S1"],
      "data_points_reported": 245,
      "data_quality_score": 0.87,
      "materiality_topics": 8,
      "targets_set": 12,
      "policies_disclosed": 15
    },
    "esrs_e1_climate": {
      "scope1_emissions_tonnes": 75049,
      "scope2_emissions_tonnes": 42300,
      "scope3_emissions_tonnes": 185000,
      "total_ghg_emissions_tonnes": 302349,
      "emissions_intensity": 125.5,
      "intensity_unit": "tCO2e/EUR million revenue",
      "reduction_target_pct": 45,
      "reduction_target_year": 2030,
      "sbti_aligned": true,
      "net_zero_target_year": 2050
    },
    "esrs_e2_pollution": {
      "pollutants_reported": [
        {
          "pollutant": "NOx",
          "value": 125.5,
          "unit": "tonnes/year",
          "trend": "decreasing"
        },
        {
          "pollutant": "SOx",
          "value": 45.2,
          "unit": "tonnes/year",
          "trend": "stable"
        }
      ]
    },
    "esrs_s1_workforce": {
      "total_employees": 5000,
      "gender_diversity_pct": 42,
      "training_hours_per_employee": 32,
      "health_safety_incidents": 12,
      "living_wage_compliance_pct": 100
    },
    "download_urls": {
      "xbrl": "https://api.greenlang.io/v1/reports/csrd_rpt_a1b2c3d4/download?format=xbrl",
      "pdf": "https://api.greenlang.io/v1/reports/csrd_rpt_a1b2c3d4/download?format=pdf",
      "json": "https://api.greenlang.io/v1/reports/csrd_rpt_a1b2c3d4/download?format=json"
    },
    "validation": {
      "status": "passed",
      "errors": [],
      "warnings": [
        {
          "code": "ESRS_E1_W001",
          "message": "Scope 3 category 15 (investments) not disclosed",
          "recommendation": "Consider disclosing investment portfolio emissions"
        }
      ]
    },
    "audit_trail": {
      "chain_id": "chain_xyz789",
      "merkle_root": "sha256:a1b2c3d4...",
      "record_count": 48
    },
    "provenance_hash": "sha256:e5f6g7h8i9j0..."
  },
  "timestamp": "2025-12-06T10:35:00Z"
}
```

### Example (Python)

```python
import requests
import time

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

# Submit CSRD report generation
payload = {
    "organization_id": "ORG-001",
    "reporting_year": 2025,
    "reporting_standards": ["ESRS_E1", "ESRS_E2", "ESRS_S1"],
    "data_sources": {
        "emissions_data": {
            "source": "api",
            "filters": {"facility_ids": ["FACILITY-001"]}
        }
    },
    "output_format": "XBRL",
    "assurance_level": "limited"
}

response = requests.post(
    "https://api.greenlang.io/v1/compliance/csrd",
    headers=headers,
    json=payload
)

result = response.json()
report_id = result["data"]["report_id"]

# Poll for completion
while True:
    status_response = requests.get(
        f"https://api.greenlang.io/v1/compliance/reports/{report_id}",
        headers=headers
    )
    status = status_response.json()["data"]["status"]

    if status == "completed":
        print("Report generated successfully!")
        break
    elif status == "failed":
        print(f"Report generation failed: {status_response.json()['data']['error']}")
        break

    time.sleep(10)

# Download the report
download_url = status_response.json()["data"]["download_urls"]["pdf"]
```

---

## CBAM Calculations

Calculate Carbon Border Adjustment Mechanism obligations for imported goods.

### Calculate CBAM Obligations

```http
POST /api/v1/compliance/cbam
```

### Request Body

```json
{
  "declaration_id": "CBAM-2025-Q4",
  "reporting_period": {
    "quarter": "Q4",
    "year": 2025
  },
  "declarant": {
    "organization_id": "ORG-001",
    "eori_number": "DE123456789000",
    "authorized_representative": "John Smith"
  },
  "imports": [
    {
      "product_id": "IMP-001",
      "cn_code": "7208",
      "product_description": "Hot-rolled steel coils",
      "country_of_origin": "CN",
      "quantity": 1000,
      "quantity_unit": "tonnes",
      "customs_value_eur": 850000,
      "installation_data": {
        "installation_id": "CN-STEEL-001",
        "installation_name": "Baosteel Facility 1",
        "direct_emissions_tco2e": 1850,
        "indirect_emissions_tco2e": 420,
        "production_quantity": 1000,
        "production_unit": "tonnes",
        "carbon_price_paid_eur": 12500,
        "carbon_price_type": "national_carbon_tax"
      }
    },
    {
      "product_id": "IMP-002",
      "cn_code": "2523",
      "product_description": "Portland cement",
      "country_of_origin": "TR",
      "quantity": 5000,
      "quantity_unit": "tonnes",
      "customs_value_eur": 450000,
      "installation_data": null,
      "use_default_values": true
    }
  ],
  "calculation_options": {
    "use_default_values_where_missing": true,
    "eu_ets_price_source": "official",
    "include_indirect_emissions": true
  }
}
```

### Request Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `declaration_id` | string | Yes | Unique declaration identifier |
| `reporting_period` | object | Yes | Quarter and year of reporting |
| `declarant` | object | Yes | Declarant information |
| `imports` | array | Yes | List of imported products |
| `calculation_options` | object | No | Calculation options |

### CBAM Product Categories (CN Codes)

| CN Code Range | Product Category |
|---------------|-----------------|
| 2523 | Cement |
| 2804 40 00 | Hydrogen |
| 2814 | Ammonia |
| 2834 21 00 | Nitric acid |
| 3102, 3105 | Fertilizers |
| 7201-7229 | Iron and steel |
| 7601-7603 | Aluminium |
| 2716 | Electricity |

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "declaration_id": "CBAM-2025-Q4",
    "calculation_id": "cbam_calc_xyz789",
    "status": "completed",
    "reporting_period": {
      "quarter": "Q4",
      "year": 2025
    },
    "summary": {
      "total_imports": 2,
      "total_quantity_tonnes": 6000,
      "total_customs_value_eur": 1300000,
      "total_embedded_emissions_tco2e": 6270,
      "total_cbam_liability_eur": 502345,
      "carbon_price_credits_eur": 12500,
      "net_cbam_liability_eur": 489845
    },
    "product_calculations": [
      {
        "product_id": "IMP-001",
        "cn_code": "7208",
        "product_description": "Hot-rolled steel coils",
        "quantity_tonnes": 1000,
        "embedded_emissions": {
          "direct_tco2e": 1850,
          "indirect_tco2e": 420,
          "total_tco2e": 2270,
          "emission_intensity_tco2e_per_tonne": 2.27
        },
        "default_value_used": false,
        "eu_benchmark_tco2e_per_tonne": 1.85,
        "excess_emissions_tco2e": 420,
        "eu_ets_price_eur_tco2e": 85.50,
        "cbam_liability_eur": 194085,
        "carbon_price_paid": {
          "amount_eur": 12500,
          "type": "national_carbon_tax",
          "eligible_for_deduction": true
        },
        "net_liability_eur": 181585
      },
      {
        "product_id": "IMP-002",
        "cn_code": "2523",
        "product_description": "Portland cement",
        "quantity_tonnes": 5000,
        "embedded_emissions": {
          "direct_tco2e": 3500,
          "indirect_tco2e": 500,
          "total_tco2e": 4000,
          "emission_intensity_tco2e_per_tonne": 0.80
        },
        "default_value_used": true,
        "default_value_source": "EU CBAM Regulation Annex IV",
        "eu_benchmark_tco2e_per_tonne": 0.65,
        "excess_emissions_tco2e": 750,
        "eu_ets_price_eur_tco2e": 85.50,
        "cbam_liability_eur": 308260,
        "carbon_price_paid": null,
        "net_liability_eur": 308260
      }
    ],
    "eu_ets_price": {
      "value": 85.50,
      "unit": "EUR/tCO2e",
      "source": "EU ETS Official Auction",
      "date": "2025-12-01"
    },
    "compliance_status": {
      "declaration_due_date": "2026-01-31",
      "certificates_required": 5733,
      "certificates_purchased": 0,
      "certificates_to_purchase": 5733
    },
    "provenance_hash": "sha256:f6g7h8i9j0k1..."
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

### Example (cURL)

```bash
curl -X POST "https://api.greenlang.io/v1/compliance/cbam" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "declaration_id": "CBAM-2025-Q4",
    "reporting_period": {"quarter": "Q4", "year": 2025},
    "declarant": {
      "organization_id": "ORG-001",
      "eori_number": "DE123456789000"
    },
    "imports": [
      {
        "cn_code": "7208",
        "product_description": "Steel coils",
        "country_of_origin": "CN",
        "quantity": 1000,
        "quantity_unit": "tonnes",
        "use_default_values": true
      }
    ]
  }'
```

---

## EUDR Compliance

Check EU Deforestation Regulation compliance for commodities.

### Submit EUDR Due Diligence

```http
POST /api/v1/compliance/eudr
```

### Request Body

```json
{
  "statement_id": "EUDR-2025-001",
  "operator": {
    "organization_id": "ORG-001",
    "eori_number": "DE123456789000",
    "operator_type": "importer"
  },
  "commodities": [
    {
      "commodity_id": "COM-001",
      "commodity_type": "palm_oil",
      "cn_code": "1511",
      "product_description": "Crude palm oil",
      "quantity": 500,
      "quantity_unit": "tonnes",
      "country_of_production": "MY",
      "supply_chain": {
        "producer": {
          "name": "Palm Oil Plantation Sdn Bhd",
          "country": "MY",
          "registration_number": "REG-12345"
        },
        "geolocation": [
          {
            "type": "polygon",
            "coordinates": [
              [101.5, 3.2],
              [101.6, 3.2],
              [101.6, 3.1],
              [101.5, 3.1],
              [101.5, 3.2]
            ],
            "area_hectares": 1500
          }
        ],
        "production_date": "2025-06-15",
        "certifications": [
          {
            "type": "RSPO",
            "certificate_number": "RSPO-2025-12345",
            "valid_until": "2026-06-30"
          }
        ]
      }
    },
    {
      "commodity_id": "COM-002",
      "commodity_type": "soy",
      "cn_code": "1201",
      "product_description": "Soybeans",
      "quantity": 2000,
      "quantity_unit": "tonnes",
      "country_of_production": "BR",
      "supply_chain": {
        "producer": {
          "name": "Fazenda Santa Rosa",
          "country": "BR",
          "car_number": "MT-1234567-ABC"
        },
        "geolocation": [
          {
            "type": "point",
            "coordinates": [-55.1234, -15.5678],
            "accuracy_meters": 50
          }
        ],
        "production_date": "2025-03-20"
      }
    }
  ],
  "analysis_options": {
    "include_satellite_analysis": true,
    "deforestation_cutoff_date": "2020-12-31",
    "include_risk_assessment": true
  }
}
```

### Covered Commodities

| Commodity | CN Codes |
|-----------|----------|
| Cattle | 0102, 0201, 0202, 4101-4107 |
| Cocoa | 1801, 1802, 1803, 1804, 1805, 1806 |
| Coffee | 0901 |
| Oil Palm | 1511, 1513, 3823 |
| Rubber | 4001, 4002, 4005, 4006 |
| Soy | 1201, 1208, 1507, 2304 |
| Wood | 4401-4421, 9401, 9403 |

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "statement_id": "EUDR-2025-001",
    "analysis_id": "eudr_analysis_abc123",
    "status": "completed",
    "overall_compliance": "compliant_with_conditions",
    "risk_level": "low",
    "commodities_analyzed": 2,
    "summary": {
      "compliant": 1,
      "non_compliant": 0,
      "requires_review": 1
    },
    "commodity_results": [
      {
        "commodity_id": "COM-001",
        "commodity_type": "palm_oil",
        "compliance_status": "compliant",
        "risk_level": "negligible",
        "deforestation_check": {
          "status": "passed",
          "analysis_date": "2025-12-06",
          "satellite_imagery_source": "Sentinel-2",
          "imagery_date": "2025-11-30",
          "forest_cover_2020": {
            "area_hectares": 0,
            "percentage": 0
          },
          "forest_cover_current": {
            "area_hectares": 0,
            "percentage": 0
          },
          "deforestation_detected": false,
          "confidence_level": 0.95
        },
        "legality_check": {
          "status": "passed",
          "local_laws_compliant": true,
          "land_rights_verified": true,
          "certifications_valid": true
        },
        "traceability_check": {
          "status": "passed",
          "geolocation_verified": true,
          "supply_chain_documented": true,
          "producer_identified": true
        },
        "human_rights_check": {
          "status": "passed",
          "fpic_obtained": true,
          "no_forced_labor": true,
          "indigenous_rights_respected": true
        }
      },
      {
        "commodity_id": "COM-002",
        "commodity_type": "soy",
        "compliance_status": "requires_review",
        "risk_level": "standard",
        "deforestation_check": {
          "status": "passed",
          "satellite_imagery_source": "Sentinel-2",
          "deforestation_detected": false,
          "confidence_level": 0.92
        },
        "legality_check": {
          "status": "review_needed",
          "issues": [
            {
              "code": "EUDR_LEG_001",
              "message": "CAR registration requires verification",
              "severity": "warning",
              "recommendation": "Obtain official CAR certificate"
            }
          ]
        },
        "traceability_check": {
          "status": "passed",
          "geolocation_verified": true
        }
      }
    ],
    "required_actions": [
      {
        "commodity_id": "COM-002",
        "action": "Verify CAR registration with Brazilian authorities",
        "priority": "high",
        "deadline": "2025-12-20"
      }
    ],
    "due_diligence_statement": {
      "reference_number": "DDS-2025-001-ABC",
      "submission_ready": false,
      "blocking_issues": 1
    },
    "satellite_analysis": {
      "provider": "GreenLang Satellite Analytics",
      "imagery_sources": ["Sentinel-2", "Landsat-8"],
      "analysis_period": "2019-01-01 to 2025-12-01",
      "total_area_analyzed_hectares": 1500
    },
    "provenance_hash": "sha256:g7h8i9j0k1l2..."
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

---

## Report Management

### Get Report Status

```http
GET /api/v1/compliance/reports/{report_id}
```

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `report_id` | string | Report identifier |

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "report_id": "csrd_rpt_a1b2c3d4",
    "report_type": "csrd",
    "status": "completed",
    "organization_id": "ORG-001",
    "created_at": "2025-12-06T10:30:00Z",
    "completed_at": "2025-12-06T10:35:00Z",
    "download_urls": {
      "xbrl": "https://api.greenlang.io/v1/reports/csrd_rpt_a1b2c3d4/download?format=xbrl",
      "pdf": "https://api.greenlang.io/v1/reports/csrd_rpt_a1b2c3d4/download?format=pdf"
    },
    "metadata": {
      "reporting_year": 2025,
      "standards": ["ESRS_E1", "ESRS_E2"]
    }
  },
  "timestamp": "2025-12-06T10:40:00Z"
}
```

### Download Report

```http
GET /api/v1/compliance/reports/{report_id}/download
```

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `format` | string | No | Download format: `pdf`, `xbrl`, `json`, `csv`. Default: `pdf` |

### List Reports

```http
GET /api/v1/compliance/reports
```

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `report_type` | string | No | Filter by type: `csrd`, `cbam`, `eudr` |
| `organization_id` | string | No | Filter by organization |
| `year` | integer | No | Filter by reporting year |
| `status` | string | No | Filter by status |
| `page` | integer | No | Page number. Default: 1 |
| `per_page` | integer | No | Items per page. Default: 20 |

---

## Regulatory Deadlines

### Get Regulatory Deadlines

```http
GET /api/v1/compliance/deadlines
```

### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `regulation` | string | No | Filter by regulation: `csrd`, `cbam`, `eudr`, `eu_ets`, `sb253` |
| `organization_id` | string | No | Filter by organization |
| `start_date` | string | No | Start date (ISO 8601) |
| `end_date` | string | No | End date (ISO 8601) |
| `include_completed` | boolean | No | Include completed deadlines. Default: false |

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "deadlines": [
      {
        "deadline_id": "dl_001",
        "regulation": "cbam",
        "deadline_type": "quarterly_report",
        "title": "CBAM Q4 2025 Report Submission",
        "description": "Submit CBAM declaration for Q4 2025 imports",
        "due_date": "2026-01-31",
        "days_remaining": 56,
        "status": "upcoming",
        "priority": "high",
        "applicable_to": ["importers", "authorized_representatives"],
        "submission_portal": "https://cbam.ec.europa.eu",
        "requirements": [
          "Complete import data for all CBAM goods",
          "Emission data from installations or default values",
          "Carbon price information from country of origin"
        ],
        "related_reports": []
      },
      {
        "deadline_id": "dl_002",
        "regulation": "csrd",
        "deadline_type": "annual_report",
        "title": "CSRD Annual Sustainability Report 2025",
        "description": "Submit CSRD-compliant sustainability report",
        "due_date": "2026-04-30",
        "days_remaining": 145,
        "status": "upcoming",
        "priority": "high",
        "applicable_to": ["large_companies", "listed_smes"],
        "requirements": [
          "ESRS-compliant disclosures",
          "Double materiality assessment",
          "Limited or reasonable assurance"
        ]
      },
      {
        "deadline_id": "dl_003",
        "regulation": "eudr",
        "deadline_type": "due_diligence",
        "title": "EUDR Due Diligence Statement",
        "description": "Submit due diligence statement for covered commodities",
        "due_date": "2025-12-30",
        "days_remaining": 24,
        "status": "urgent",
        "priority": "critical",
        "applicable_to": ["operators", "traders"]
      }
    ],
    "summary": {
      "total_deadlines": 3,
      "urgent": 1,
      "upcoming": 2,
      "completed": 0
    }
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

### Create Deadline Reminder

```http
POST /api/v1/compliance/deadlines/reminders
```

### Request Body

```json
{
  "deadline_id": "dl_001",
  "reminder_days_before": [30, 14, 7, 1],
  "notification_channels": ["email", "webhook"],
  "recipients": ["compliance@example.com"]
}
```

---

## Audit Trail

### Get Audit Trail

```http
GET /api/v1/compliance/audit-trail/{report_id}
```

### Response (200 OK)

```json
{
  "success": true,
  "data": {
    "report_id": "csrd_rpt_a1b2c3d4",
    "chain_id": "chain_xyz789",
    "merkle_root": "sha256:a1b2c3d4e5f6g7h8...",
    "record_count": 48,
    "records": [
      {
        "record_id": "rec_001",
        "timestamp": "2025-12-06T10:30:00Z",
        "action": "report_initiated",
        "actor": "api_client",
        "actor_id": "client_abc123",
        "provenance_hash": "sha256:abc123...",
        "input_hash": "sha256:def456...",
        "output_hash": null
      },
      {
        "record_id": "rec_002",
        "timestamp": "2025-12-06T10:30:05Z",
        "action": "emissions_data_fetched",
        "actor": "GL-010",
        "actor_id": "GL-010-001",
        "provenance_hash": "sha256:ghi789...",
        "data_source": "emissions_api",
        "records_fetched": 365
      },
      {
        "record_id": "rec_003",
        "timestamp": "2025-12-06T10:30:15Z",
        "action": "esrs_e1_calculated",
        "actor": "GL-010",
        "actor_id": "GL-010-001",
        "provenance_hash": "sha256:jkl012...",
        "formula_id": "GHG_PROTOCOL_SCOPE1",
        "formula_reference": "GHG Protocol Corporate Standard"
      }
    ],
    "verification": {
      "chain_valid": true,
      "merkle_proof_valid": true,
      "all_signatures_valid": true,
      "verification_timestamp": "2025-12-06T10:40:00Z"
    }
  },
  "timestamp": "2025-12-06T10:40:00Z"
}
```

---

## Error Codes

| Code | HTTP Status | Description | Resolution |
|------|-------------|-------------|------------|
| `INVALID_REPORTING_PERIOD` | 400 | Invalid reporting period | Check year/quarter format |
| `MISSING_REQUIRED_DATA` | 400 | Required data not provided | Provide all required fields |
| `INVALID_CN_CODE` | 400 | Invalid CN code | Use valid customs nomenclature code |
| `GEOLOCATION_INVALID` | 400 | Invalid geolocation format | Use valid GeoJSON coordinates |
| `ORGANIZATION_NOT_FOUND` | 404 | Organization not found | Verify organization_id |
| `REPORT_NOT_FOUND` | 404 | Report not found | Verify report_id |
| `SATELLITE_DATA_UNAVAILABLE` | 503 | Satellite imagery not available | Retry later or use alternative period |
| `DEFORESTATION_DETECTED` | 200 | Deforestation detected in area | Review supply chain |
| `CBAM_PRODUCT_NOT_COVERED` | 400 | Product not covered by CBAM | Verify CN code against CBAM scope |
| `EMISSION_DATA_INCOMPLETE` | 400 | Incomplete emission data | Provide all emission fields |
| `ASSURANCE_LEVEL_INVALID` | 400 | Invalid assurance level | Use: none, limited, reasonable |

---

## Rate Limiting

Compliance endpoints have the following rate limits:

| Endpoint | Rate Limit |
|----------|------------|
| `POST /compliance/csrd` | 10/minute |
| `POST /compliance/cbam` | 50/minute |
| `POST /compliance/eudr` | 20/minute |
| `GET /compliance/reports/*` | 200/minute |
| `GET /compliance/deadlines` | 100/minute |
| `GET /compliance/audit-trail/*` | 50/minute |

---

## Best Practices

1. **Start report generation early** - Complex reports may take several minutes
2. **Use webhooks** for report completion notifications
3. **Validate data before submission** using the validation endpoint
4. **Store provenance hashes** for regulatory audit requirements
5. **Set up deadline reminders** to ensure timely submissions
6. **Include complete supply chain data** for EUDR compliance
7. **Use actual installation data** for CBAM when available (not defaults)

---

## See Also

- [Main API Reference](../process_heat_api_reference.md)
- [Orchestrator API](orchestrator.md)
- [Emissions API](emissions.md)
- [Common Data Models](../schemas/common_models.md)
