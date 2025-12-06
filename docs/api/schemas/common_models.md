# Common Data Models

This document defines the shared data models used across the GreenLang Process Heat APIs. These models ensure consistent data structures for inputs, outputs, and intermediate calculations.

---

## Table of Contents

1. [Core Models](#core-models)
2. [Fuel and Energy Models](#fuel-and-energy-models)
3. [Emission Models](#emission-models)
4. [Calculation Models](#calculation-models)
5. [Provenance and Audit Models](#provenance-and-audit-models)
6. [Compliance Models](#compliance-models)
7. [JSON Schema Definitions](#json-schema-definitions)

---

## Core Models

### APIResponse

Standard response wrapper for all API responses.

```json
{
  "success": true,
  "data": {},
  "error": null,
  "metadata": {
    "request_id": "string",
    "processing_time_ms": 0
  },
  "timestamp": "2025-12-06T10:30:00Z"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `success` | boolean | Yes | Request success status |
| `data` | object/array | No | Response data (null on error) |
| `error` | ErrorObject | No | Error details (null on success) |
| `metadata` | Metadata | No | Request metadata |
| `timestamp` | string | Yes | ISO 8601 response timestamp |

### ErrorObject

Error response structure.

```json
{
  "code": "VALIDATION_ERROR",
  "message": "Invalid boiler_id format",
  "details": {
    "field": "boiler_id",
    "constraint": "Must match pattern BLR-[0-9]{3}",
    "received": "invalid-id"
  },
  "documentation_url": "https://docs.greenlang.io/errors/VALIDATION_ERROR"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | string | Yes | Error code identifier |
| `message` | string | Yes | Human-readable error message |
| `details` | object | No | Additional error context |
| `documentation_url` | string | No | Link to error documentation |

### Pagination

Pagination metadata for list responses.

```json
{
  "page": 1,
  "per_page": 20,
  "total_items": 1250,
  "total_pages": 63,
  "has_next": true,
  "has_prev": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `page` | integer | Current page number (1-indexed) |
| `per_page` | integer | Items per page |
| `total_items` | integer | Total number of items |
| `total_pages` | integer | Total number of pages |
| `has_next` | boolean | Whether next page exists |
| `has_prev` | boolean | Whether previous page exists |

---

## Fuel and Energy Models

### FuelData

Fuel consumption and properties data.

```json
{
  "fuel_type": "natural_gas",
  "fuel_consumption": 150.0,
  "fuel_unit": "MMBTU/hr",
  "fuel_hhv": 1020.0,
  "fuel_hhv_unit": "BTU/SCF",
  "fuel_lhv": 918.0,
  "fuel_carbon_content_pct": 75.4,
  "fuel_moisture_content_pct": null,
  "fuel_ash_content_pct": null,
  "fuel_sulfur_content_pct": null,
  "fuel_density": null,
  "fuel_density_unit": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `fuel_type` | FuelType | Yes | Fuel type identifier |
| `fuel_consumption` | float | Yes | Consumption rate or total |
| `fuel_unit` | FuelUnit | Yes | Consumption unit |
| `fuel_hhv` | float | No | Higher heating value |
| `fuel_hhv_unit` | string | No | HHV unit |
| `fuel_lhv` | float | No | Lower heating value |
| `fuel_carbon_content_pct` | float | No | Carbon content (%) |
| `fuel_moisture_content_pct` | float | No | Moisture content (%) |
| `fuel_ash_content_pct` | float | No | Ash content (%) |
| `fuel_sulfur_content_pct` | float | No | Sulfur content (%) |
| `fuel_density` | float | No | Fuel density |
| `fuel_density_unit` | string | No | Density unit |

### FuelType (Enum)

Supported fuel types.

| Value | Description | Default HHV | Default EF (kg CO2/MMBTU) |
|-------|-------------|-------------|---------------------------|
| `natural_gas` | Natural gas | 1020 BTU/SCF | 53.06 |
| `fuel_oil_no2` | No. 2 fuel oil | 137,000 BTU/gal | 73.96 |
| `fuel_oil_no6` | No. 6 fuel oil | 150,000 BTU/gal | 75.10 |
| `diesel` | Diesel fuel | 137,000 BTU/gal | 73.96 |
| `propane` | Propane (LPG) | 91,500 BTU/gal | 62.87 |
| `butane` | Butane | 103,000 BTU/gal | 64.77 |
| `coal_bituminous` | Bituminous coal | 24.93 MMBTU/ton | 93.28 |
| `coal_subbituminous` | Subbituminous coal | 17.25 MMBTU/ton | 97.17 |
| `coal_anthracite` | Anthracite coal | 25.09 MMBTU/ton | 103.69 |
| `coal_lignite` | Lignite coal | 14.21 MMBTU/ton | 97.72 |
| `biomass_wood` | Wood biomass | 8.0 MMBTU/ton | 93.80 |
| `biomass_agricultural` | Agricultural biomass | 7.0 MMBTU/ton | 118.17 |
| `biogas` | Biogas | 600 BTU/SCF | 52.07 |
| `landfill_gas` | Landfill gas | 500 BTU/SCF | 52.07 |

### FuelUnit (Enum)

Fuel consumption units.

| Value | Description |
|-------|-------------|
| `MMBTU/hr` | Million BTU per hour |
| `MMBTU` | Million BTU (total) |
| `therms` | Therms (100,000 BTU) |
| `MCF` | Thousand cubic feet |
| `SCF` | Standard cubic feet |
| `gallons` | US gallons |
| `gallons/hr` | US gallons per hour |
| `kg` | Kilograms |
| `kg/hr` | Kilograms per hour |
| `tonnes` | Metric tonnes |
| `tonnes/hr` | Metric tonnes per hour |
| `MWh` | Megawatt-hours |
| `kWh` | Kilowatt-hours |

---

## Emission Models

### EmissionFactor

Emission factor specification.

```json
{
  "pollutant": "CO2",
  "value": 53.06,
  "unit": "kg/MMBTU",
  "source": {
    "reference": "40 CFR Part 98 Table C-1",
    "methodology": "EPA_PART_98",
    "publication_date": "2024-01-01",
    "last_updated": "2024-06-15"
  },
  "uncertainty": {
    "lower_bound": 52.0,
    "upper_bound": 54.1,
    "confidence_level": 0.95
  },
  "applicability": {
    "fuel_type": "natural_gas",
    "region": "US",
    "year": 2025,
    "combustion_type": "boiler"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pollutant` | Pollutant | Yes | Pollutant type |
| `value` | float | Yes | Emission factor value |
| `unit` | string | Yes | Emission factor unit |
| `source` | SourceReference | No | Source documentation |
| `uncertainty` | UncertaintyBounds | No | Uncertainty range |
| `applicability` | Applicability | No | Applicability scope |

### Pollutant (Enum)

Supported pollutants.

| Value | Description | GWP (AR5 100-year) |
|-------|-------------|-------------------|
| `CO2` | Carbon dioxide | 1 |
| `CH4` | Methane | 28 |
| `N2O` | Nitrous oxide | 298 |
| `CO` | Carbon monoxide | - |
| `NOx` | Nitrogen oxides | - |
| `SO2` | Sulfur dioxide | - |
| `PM` | Particulate matter | - |
| `PM2.5` | Fine particulate matter | - |
| `PM10` | Coarse particulate matter | - |
| `VOC` | Volatile organic compounds | - |
| `HFCs` | Hydrofluorocarbons | varies |
| `PFCs` | Perfluorocarbons | varies |
| `SF6` | Sulfur hexafluoride | 23,500 |
| `NF3` | Nitrogen trifluoride | 16,100 |

### EmissionResult

Emission calculation result.

```json
{
  "pollutant": "CO2",
  "value": 8567.2,
  "unit": "kg/hr",
  "annual_value": 75048792.0,
  "annual_unit": "kg/yr",
  "annual_tonnes": 75048.8,
  "co2e_value": 8567.2,
  "co2e_unit": "kg/hr",
  "emission_rate": {
    "value": 53.06,
    "unit": "kg/MMBTU"
  },
  "uncertainty": {
    "lower_bound": 8353.0,
    "upper_bound": 8781.4,
    "confidence_level": 0.95,
    "primary_sources": [
      "Fuel flow measurement: +/- 1.5%",
      "Emission factor: +/- 1.8%"
    ]
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pollutant` | Pollutant | Yes | Pollutant type |
| `value` | float | Yes | Emission value |
| `unit` | string | Yes | Emission unit |
| `annual_value` | float | No | Annualized value |
| `annual_unit` | string | No | Annual unit |
| `annual_tonnes` | float | No | Annual tonnes |
| `co2e_value` | float | No | CO2-equivalent value |
| `co2e_unit` | string | No | CO2e unit |
| `emission_rate` | EmissionRate | No | Emission rate (per fuel unit) |
| `uncertainty` | UncertaintyResult | No | Uncertainty analysis |

---

## Calculation Models

### CalculationResult

Generic calculation result wrapper.

```json
{
  "calculation_id": "calc_a1b2c3d4",
  "calculation_type": "efficiency",
  "status": "completed",
  "timestamp": "2025-12-06T10:30:00Z",
  "inputs_hash": "sha256:abc123...",
  "result": {
    "value": 85.7,
    "unit": "%",
    "confidence": 0.95
  },
  "method": {
    "name": "ASME_PTC_4.1_LOSSES",
    "reference": "ASME PTC 4.1-2013",
    "formula_id": "EFF_LOSSES"
  },
  "uncertainty": {
    "lower_bound": 84.1,
    "upper_bound": 87.3,
    "confidence_level": 0.95
  },
  "provenance_hash": "sha256:def456..."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `calculation_id` | string | Yes | Unique calculation ID |
| `calculation_type` | string | Yes | Type of calculation |
| `status` | string | Yes | Calculation status |
| `timestamp` | string | Yes | Calculation timestamp |
| `inputs_hash` | string | Yes | SHA-256 hash of inputs |
| `result` | ResultValue | Yes | Calculated result |
| `method` | CalculationMethod | Yes | Method used |
| `uncertainty` | UncertaintyResult | No | Uncertainty bounds |
| `provenance_hash` | string | Yes | Provenance chain hash |

### UncertaintyResult

Uncertainty analysis result.

```json
{
  "lower_bound": 84.1,
  "upper_bound": 87.3,
  "mean": 85.7,
  "std_deviation": 0.82,
  "confidence_level": 0.95,
  "distribution": "normal",
  "methodology": "monte_carlo",
  "samples": 10000,
  "primary_contributors": [
    {
      "parameter": "fuel_flow_rate",
      "contribution_pct": 45.2,
      "uncertainty_pct": 1.5
    },
    {
      "parameter": "flue_gas_o2",
      "contribution_pct": 28.5,
      "uncertainty_pct": 2.0
    }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `lower_bound` | float | Yes | Lower uncertainty bound |
| `upper_bound` | float | Yes | Upper uncertainty bound |
| `mean` | float | No | Mean value |
| `std_deviation` | float | No | Standard deviation |
| `confidence_level` | float | Yes | Confidence level (0-1) |
| `distribution` | string | No | Assumed distribution |
| `methodology` | string | No | Uncertainty methodology |
| `samples` | integer | No | Monte Carlo samples (if applicable) |
| `primary_contributors` | array | No | Primary uncertainty sources |

### EfficiencyResult

Boiler/equipment efficiency result.

```json
{
  "gross_efficiency_pct": 85.7,
  "net_efficiency_pct": 83.2,
  "combustion_efficiency_pct": 88.5,
  "total_losses_pct": 14.3,
  "loss_breakdown": {
    "dry_flue_gas_loss_pct": 8.2,
    "moisture_in_fuel_loss_pct": 0.0,
    "moisture_from_h2_loss_pct": 3.8,
    "radiation_loss_pct": 1.5,
    "blowdown_loss_pct": 0.8,
    "unburned_loss_pct": 0.0,
    "other_losses_pct": 0.0
  },
  "excess_air_pct": 17.2,
  "heat_input_btu_hr": 162000000,
  "heat_output_btu_hr": 138834000,
  "calculation_method": "ASME_PTC_4.1_LOSSES",
  "formula_reference": "ASME PTC 4.1-2013"
}
```

---

## Provenance and Audit Models

### ProvenanceRecord

Immutable provenance record for audit trails.

```json
{
  "record_id": "rec_a1b2c3d4",
  "provenance_hash": "sha256:a1b2c3d4e5f6g7h8i9j0...",
  "provenance_type": "CALCULATION",
  "agent_id": "GL-002-001",
  "agent_version": "2.1.0",
  "timestamp": "2025-12-06T10:30:00Z",
  "input_hash": "sha256:abc123...",
  "output_hash": "sha256:def456...",
  "formula_id": "ASME_PTC_4.1",
  "formula_reference": "ASME PTC 4.1-2013",
  "parent_records": ["rec_xyz789"],
  "data_lineage": {
    "source_id": "sensor_001",
    "source_type": "OPC_UA",
    "source_timestamp": "2025-12-06T10:29:55Z",
    "quality_score": 0.98
  },
  "classification": "internal",
  "compliance_frameworks": ["ISO_14064", "GHG_PROTOCOL"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `record_id` | string | Yes | Unique record identifier |
| `provenance_hash` | string | Yes | SHA-256 provenance hash |
| `provenance_type` | ProvenanceType | Yes | Type of record |
| `agent_id` | string | Yes | Agent that created record |
| `agent_version` | string | Yes | Agent version |
| `timestamp` | string | Yes | Record timestamp |
| `input_hash` | string | Yes | Hash of input data |
| `output_hash` | string | Yes | Hash of output data |
| `formula_id` | string | No | Formula identifier |
| `formula_reference` | string | No | Standard reference |
| `parent_records` | array | No | Parent record IDs |
| `data_lineage` | DataLineage | No | Source data lineage |
| `classification` | string | No | Data classification |
| `compliance_frameworks` | array | No | Applicable frameworks |

### ProvenanceType (Enum)

Types of provenance records.

| Value | Description |
|-------|-------------|
| `INPUT` | Data input record |
| `OUTPUT` | Data output record |
| `TRANSFORMATION` | Data transformation |
| `CALCULATION` | Calculation performed |
| `AGGREGATION` | Data aggregation |
| `VALIDATION` | Data validation |
| `CORRECTION` | Data correction |
| `APPROVAL` | Manual approval |

### DataLineage

Data source lineage information.

```json
{
  "source_id": "sensor_001",
  "source_type": "OPC_UA",
  "source_timestamp": "2025-12-06T10:29:55Z",
  "source_hash": "sha256:ghi789...",
  "transformation_chain": [
    "transform_001",
    "transform_002"
  ],
  "quality_score": 0.98,
  "metadata": {
    "sensor_calibration_date": "2025-11-01",
    "measurement_uncertainty": 0.015
  }
}
```

### ProvenanceChain

Chain of provenance records with Merkle root.

```json
{
  "chain_id": "chain_xyz789",
  "merkle_root": "sha256:a1b2c3d4e5f6...",
  "record_count": 48,
  "start_timestamp": "2025-12-06T10:00:00Z",
  "end_timestamp": "2025-12-06T10:35:00Z",
  "agent_ids": ["GL-001", "GL-002", "GL-010"],
  "verification_status": "valid"
}
```

---

## Compliance Models

### ComplianceStatus

Compliance check result.

```json
{
  "regulation": "cbam",
  "compliance_status": "compliant",
  "check_timestamp": "2025-12-06T10:30:00Z",
  "reporting_period": {
    "start": "2025-10-01",
    "end": "2025-12-31"
  },
  "requirements_met": [
    "emissions_declared",
    "installation_data_provided",
    "carbon_price_documented"
  ],
  "requirements_pending": [],
  "requirements_failed": [],
  "next_deadline": {
    "action": "quarterly_report",
    "due_date": "2026-01-31",
    "days_remaining": 56
  }
}
```

### RegulatoryDeadline

Regulatory deadline specification.

```json
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
    "Emission data from installations or default values"
  ]
}
```

### DeadlineStatus (Enum)

Deadline status values.

| Value | Description |
|-------|-------------|
| `upcoming` | Deadline approaching (>14 days) |
| `urgent` | Deadline soon (<=14 days) |
| `overdue` | Past due date |
| `completed` | Submission completed |
| `waived` | Deadline waived/not applicable |

---

## JSON Schema Definitions

### FuelData JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://api.greenlang.io/schemas/fuel-data.json",
  "title": "FuelData",
  "description": "Fuel consumption and properties data",
  "type": "object",
  "required": ["fuel_type", "fuel_consumption", "fuel_unit"],
  "properties": {
    "fuel_type": {
      "type": "string",
      "enum": [
        "natural_gas", "fuel_oil_no2", "fuel_oil_no6", "diesel",
        "propane", "butane", "coal_bituminous", "coal_subbituminous",
        "coal_anthracite", "coal_lignite", "biomass_wood",
        "biomass_agricultural", "biogas", "landfill_gas"
      ],
      "description": "Fuel type identifier"
    },
    "fuel_consumption": {
      "type": "number",
      "exclusiveMinimum": 0,
      "description": "Fuel consumption value"
    },
    "fuel_unit": {
      "type": "string",
      "enum": [
        "MMBTU/hr", "MMBTU", "therms", "MCF", "SCF",
        "gallons", "gallons/hr", "kg", "kg/hr",
        "tonnes", "tonnes/hr", "MWh", "kWh"
      ],
      "description": "Fuel consumption unit"
    },
    "fuel_hhv": {
      "type": "number",
      "exclusiveMinimum": 0,
      "description": "Higher heating value"
    },
    "fuel_hhv_unit": {
      "type": "string",
      "description": "HHV unit"
    },
    "fuel_carbon_content_pct": {
      "type": "number",
      "minimum": 0,
      "maximum": 100,
      "description": "Carbon content percentage"
    }
  },
  "additionalProperties": true
}
```

### EmissionFactor JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://api.greenlang.io/schemas/emission-factor.json",
  "title": "EmissionFactor",
  "description": "Emission factor specification",
  "type": "object",
  "required": ["pollutant", "value", "unit"],
  "properties": {
    "pollutant": {
      "type": "string",
      "enum": [
        "CO2", "CH4", "N2O", "CO", "NOx", "SO2",
        "PM", "PM2.5", "PM10", "VOC", "HFCs", "PFCs", "SF6", "NF3"
      ],
      "description": "Pollutant type"
    },
    "value": {
      "type": "number",
      "exclusiveMinimum": 0,
      "description": "Emission factor value"
    },
    "unit": {
      "type": "string",
      "pattern": "^[a-zA-Z0-9/]+$",
      "description": "Emission factor unit"
    },
    "source": {
      "type": "object",
      "properties": {
        "reference": { "type": "string" },
        "methodology": { "type": "string" },
        "publication_date": { "type": "string", "format": "date" }
      }
    },
    "uncertainty": {
      "type": "object",
      "properties": {
        "lower_bound": { "type": "number" },
        "upper_bound": { "type": "number" },
        "confidence_level": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    }
  },
  "additionalProperties": true
}
```

### ProvenanceRecord JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://api.greenlang.io/schemas/provenance-record.json",
  "title": "ProvenanceRecord",
  "description": "Immutable provenance record for audit trails",
  "type": "object",
  "required": [
    "record_id", "provenance_hash", "provenance_type",
    "agent_id", "agent_version", "timestamp",
    "input_hash", "output_hash"
  ],
  "properties": {
    "record_id": {
      "type": "string",
      "pattern": "^rec_[a-zA-Z0-9]+$",
      "description": "Unique record identifier"
    },
    "provenance_hash": {
      "type": "string",
      "pattern": "^sha256:[a-f0-9]{64}$",
      "description": "SHA-256 provenance hash"
    },
    "provenance_type": {
      "type": "string",
      "enum": [
        "INPUT", "OUTPUT", "TRANSFORMATION",
        "CALCULATION", "AGGREGATION", "VALIDATION",
        "CORRECTION", "APPROVAL"
      ],
      "description": "Type of provenance record"
    },
    "agent_id": {
      "type": "string",
      "description": "Agent that created record"
    },
    "agent_version": {
      "type": "string",
      "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$",
      "description": "Agent version (semver)"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "Record timestamp"
    },
    "input_hash": {
      "type": "string",
      "pattern": "^sha256:[a-f0-9]{64}$",
      "description": "Hash of input data"
    },
    "output_hash": {
      "type": "string",
      "pattern": "^sha256:[a-f0-9]{64}$",
      "description": "Hash of output data"
    },
    "formula_id": {
      "type": "string",
      "description": "Formula identifier"
    },
    "formula_reference": {
      "type": "string",
      "description": "Engineering standard reference"
    },
    "parent_records": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Parent record IDs"
    },
    "compliance_frameworks": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": [
          "SOX", "ISO_14064", "GHG_PROTOCOL",
          "EPA_PART_98", "EU_ETS", "CSRD"
        ]
      },
      "description": "Applicable compliance frameworks"
    }
  },
  "additionalProperties": true
}
```

### CalculationResult JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://api.greenlang.io/schemas/calculation-result.json",
  "title": "CalculationResult",
  "description": "Generic calculation result wrapper",
  "type": "object",
  "required": [
    "calculation_id", "calculation_type", "status",
    "timestamp", "result", "provenance_hash"
  ],
  "properties": {
    "calculation_id": {
      "type": "string",
      "pattern": "^calc_[a-zA-Z0-9]+$",
      "description": "Unique calculation ID"
    },
    "calculation_type": {
      "type": "string",
      "description": "Type of calculation"
    },
    "status": {
      "type": "string",
      "enum": ["pending", "processing", "completed", "failed"],
      "description": "Calculation status"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "Calculation timestamp"
    },
    "result": {
      "type": "object",
      "properties": {
        "value": { "type": "number" },
        "unit": { "type": "string" },
        "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
      },
      "required": ["value", "unit"]
    },
    "method": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "reference": { "type": "string" },
        "formula_id": { "type": "string" }
      }
    },
    "uncertainty": {
      "type": "object",
      "properties": {
        "lower_bound": { "type": "number" },
        "upper_bound": { "type": "number" },
        "confidence_level": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },
    "provenance_hash": {
      "type": "string",
      "pattern": "^sha256:[a-f0-9]{64}$",
      "description": "Provenance chain hash"
    }
  },
  "additionalProperties": true
}
```

---

## See Also

- [Main API Reference](../process_heat_api_reference.md)
- [Orchestrator API](../endpoints/orchestrator.md)
- [Emissions API](../endpoints/emissions.md)
- [Compliance API](../endpoints/compliance.md)
- [OpenAPI Specification](../openapi/process_heat_openapi.yaml)
