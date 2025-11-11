# Product Carbon Footprint & Digital Product Passport Requirements
**EU Battery Regulation (EU) 2023/1542 & Ecodesign for Sustainable Products Regulation**
**Battery Passport Deadline: February 18, 2027**

## Executive Summary
Digital Product Passports (DPP) and Product Carbon Footprints (PCF) are becoming mandatory for various product categories, starting with batteries, to enable circularity, traceability, and informed purchasing decisions throughout product lifecycles.

## Key Requirements - Battery Passport (First Implementation)

### 1. Battery Categories Covered
- **EV batteries**: >2 kWh capacity
- **Industrial batteries**: >2 kWh capacity
- **LMT batteries**: Light means of transport (e-scooters, e-bikes)
- **SLI batteries**: Starting, lighting, ignition
- **Portable batteries**: From 2028

### 2. Information Requirements
- **Product identification**: Unique identifier, QR code
- **Carbon footprint**: Manufacturing and lifecycle emissions
- **Material composition**: Critical raw materials content
- **Performance data**: Capacity, power, efficiency, durability
- **Supply chain**: Due diligence information
- **End-of-life**: Recycling instructions, dismantling info

### 3. Timeline
- **Feb 18, 2024**: Regulation enters into force
- **Feb 18, 2025**: Carbon footprint declaration
- **Feb 18, 2027**: Digital battery passport mandatory
- **Feb 18, 2028**: Carbon footprint performance classes
- **Feb 18, 2028**: Recycled content declaration

## Digital Product Passport Data Model

### Core Product Information
```json
{
  "product_passport": {
    "passport_id": "DPP-BAT-2027-XXXXX",
    "product_identification": {
      "manufacturer": "Company Name",
      "model": "Model XYZ",
      "batch_number": "2027-01-001",
      "serial_number": "SN123456789",
      "manufacturing_date": "2027-01-15",
      "manufacturing_location": "Factory, Country"
    },
    "compliance": {
      "ce_marking": true,
      "standards": ["IEC 62133", "UN 38.3"],
      "declaration_of_conformity": "DOC-2027-12345"
    },
    "qr_code": {
      "data_carrier": "QR_CODE",
      "unique_identifier": "BAT-UUID-123e4567-e89b",
      "url": "https://dpp.company.com/BAT-UUID-123"
    }
  }
}
```

### Carbon Footprint Declaration
```json
{
  "carbon_footprint": {
    "calculation_standard": "ISO 14067",
    "functional_unit": "1 kWh battery capacity",
    "system_boundaries": "cradle_to_gate",
    "total_cf": {
      "value": 75.5,
      "unit": "kg CO2-eq/kWh",
      "uncertainty": 10
    },
    "lifecycle_phases": {
      "raw_materials": {
        "total": 45.2,
        "breakdown": {
          "cathode_materials": 25.5,
          "anode_materials": 8.3,
          "electrolyte": 4.2,
          "casing": 3.1,
          "electronics": 4.1
        }
      },
      "manufacturing": {
        "total": 20.3,
        "electricity": 15.2,
        "heat": 3.1,
        "direct_emissions": 2.0
      },
      "distribution": {
        "total": 5.0,
        "transport_mode": "sea_road",
        "distance_km": 8000
      },
      "end_of_life": {
        "total": 5.0,
        "recycling_credit": -10.0
      }
    },
    "performance_class": "B",
    "verification": {
      "verifier": "Certification Body",
      "date": "2027-01-10",
      "certificate": "CERT-2027-001"
    }
  }
}
```

### Material Composition & Circularity
```json
{
  "material_composition": {
    "battery_chemistry": "NMC811",
    "critical_raw_materials": {
      "cobalt": {
        "content_kg": 5.2,
        "percentage": 8.5,
        "source_countries": ["DRC", "Australia"],
        "recycled_content": 0.15
      },
      "lithium": {
        "content_kg": 3.8,
        "percentage": 6.2,
        "source_countries": ["Chile", "Australia"],
        "recycled_content": 0.05
      },
      "nickel": {
        "content_kg": 12.5,
        "percentage": 20.4,
        "source_countries": ["Indonesia", "Canada"],
        "recycled_content": 0.25
      },
      "graphite": {
        "content_kg": 8.0,
        "percentage": 13.1,
        "source_countries": ["China", "Mozambique"],
        "recycled_content": 0.10
      }
    },
    "hazardous_substances": {
      "reach_compliance": true,
      "rohs_compliance": true,
      "substances_of_concern": []
    },
    "recycled_content": {
      "cobalt": 15,
      "lithium": 5,
      "nickel": 25,
      "lead": 85
    }
  }
}
```

### Performance & Durability Data
```json
{
  "performance_data": {
    "rated_capacity": {
      "value": 75,
      "unit": "kWh"
    },
    "nominal_voltage": {
      "value": 400,
      "unit": "V"
    },
    "expected_lifetime": {
      "cycles": 3000,
      "years": 10,
      "warranty_years": 8
    },
    "state_of_health": {
      "initial_soh": 100,
      "current_soh": 95,
      "measurement_date": "2027-06-01"
    },
    "performance_metrics": {
      "round_trip_efficiency": 0.95,
      "self_discharge_rate": 0.02,
      "temperature_range": {
        "min": -20,
        "max": 60,
        "unit": "celsius"
      },
      "c_rate": {
        "charge": 1,
        "discharge": 2
      }
    }
  }
}
```

## Product Carbon Footprint Methodology

### System Boundaries Definition
```
PCF Scope = Raw Materials + Manufacturing + Distribution + Use + End-of-Life

Cradle-to-Gate = Raw Materials + Manufacturing
Cradle-to-Grave = All lifecycle stages
```

### Calculation Standards
- **ISO 14067**: Product carbon footprint
- **ISO 14040/44**: Life cycle assessment
- **PEF Method**: EU Product Environmental Footprint
- **GHG Protocol Product Standard**

### Data Quality Requirements
```
Data Quality Rating = (TiR + GeR + TeR + Completeness + Reliability) / 5

Where (1=best, 5=worst):
- TiR: Time representativeness
- GeR: Geographic representativeness
- TeR: Technological representativeness
- Completeness: Data coverage
- Reliability: Source quality
```

### Allocation Methods
```
Economic Allocation:
Emissions_Product = Total_Emissions × (Value_Product / Total_Value)

Mass Allocation:
Emissions_Product = Total_Emissions × (Mass_Product / Total_Mass)

Energy Allocation:
Emissions_Product = Total_Emissions × (Energy_Product / Total_Energy)
```

## Digital Product Passport Technical Architecture

### Data Storage Requirements
```json
{
  "storage_architecture": {
    "decentralized_registry": {
      "technology": "distributed_ledger",
      "nodes": ["manufacturer", "authorities", "recyclers"],
      "consensus": "proof_of_authority"
    },
    "data_hosting": {
      "passport_data": "manufacturer_server",
      "backup": "cloud_redundancy",
      "retention": "product_lifetime_plus_10_years"
    },
    "access_control": {
      "public_data": ["basic_info", "performance_class"],
      "restricted_data": ["detailed_composition", "supplier_info"],
      "authentication": "oauth2_oidc"
    }
  }
}
```

### API Specifications
```yaml
openapi: 3.0.0
paths:
  /passport/{id}:
    get:
      summary: Retrieve product passport
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ProductPassport'

  /passport/{id}/carbon-footprint:
    get:
      summary: Get carbon footprint data
      security:
        - bearerAuth: []
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CarbonFootprint'
```

### QR Code Requirements
- **Data capacity**: Minimum 200 bytes
- **Error correction**: Level H (30%)
- **Size**: Minimum 2cm × 2cm
- **Durability**: Readable for product lifetime
- **Placement**: Accessible without dismantling

## Penalties for Non-Compliance

### Battery Regulation Penalties
- **Maximum fines**: Up to 4% of annual turnover
- **Product recall**: For non-compliant products
- **Market prohibition**: Until compliance achieved
- **Criminal penalties**: For falsification

### Administrative Measures
- Import/export restrictions
- Withdrawal from market
- Public warnings
- Corrective action orders

### Market Surveillance
- Random sampling and testing
- Document verification
- Laboratory analysis
- Supply chain audits

## Scope - Extended to Other Products

### Textiles (From 2025)
- Fiber composition
- Chemical content
- Manufacturing location
- Social compliance
- Recycling information

### Electronics (From 2026)
- Material composition
- Repairability score
- Software support duration
- Energy efficiency
- Recycling instructions

### Construction Products (From 2027)
- Environmental performance
- Embodied carbon
- Recycled content
- Hazardous substances
- Demolition instructions

### Furniture (From 2028)
- Material sources
- Durability rating
- Repair information
- Chemical emissions
- End-of-life options

## Implementation Requirements

### Technical Infrastructure

#### Data Management System
- Product database
- Carbon calculation engine
- Supply chain integration
- Document repository
- Version control

#### Digital Passport Platform
- UUID generation
- QR code management
- API gateway
- Access control
- Audit logging

#### Verification System
- Third-party integration
- Certificate management
- Compliance checking
- Update mechanisms

### Implementation Steps

1. **Product Registration**
   - Generate unique identifier
   - Create digital record
   - Link physical product

2. **Data Collection**
   - Supply chain mapping
   - Material tracking
   - Process data gathering
   - Testing and analysis

3. **Carbon Footprint Calculation**
   - Define boundaries
   - Collect activity data
   - Apply emission factors
   - Uncertainty assessment

4. **Passport Creation**
   - Compile required data
   - Generate QR code
   - Set access permissions
   - Deploy to platform

5. **Verification**
   - Third-party audit
   - Data validation
   - Certificate issuance
   - Compliance confirmation

## Data Standards and Protocols

### Interoperability Standards
- **EPCIS 2.0**: Supply chain visibility
- **GS1 Digital Link**: Product identification
- **JSON-LD**: Linked data format
- **W3C DID**: Decentralized identifiers
- **IPFS**: Distributed storage

### Carbon Accounting Standards
- **ISO 14067**: Product carbon footprint
- **PAS 2050**: Product GHG emissions
- **GHG Protocol**: Product standard
- **EN 15804**: Construction products

### Data Exchange Formats
```json
{
  "message_format": "JSON",
  "encoding": "UTF-8",
  "compression": "gzip",
  "encryption": "AES-256",
  "signature": "ECDSA"
}
```

## Future Developments

### Upcoming Requirements
- **2028**: Portable batteries included
- **2029**: Maximum carbon thresholds
- **2030**: Mandatory recycled content
- **2031**: Full supply chain transparency
- **2035**: Circular economy targets

### Technology Evolution
- Blockchain integration
- AI-powered verification
- IoT sensor data
- Real-time tracking
- Automated reporting