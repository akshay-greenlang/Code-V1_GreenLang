# AGENT-EUDR-037: Due Diligence Statement Creator API

**Agent ID:** `GL-EUDR-DDSC-037`
**Prefix:** `/api/v1/eudr/dds-creator`
**Version:** 1.0.0
**PRD:** PRD-AGENT-EUDR-037
**Regulation:** EU 2023/1115 (EUDR) -- DDS lifecycle per Articles 4, 8, 9, 10, 12, 13, 14, 31

## Purpose

The Due Diligence Statement Creator agent manages the full lifecycle of the
EUDR Due Diligence Statement (DDS) -- the mandatory document that operators
must submit to the EU Information System before placing or making available
relevant commodities on the EU market. It handles DDS creation, Article 9
data assembly, geolocation formatting, risk data integration, supply chain
compilation, compliance validation, document packaging, digital signing,
version control, amendment tracking, and EU Information System submission.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/create-dds` | Create new DDS | JWT |
| POST | `/assemble-dds` | Assemble complete DDS | JWT |
| GET | `/dds` | List DDS records | JWT |
| GET | `/dds/{statement_id}` | Get DDS by ID | JWT |
| GET | `/dds/{statement_id}/summary` | Get DDS summary | JWT |
| PUT | `/dds/{statement_id}/status` | Update DDS status | JWT |
| DELETE | `/dds/{statement_id}` | Withdraw DDS | JWT |
| POST | `/dds/{statement_id}/geolocations` | Format geolocation | JWT |
| POST | `/dds/{statement_id}/geolocations/batch` | Batch format geolocations | JWT |
| GET | `/dds/{statement_id}/geolocations/geojson` | Export as GeoJSON | JWT |
| POST | `/dds/{statement_id}/risk-references` | Integrate risk data | JWT |
| POST | `/dds/{statement_id}/risk-references/batch` | Batch integrate risk | JWT |
| GET | `/dds/{statement_id}/risk-references/overall` | Get overall risk level | JWT |
| POST | `/dds/{statement_id}/supply-chain` | Compile supply chain | JWT |
| GET | `/dds/{statement_id}/supply-chain/completeness` | Validate completeness | JWT |
| GET | `/dds/{statement_id}/supply-chain/countries` | Get country summary | JWT |
| POST | `/dds/{statement_id}/validate` | Validate DDS compliance | JWT |
| GET | `/dds/{statement_id}/compliance` | Get compliance report | JWT |
| POST | `/dds/{statement_id}/documents` | Add document to package | JWT |
| POST | `/dds/{statement_id}/package` | Create submission package | JWT |
| GET | `/dds/{statement_id}/package/validate` | Validate package | JWT |
| GET | `/dds/{statement_id}/package/manifest` | Get document manifest | JWT |
| POST | `/dds/{statement_id}/sign` | Apply digital signature | JWT |
| GET | `/dds/{statement_id}/signature/validate` | Validate signature | JWT |
| POST | `/dds/{statement_id}/amend` | Create amendment | JWT |
| GET | `/dds/{statement_id}/versions` | Get version history | JWT |
| GET | `/dds/{statement_id}/versions/latest` | Get latest version | JWT |
| GET | `/dds/{statement_id}/amendments` | Get amendment records | JWT |
| POST | `/dds/{statement_id}/submit` | Submit to EU IS | JWT |
| GET | `/health` | Health check | None |

**Total: 30 endpoints**

---

## Endpoints

### POST /api/v1/eudr/dds-creator/create-dds

Create a new Due Diligence Statement in draft status with operator information
and the commodities being placed on the market.

**Request:**

```json
{
  "operator_id": "OP-2024-001",
  "operator_name": "EuroCocoa Trading GmbH",
  "operator_address": "Handelstrasse 42, 20457 Hamburg, Germany",
  "operator_eori_number": "DE123456789000",
  "statement_type": "placing",
  "commodities": ["cocoa"],
  "language": "en"
}
```

**Response (200 OK):**

```json
{
  "statement_id": "DDS-2026-001",
  "operator_id": "OP-2024-001",
  "operator_name": "EuroCocoa Trading GmbH",
  "statement_type": "placing",
  "commodities": ["cocoa"],
  "status": "draft",
  "version": 1,
  "completeness_pct": 15.0,
  "created_at": "2026-04-04T10:00:00Z",
  "provenance_hash": "sha256:a1b2c3d4..."
}
```

---

### POST /api/v1/eudr/dds-creator/dds/{statement_id}/geolocations

Format geolocation data for a production plot per EUDR Article 9(1)(d)
requirements, ensuring WGS 84 coordinates and proper polygon formatting.

**Request:**

```json
{
  "plot_id": "plot-GH-001",
  "latitude": 6.1256,
  "longitude": -1.5231,
  "area_hectares": 4.5,
  "polygon_coordinates": [
    [-1.524, 6.124], [-1.524, 6.128], [-1.522, 6.128], [-1.522, 6.124], [-1.524, 6.124]
  ],
  "country_code": "GH",
  "collection_method": "gps_field_survey"
}
```

**Response (200 OK):**

```json
{
  "geolocation_id": "geo_001",
  "statement_id": "DDS-2026-001",
  "plot_id": "plot-GH-001",
  "formatted_coordinates": {
    "latitude": 6.1256,
    "longitude": -1.5231,
    "coordinate_system": "WGS84"
  },
  "area_hectares": 4.5,
  "country_code": "GH",
  "article9_compliant": true,
  "added_at": "2026-04-04T10:05:00Z"
}
```

---

### POST /api/v1/eudr/dds-creator/dds/{statement_id}/validate

Validate a DDS against all EUDR Article 4 requirements, checking operator
information, product descriptions, geolocation completeness, risk assessment
references, supply chain data, and mandatory declarations.

**Response (200 OK):**

```json
{
  "statement_id": "DDS-2026-001",
  "validation_status": "passed",
  "completeness_pct": 100.0,
  "checks": [
    {"check": "operator_info", "status": "passed", "details": "All operator fields complete"},
    {"check": "product_descriptions", "status": "passed", "details": "3 products with HS codes"},
    {"check": "geolocations", "status": "passed", "details": "12 plots with valid coordinates"},
    {"check": "risk_assessment", "status": "passed", "details": "Risk assessment RAO-001 linked"},
    {"check": "supply_chain", "status": "passed", "details": "Full traceability to origin"},
    {"check": "deforestation_free", "status": "passed", "details": "All plots verified"},
    {"check": "legally_produced", "status": "passed", "details": "Legal compliance confirmed"}
  ],
  "warnings": [],
  "ready_for_submission": true,
  "validated_at": "2026-04-04T10:15:00Z"
}
```

---

### POST /api/v1/eudr/dds-creator/dds/{statement_id}/sign

Apply a qualified electronic signature (eIDAS) to the DDS before submission.

**Request:**

```json
{
  "signer_name": "Hans Mueller",
  "signer_role": "Compliance Director",
  "signer_organization": "EuroCocoa Trading GmbH",
  "signature_type": "qualified_electronic",
  "signed_hash": "sha256:f1g2h3i4j5k6l7m8n9o0p1q2r3s4t5u6v7w8x9y0z1a2b3c4d5e6f7g8h9i0"
}
```

**Response (200 OK):**

```json
{
  "signature_id": "sig_001",
  "statement_id": "DDS-2026-001",
  "signer_name": "Hans Mueller",
  "signature_type": "qualified_electronic",
  "signature_valid": true,
  "eidas_compliant": true,
  "signed_at": "2026-04-04T10:20:00Z",
  "certificate_serial": "ABC123DEF456"
}
```

---

### POST /api/v1/eudr/dds-creator/dds/{statement_id}/submit

Submit a validated and signed DDS to the EU Information System. The DDS
must be in a validated and signed state.

**Request:**

```json
{
  "additional_documents": [
    {
      "type": "supporting_evidence",
      "filename": "satellite_evidence_package.zip",
      "hash_sha256": "sha256:..."
    }
  ]
}
```

**Response (200 OK):**

```json
{
  "submission_id": "sub_001",
  "statement_id": "DDS-2026-001",
  "eu_reference_number": "EU-DDS-2026-DE-00001234",
  "status": "submitted",
  "submitted_at": "2026-04-04T10:25:00Z",
  "expected_confirmation": "2026-04-04T11:25:00Z",
  "documents_submitted": 5,
  "package_hash": "sha256:p1q2r3s4..."
}
```

---

### POST /api/v1/eudr/dds-creator/dds/{statement_id}/amend

Create an amendment to a previously submitted DDS when new information
requires changes.

**Request:**

```json
{
  "reason": "Updated geolocation data from field survey",
  "description": "Added GPS coordinates for 3 previously unmapped plots",
  "previous_version": 1,
  "changed_fields": ["geolocations"],
  "changed_by": "field-team@company.com",
  "approved_by": "compliance-manager@company.com"
}
```

**Response (200 OK):**

```json
{
  "amendment_id": "amd_001",
  "statement_id": "DDS-2026-001",
  "new_version": 2,
  "reason": "Updated geolocation data from field survey",
  "changed_fields": ["geolocations"],
  "status": "draft",
  "requires_resubmission": true,
  "created_at": "2026-04-04T10:30:00Z"
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 404 | `dds_not_found` | DDS statement ID not found |
| 422 | `validation_failed` | DDS does not meet Article 4 requirements |
| 422 | `signature_invalid` | Digital signature verification failed |
| 422 | `not_ready_for_submission` | DDS not validated or not signed |
| 500 | `submission_failed` | EU IS submission failed |
