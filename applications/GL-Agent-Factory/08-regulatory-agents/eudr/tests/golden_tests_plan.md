# EUDR Golden Tests Plan - 200 Tests

**Version:** 1.0.0
**Date:** December 4, 2025
**Target:** 200 Golden Tests
**Deadline:** December 30, 2025

---

## Executive Summary

This document outlines the comprehensive golden test suite for the EUDR Compliance Agent. The 200 tests are designed to validate all aspects of EUDR compliance verification including geolocation validation, deforestation detection, supply chain traceability, and DDS generation.

---

## Test Distribution

| Category | Test Count | Priority | Owner |
|----------|------------|----------|-------|
| Commodity Validation | 35 | P0 | gl-test-engineer |
| Geolocation Validation | 25 | P0 | gl-test-engineer |
| Satellite/Deforestation | 30 | P0 | gl-satellite-ml-specialist |
| Supply Chain Traceability | 25 | P0 | gl-test-engineer |
| DDS Generation | 20 | P0 | gl-test-engineer |
| Risk Assessment | 20 | P1 | gl-test-engineer |
| Edge Cases | 25 | P1 | gl-test-engineer |
| Integration/E2E | 20 | P0 | gl-test-engineer |
| **TOTAL** | **200** | | |

---

## Category 1: Commodity Validation Tests (35 tests)

### 1.1 Cattle Tests (5 tests)
| Test ID | Description | Input | Expected Output |
|---------|-------------|-------|-----------------|
| EUDR-GT-CAT-001 | Valid cattle import from Brazil | Complete valid data | COMPLIANT |
| EUDR-GT-CAT-002 | Valid beef from Argentina | Point geometry | COMPLIANT |
| EUDR-GT-CAT-003 | Invalid - deforested Amazon | Hotspot coordinates | NON_COMPLIANT |
| EUDR-GT-CAT-004 | Invalid - pre-cutoff date | Date: 2019-06-15 | NON_COMPLIANT |
| EUDR-GT-CAT-005 | Invalid - missing coordinates | No geolocation | VALIDATION_ERROR |

### 1.2 Cocoa Tests (5 tests)
| Test ID | Description | Input | Expected Output |
|---------|-------------|-------|-----------------|
| EUDR-GT-COC-001 | Valid cocoa from Ghana | RA certified | COMPLIANT |
| EUDR-GT-COC-002 | Valid cocoa from Cote d'Ivoire | UTZ certified | COMPLIANT |
| EUDR-GT-COC-003 | Invalid - Tai National Park | Protected area | NON_COMPLIANT |
| EUDR-GT-COC-004 | Partial - incomplete traceability | Missing nodes | PENDING_VERIFICATION |
| EUDR-GT-COC-005 | Mixed origin batch | 70% compliant | PARTIAL_COMPLIANCE |

### 1.3 Coffee Tests (5 tests)
| Test ID | Description | Input | Expected Output |
|---------|-------------|-------|-----------------|
| EUDR-GT-COF-001 | Valid Arabica from Colombia | Organic certified | COMPLIANT |
| EUDR-GT-COF-002 | Valid Robusta from Vietnam | Large plantation | COMPLIANT |
| EUDR-GT-COF-003 | Invalid - Ethiopian highlands | Recent clearing | NON_COMPLIANT |
| EUDR-GT-COF-004 | Invalid - self-intersecting polygon | Bad geometry | VALIDATION_ERROR |
| EUDR-GT-COF-005 | Multi-farm shipment | 3 farms verified | COMPLIANT |

### 1.4 Palm Oil Tests (5 tests)
| Test ID | Description | Input | Expected Output |
|---------|-------------|-------|-----------------|
| EUDR-GT-PLM-001 | Valid RSPO certified | Malaysia segregated | COMPLIANT |
| EUDR-GT-PLM-002 | Valid smallholder | Indonesia GPS point | COMPLIANT |
| EUDR-GT-PLM-003 | Invalid - deep peat | >3m peat depth | NON_COMPLIANT |
| EUDR-GT-PLM-004 | Invalid - recent expansion | Post-2020 clearing | NON_COMPLIANT |
| EUDR-GT-PLM-005 | Mass balance traceability | RSPO MB chain | COMPLIANT |

### 1.5 Rubber Tests (5 tests)
| Test ID | Description | Input | Expected Output |
|---------|-------------|-------|-----------------|
| EUDR-GT-RUB-001 | Valid from Thailand | Established plantation | COMPLIANT |
| EUDR-GT-RUB-002 | Valid from Malaysia | FSC certified | COMPLIANT |
| EUDR-GT-RUB-003 | Invalid - Cambodia encroachment | Protected forest | NON_COMPLIANT |
| EUDR-GT-RUB-004 | Mixed compound | 60% natural rubber | APPLICABLE_PORTION |
| EUDR-GT-RUB-005 | Smallholder cooperative | 50 aggregated | COMPLIANT |

### 1.6 Soy Tests (5 tests)
| Test ID | Description | Input | Expected Output |
|---------|-------------|-------|-----------------|
| EUDR-GT-SOY-001 | Valid from Argentina | Pampas region | COMPLIANT |
| EUDR-GT-SOY-002 | Valid RTRS certified | Pre-2020 conversion | COMPLIANT |
| EUDR-GT-SOY-003 | Invalid - Cerrado conversion | 2021 clearing | NON_COMPLIANT |
| EUDR-GT-SOY-004 | Soy meal derivative | Processed product | COMPLIANT |
| EUDR-GT-SOY-005 | Mixed origin shipment | BR + US mixed | REQUIRES_SEGREGATION |

### 1.7 Wood Tests (5 tests)
| Test ID | Description | Input | Expected Output |
|---------|-------------|-------|-----------------|
| EUDR-GT-WOD-001 | Valid FSC timber | Sweden oak | COMPLIANT |
| EUDR-GT-WOD-002 | Valid PEFC pulp | Canada coniferous | COMPLIANT |
| EUDR-GT-WOD-003 | Invalid - Myanmar teak | No documentation | NON_COMPLIANT |
| EUDR-GT-WOD-004 | Recycled content | 80% recycled | PARTIAL_APPLICABLE |
| EUDR-GT-WOD-005 | Invalid - charcoal deforestation | Nigeria clearing | NON_COMPLIANT |

---

## Category 2: Geolocation Validation Tests (25 tests)

### 2.1 Valid Geometry Tests (10 tests)
| Test ID | Description | Geometry Type | Expected |
|---------|-------------|---------------|----------|
| EUDR-GT-GEO-001 | Valid point coordinate | Point | VALID |
| EUDR-GT-GEO-002 | Valid simple polygon | Polygon (4 vertices) | VALID |
| EUDR-GT-GEO-003 | Valid complex polygon | Polygon (100 vertices) | VALID |
| EUDR-GT-GEO-004 | Valid polygon with hole | Polygon (2 rings) | VALID |
| EUDR-GT-GEO-005 | Valid multi-polygon | MultiPolygon (5 parts) | VALID |
| EUDR-GT-GEO-006 | Maximum precision | 8 decimal places | VALID |
| EUDR-GT-GEO-007 | International date line | Lon: 179.999 | VALID |
| EUDR-GT-GEO-008 | Southern hemisphere | Lat: -50.0 | VALID |
| EUDR-GT-GEO-009 | Equator crossing | Lat: 0.0 | VALID |
| EUDR-GT-GEO-010 | Prime meridian | Lon: 0.0 | VALID |

### 2.2 Invalid Geometry Tests (10 tests)
| Test ID | Description | Error Type | Expected |
|---------|-------------|------------|----------|
| EUDR-GT-GEO-011 | Out of bounds longitude | Lon: 200 | ERROR |
| EUDR-GT-GEO-012 | Out of bounds latitude | Lat: -95 | ERROR |
| EUDR-GT-GEO-013 | Unclosed polygon | First != Last | ERROR |
| EUDR-GT-GEO-014 | Self-intersecting polygon | Figure-8 | ERROR |
| EUDR-GT-GEO-015 | Empty geometry | {} | ERROR |
| EUDR-GT-GEO-016 | Insufficient vertices | 2 vertices | ERROR |
| EUDR-GT-GEO-017 | Negative area polygon | Clockwise winding | WARNING |
| EUDR-GT-GEO-018 | Duplicate consecutive points | Same vertex 2x | WARNING |
| EUDR-GT-GEO-019 | NaN coordinates | NaN values | ERROR |
| EUDR-GT-GEO-020 | Infinity coordinates | Infinity values | ERROR |

### 2.3 CRS Transformation Tests (5 tests)
| Test ID | Description | Source CRS | Expected |
|---------|-------------|------------|----------|
| EUDR-GT-CRS-001 | WGS84 passthrough | EPSG:4326 | No transform |
| EUDR-GT-CRS-002 | Web Mercator to WGS84 | EPSG:3857 | Transformed |
| EUDR-GT-CRS-003 | UTM Zone 33N to WGS84 | EPSG:32633 | Transformed |
| EUDR-GT-CRS-004 | Unknown CRS | No CRS specified | Default WGS84 |
| EUDR-GT-CRS-005 | Brazilian SIRGAS | EPSG:4674 | Transformed |

---

## Category 3: Satellite/Deforestation Tests (30 tests)

### 3.1 Forest Cover Detection (10 tests)
| Test ID | Description | Baseline | Current | Expected |
|---------|-------------|----------|---------|----------|
| EUDR-GT-SAT-001 | No change - stable forest | 95% | 94% | NO_DEFORESTATION |
| EUDR-GT-SAT-002 | No change - stable non-forest | 5% | 5% | NOT_APPLICABLE |
| EUDR-GT-SAT-003 | Minor degradation | 90% | 80% | DEGRADATION_DETECTED |
| EUDR-GT-SAT-004 | Major deforestation | 85% | 20% | DEFORESTATION_DETECTED |
| EUDR-GT-SAT-005 | Complete clearing | 95% | 0% | DEFORESTATION_DETECTED |
| EUDR-GT-SAT-006 | Reforestation | 20% | 60% | REFORESTATION |
| EUDR-GT-SAT-007 | Fire-related loss | 80% | 30% | FIRE_LOSS_DETECTED |
| EUDR-GT-SAT-008 | Selective logging | 90% | 70% | DEGRADATION_DETECTED |
| EUDR-GT-SAT-009 | Edge of forest | 50% | 45% | NO_SIGNIFICANT_CHANGE |
| EUDR-GT-SAT-010 | Agricultural mosaic | 40% | 35% | NO_SIGNIFICANT_CHANGE |

### 3.2 Temporal Analysis Tests (10 tests)
| Test ID | Description | Analysis Period | Expected |
|---------|-------------|-----------------|----------|
| EUDR-GT-TMP-001 | Pre-cutoff baseline | Dec 2020 | BASELINE_ESTABLISHED |
| EUDR-GT-TMP-002 | Current verification | Dec 2025 | VERIFIED |
| EUDR-GT-TMP-003 | Change in 2021 | 2020-2021 | CHANGE_POST_CUTOFF |
| EUDR-GT-TMP-004 | Change in 2019 | 2019-2020 | CHANGE_PRE_CUTOFF_OK |
| EUDR-GT-TMP-005 | Seasonal variation | Dry season | SEASONAL_NOT_DEFORESTATION |
| EUDR-GT-TMP-006 | Cloud gap analysis | 60% cloud | INSUFFICIENT_DATA |
| EUDR-GT-TMP-007 | Multi-year trend | 2020-2025 | TREND_ANALYZED |
| EUDR-GT-TMP-008 | Recent alert | Last 30 days | ALERT_DETECTED |
| EUDR-GT-TMP-009 | Historical archive | 2015-2020 | HISTORICAL_VERIFIED |
| EUDR-GT-TMP-010 | Exactly cutoff date | 2020-12-31 | CUTOFF_BASELINE |

### 3.3 Multi-Source Fusion Tests (10 tests)
| Test ID | Description | Sources | Expected |
|---------|-------------|---------|----------|
| EUDR-GT-FUS-001 | Sentinel-2 only | S2 | SINGLE_SOURCE |
| EUDR-GT-FUS-002 | Landsat only | L8/L9 | SINGLE_SOURCE |
| EUDR-GT-FUS-003 | GFW alerts only | GFW | ALERT_BASED |
| EUDR-GT-FUS-004 | S2 + Landsat fusion | S2, L8 | FUSED_RESULT |
| EUDR-GT-FUS-005 | All sources | S2, L8, GFW | COMPREHENSIVE |
| EUDR-GT-FUS-006 | Conflicting results | S2: clear, GFW: alert | MANUAL_REVIEW |
| EUDR-GT-FUS-007 | Radar backup | S1 SAR | CLOUD_PENETRATION |
| EUDR-GT-FUS-008 | Resolution difference | 10m vs 30m | WEIGHTED_FUSION |
| EUDR-GT-FUS-009 | Temporal interpolation | Gap filled | INTERPOLATED |
| EUDR-GT-FUS-010 | Confidence scoring | Multiple sources | HIGH_CONFIDENCE |

---

## Category 4: Supply Chain Traceability Tests (25 tests)

### 4.1 Chain of Custody Tests (10 tests)
| Test ID | Description | Chain Length | Expected |
|---------|-------------|--------------|----------|
| EUDR-GT-COC-001 | Complete chain - 3 nodes | Producer->Processor->Exporter | 100% TRACED |
| EUDR-GT-COC-002 | Complete chain - 5 nodes | Full supply chain | 100% TRACED |
| EUDR-GT-COC-003 | Missing intermediary | Gap at processor | TRACEABILITY_GAP |
| EUDR-GT-COC-004 | Aggregation point | 10 producers merged | MASS_BALANCE |
| EUDR-GT-COC-005 | Processing transformation | Raw to refined | CONVERSION_VERIFIED |
| EUDR-GT-COC-006 | Multi-tier supplier | 4 levels deep | FULL_CHAIN |
| EUDR-GT-COC-007 | Cross-border transit | 3 countries | TRANSIT_VERIFIED |
| EUDR-GT-COC-008 | Re-export scenario | Import then export | DUAL_DECLARATION |
| EUDR-GT-COC-009 | Blend of origins | 2 countries | SEGREGATION_REQUIRED |
| EUDR-GT-COC-010 | Single origin | Direct export | SIMPLIFIED_CHAIN |

### 4.2 Certification Verification Tests (8 tests)
| Test ID | Description | Certification | Expected |
|---------|-------------|---------------|----------|
| EUDR-GT-CRT-001 | Valid FSC | Active certificate | CERT_VALID |
| EUDR-GT-CRT-002 | Valid RSPO | Active certificate | CERT_VALID |
| EUDR-GT-CRT-003 | Expired certificate | Past valid_until | CERT_EXPIRED |
| EUDR-GT-CRT-004 | Revoked certificate | Status: revoked | CERT_REVOKED |
| EUDR-GT-CRT-005 | Scope mismatch | Wrong product | CERT_SCOPE_MISMATCH |
| EUDR-GT-CRT-006 | Multiple certifications | FSC + RA | MULTI_CERT |
| EUDR-GT-CRT-007 | Certificate holder mismatch | Different entity | CERT_HOLDER_MISMATCH |
| EUDR-GT-CRT-008 | Online verification | API check | CERT_VERIFIED_ONLINE |

### 4.3 Document Verification Tests (7 tests)
| Test ID | Description | Document Type | Expected |
|---------|-------------|---------------|----------|
| EUDR-GT-DOC-001 | Valid invoice | Complete data | DOC_VALID |
| EUDR-GT-DOC-002 | Valid bill of lading | Shipping doc | DOC_VALID |
| EUDR-GT-DOC-003 | Missing required doc | No invoice | DOC_MISSING |
| EUDR-GT-DOC-004 | Hash verification | SHA-256 match | HASH_VALID |
| EUDR-GT-DOC-005 | Hash mismatch | Tampered file | HASH_MISMATCH |
| EUDR-GT-DOC-006 | Quantity reconciliation | Input vs output | QUANTITY_VALID |
| EUDR-GT-DOC-007 | Date consistency | Sequential dates | DATES_CONSISTENT |

---

## Category 5: DDS Generation Tests (20 tests)

### 5.1 Schema Validation Tests (10 tests)
| Test ID | Description | Input | Expected |
|---------|-------------|-------|----------|
| EUDR-GT-DDS-001 | Complete valid DDS | All required fields | SCHEMA_VALID |
| EUDR-GT-DDS-002 | Missing operator info | No EORI | SCHEMA_ERROR |
| EUDR-GT-DDS-003 | Missing commodity data | No CN code | SCHEMA_ERROR |
| EUDR-GT-DDS-004 | Invalid date format | DD-MM-YYYY | SCHEMA_ERROR |
| EUDR-GT-DDS-005 | Invalid quantity | Negative value | SCHEMA_ERROR |
| EUDR-GT-DDS-006 | Invalid country code | 3-letter code | SCHEMA_ERROR |
| EUDR-GT-DDS-007 | Extra fields | Additional data | SCHEMA_VALID |
| EUDR-GT-DDS-008 | Empty array | No supply chain | SCHEMA_ERROR |
| EUDR-GT-DDS-009 | Null values | Null operator | SCHEMA_ERROR |
| EUDR-GT-DDS-010 | Unicode characters | Non-ASCII names | SCHEMA_VALID |

### 5.2 Compliance Logic Tests (5 tests)
| Test ID | Description | Scenario | Expected |
|---------|-------------|----------|----------|
| EUDR-GT-LOG-001 | Low risk pathway | Low risk country | STANDARD_DD |
| EUDR-GT-LOG-002 | Standard risk pathway | Standard country | ENHANCED_DD |
| EUDR-GT-LOG-003 | High risk pathway | High risk country | REINFORCED_DD |
| EUDR-GT-LOG-004 | SME simplified | Qualifying SME | SME_SIMPLIFIED |
| EUDR-GT-LOG-005 | Operator vs Trader | Different roles | CORRECT_PATHWAY |

### 5.3 Output Generation Tests (5 tests)
| Test ID | Description | Output | Expected |
|---------|-------------|--------|----------|
| EUDR-GT-OUT-001 | JSON output | EU schema JSON | VALID_JSON |
| EUDR-GT-OUT-002 | Reference number | New submission | UNIQUE_REF |
| EUDR-GT-OUT-003 | Amendment | Update existing | LINKED_REF |
| EUDR-GT-OUT-004 | Multi-product | Multiple commodities | SEPARATE_DECL |
| EUDR-GT-OUT-005 | Batch submission | 100 products | PERFORMANCE_OK |

---

## Category 6: Risk Assessment Tests (20 tests)

### 6.1 Risk Score Calculation (10 tests)
| Test ID | Description | Factors | Expected Score |
|---------|-------------|---------|----------------|
| EUDR-GT-RSK-001 | All favorable | Low risk, certified | 0-20 (LOW) |
| EUDR-GT-RSK-002 | Mixed factors | Some risk factors | 21-60 (MEDIUM) |
| EUDR-GT-RSK-003 | Multiple risks | High risk country, no cert | 61-100 (HIGH) |
| EUDR-GT-RSK-004 | Country weight | High risk country | +30 points |
| EUDR-GT-RSK-005 | Certification benefit | FSC certified | -20 points |
| EUDR-GT-RSK-006 | Historical violations | Prior issues | +40 points |
| EUDR-GT-RSK-007 | Satellite anomaly | Deforestation detected | +50 points |
| EUDR-GT-RSK-008 | Doc quality | Incomplete docs | +25 points |
| EUDR-GT-RSK-009 | New supplier | First transaction | +15 points |
| EUDR-GT-RSK-010 | Aggregated risk | Multiple commodities | WEIGHTED_AVG |

### 6.2 Risk Mitigation Tests (10 tests)
| Test ID | Description | Mitigation | Expected |
|---------|-------------|------------|----------|
| EUDR-GT-MIT-001 | Certification reduces risk | FSC added | RISK_REDUCED |
| EUDR-GT-MIT-002 | Satellite verification | Clear satellite | RISK_REDUCED |
| EUDR-GT-MIT-003 | Complete documentation | All docs | RISK_REDUCED |
| EUDR-GT-MIT-004 | Long relationship | 5+ years | RISK_REDUCED |
| EUDR-GT-MIT-005 | Third-party audit | Recent audit | RISK_REDUCED |
| EUDR-GT-MIT-006 | Geofencing | GPS verified | RISK_REDUCED |
| EUDR-GT-MIT-007 | Mass balance | CoC system | PARTIAL_MITIGATION |
| EUDR-GT-MIT-008 | Insurance | Compliance insurance | NO_RISK_IMPACT |
| EUDR-GT-MIT-009 | Training evidence | Staff trained | MINIMAL_IMPACT |
| EUDR-GT-MIT-010 | Combined mitigations | Multiple factors | CUMULATIVE_REDUCTION |

---

## Category 7: Edge Cases (25 tests)

### 7.1 Boundary Conditions (10 tests)
| Test ID | Description | Condition | Expected |
|---------|-------------|-----------|----------|
| EUDR-GT-EDG-001 | Exactly cutoff date | 2020-12-31 23:59:59 | COMPLIANT |
| EUDR-GT-EDG-002 | One second after | 2021-01-01 00:00:00 | REQUIRES_VERIFICATION |
| EUDR-GT-EDG-003 | Minimum plot size | 0.01 hectares | VALID |
| EUDR-GT-EDG-004 | Maximum plot size | 10,000 hectares | WARNING |
| EUDR-GT-EDG-005 | International date line | Lon: 179.999999 | VALID |
| EUDR-GT-EDG-006 | Polar coordinates | Lat: 89.999999 | VALID_UNUSUAL |
| EUDR-GT-EDG-007 | Zero quantity | 0 kg | ERROR |
| EUDR-GT-EDG-008 | Maximum quantity | 1,000,000 tonnes | VALID |
| EUDR-GT-EDG-009 | Empty batch | No products | ERROR |
| EUDR-GT-EDG-010 | Single product | 1 item | VALID |

### 7.2 Special Cases (10 tests)
| Test ID | Description | Scenario | Expected |
|---------|-------------|----------|----------|
| EUDR-GT-SPC-001 | Derived product | 35% cocoa chocolate | APPLICABLE_35% |
| EUDR-GT-SPC-002 | Recycled material | 80% recycled wood | APPLICABLE_20% |
| EUDR-GT-SPC-003 | Wild harvest | Forest coffee | SPECIAL_RULES |
| EUDR-GT-SPC-004 | Indigenous territory | With consent | VALID_WITH_CONSENT |
| EUDR-GT-SPC-005 | Diplomatic goods | Embassy import | EXEMPT |
| EUDR-GT-SPC-006 | Personal use | Under 2kg | EXEMPT |
| EUDR-GT-SPC-007 | Transit goods | In transit only | NOT_APPLICABLE |
| EUDR-GT-SPC-008 | Re-import | Same goods return | SPECIAL_HANDLING |
| EUDR-GT-SPC-009 | Samples | Trade samples | EXEMPT_UNDER_LIMIT |
| EUDR-GT-SPC-010 | Force majeure | Natural disaster | EXTENSION |

### 7.3 Error Recovery (5 tests)
| Test ID | Description | Error | Recovery |
|---------|-------------|-------|----------|
| EUDR-GT-ERR-001 | API timeout | 30s delay | Retry with backoff |
| EUDR-GT-ERR-002 | Malformed response | Invalid JSON | Graceful error |
| EUDR-GT-ERR-003 | Rate limit | >100 req/min | Exponential backoff |
| EUDR-GT-ERR-004 | Partial data | Incomplete payload | Clear error message |
| EUDR-GT-ERR-005 | Concurrent update | Race condition | Optimistic locking |

---

## Category 8: Integration/E2E Tests (20 tests)

### 8.1 Full Workflow Tests (10 tests)
| Test ID | Description | Workflow | Expected |
|---------|-------------|----------|----------|
| EUDR-GT-E2E-001 | Complete compliant flow | Full process | COMPLIANT_DDS_GENERATED |
| EUDR-GT-E2E-002 | Complete non-compliant flow | Deforestation found | NON_COMPLIANT_BLOCKED |
| EUDR-GT-E2E-003 | Amendment workflow | Update existing | AMENDMENT_ACCEPTED |
| EUDR-GT-E2E-004 | Withdrawal workflow | Cancel DDS | WITHDRAWAL_CONFIRMED |
| EUDR-GT-E2E-005 | Multi-commodity | 3 commodities | ALL_PROCESSED |
| EUDR-GT-E2E-006 | High volume | 100 products | BATCH_COMPLETE |
| EUDR-GT-E2E-007 | Real-time alert | New deforestation | ALERT_TRIGGERED |
| EUDR-GT-E2E-008 | Re-verification | Alert response | RE_VERIFIED |
| EUDR-GT-E2E-009 | Async processing | Webhook callback | CALLBACK_RECEIVED |
| EUDR-GT-E2E-010 | Report generation | Full audit | REPORT_GENERATED |

### 8.2 Performance Tests (5 tests)
| Test ID | Description | Load | Expected |
|---------|-------------|------|----------|
| EUDR-GT-PRF-001 | Single request latency | 1 request | <2s |
| EUDR-GT-PRF-002 | Concurrent requests | 100 parallel | <5s avg |
| EUDR-GT-PRF-003 | Satellite analysis | Complex polygon | <30s |
| EUDR-GT-PRF-004 | Batch processing | 1000 products | <5 min |
| EUDR-GT-PRF-005 | Peak load | 500 req/min | No degradation |

### 8.3 API Tests (5 tests)
| Test ID | Description | Endpoint | Expected |
|---------|-------------|----------|----------|
| EUDR-GT-API-001 | Health check | /health | 200 OK |
| EUDR-GT-API-002 | Validation endpoint | /validate | Valid response |
| EUDR-GT-API-003 | DDS endpoint | /generate/dds | DDS returned |
| EUDR-GT-API-004 | Authentication | Invalid token | 401 Unauthorized |
| EUDR-GT-API-005 | Rate limiting | Exceed limit | 429 Too Many Requests |

---

## Test Execution Plan

### Phase 1: Week 1 (Dec 4-10)
- Commodity validation tests (35 tests)
- Basic geolocation tests (15 tests)
- **Total: 50 tests**

### Phase 2: Week 2 (Dec 11-17)
- Remaining geolocation tests (10 tests)
- Satellite/deforestation tests (30 tests)
- Supply chain tests (25 tests)
- **Total: 65 tests**

### Phase 3: Week 3 (Dec 18-24)
- DDS generation tests (20 tests)
- Risk assessment tests (20 tests)
- Edge cases (25 tests)
- **Total: 65 tests**

### Phase 4: Week 4 (Dec 25-30)
- Integration/E2E tests (20 tests)
- Final validation and fixes
- **Total: 20 tests**

---

## Success Criteria

- **Pass Rate:** 100% of golden tests must pass
- **Coverage:** All 7 commodities covered
- **Performance:** P95 latency <2s for validation, <30s for satellite
- **Accuracy:** >99.9% geolocation accuracy, >90% deforestation detection

---

## Test Data Management

### Fixture Files
- `commodity_fixtures.yaml` - All commodity test cases
- `geolocation_fixtures.yaml` - Geometry test cases
- `satellite_fixtures.yaml` - Satellite data test cases
- `supply_chain_fixtures.yaml` - CoC test cases
- `dds_fixtures.yaml` - DDS generation test cases

### Test Execution
```bash
# Run all golden tests
pytest tests/golden/ -v --tb=short

# Run specific category
pytest tests/golden/test_commodity.py -v

# Run with coverage
pytest tests/golden/ --cov=eudr_agent --cov-report=html
```

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | GL-Test-Engineer | Initial golden tests plan |
