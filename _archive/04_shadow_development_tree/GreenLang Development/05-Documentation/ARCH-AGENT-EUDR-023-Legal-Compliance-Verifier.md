# AGENT-EUDR-023: Legal Compliance Verifier -- Technical Architecture Specification

## Document Info

| Field | Value |
|-------|-------|
| **Document ID** | ARCH-AGENT-EUDR-023 |
| **Agent ID** | GL-EUDR-LCV-023 |
| **Component** | Legal Compliance Verifier Agent |
| **Category** | EUDR Regulatory Agent -- Legal Compliance Verification |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Architecture Specification |
| **Author** | GL-AppArchitect |
| **Date** | 2026-03-10 |
| **Regulation** | Regulation (EU) 2023/1115 -- EUDR, Article 2(40), Articles 3, 8, 10, 11, 29 |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |
| **DB Migration** | V111 |
| **Metric Prefix** | `gl_eudr_lcv_` |
| **Config Prefix** | `GL_EUDR_LCV_` |
| **API Prefix** | `/v1/eudr-lcv` |
| **RBAC Prefix** | `eudr-lcv:` |

---

## 1. Executive Summary

### 1.1 Purpose

AGENT-EUDR-023 Legal Compliance Verifier is a specialized compliance agent that verifies whether commodity production and trade comply with all "relevant legislation" of the country of production as defined by EUDR Article 2(40). The regulation mandates that products placed on the EU market must be produced in accordance with the relevant legislation of the country of production, encompassing 8 distinct legal categories: land use rights, environmental protection, forest management, labour rights, human rights, tax and royalty obligations, trade and customs regulations, and third-party rights (including indigenous peoples).

### 1.2 Regulatory Driver

EUDR Article 2(40) defines "relevant legislation" as the laws applicable in the country of production concerning:

1. **Land use rights** -- land tenure, ownership, concession, and use permits
2. **Environmental protection** -- environmental impact assessment, pollution control, biodiversity
3. **Forest-related rules** -- forest management, harvesting permits, reforestation obligations
4. **Third-party rights** -- indigenous peoples' rights, customary tenure, FPIC requirements
5. **Labour rights** -- ILO core conventions, forced labour prohibition, child labour, OSH
6. **Tax and royalty obligations** -- forestry taxes, export duties, royalties, transfer pricing
7. **Trade and customs** -- import/export permits, CITES, trade sanctions, rules of origin
8. **Anti-corruption** -- bribery, facilitation payments, public procurement integrity

Non-compliance with any of these 8 categories renders the commodity non-compliant with EUDR, exposing EU operators to penalties of up to 4% of annual EU turnover.

### 1.3 Key Differentiators

- **8-category legal framework coverage** across 27 EUDR commodity-producing countries
- **Zero-hallucination deterministic compliance scoring** -- all legal checks use rule-based engines with lookup tables, never LLM inference
- **Real-time legal database integration** -- FAO LEX, ECOLEX, national legislation portals
- **Certification scheme validation** -- FSC, PEFC, RSPO, Rainforest Alliance, ISCC with chain-of-custody verification
- **Red flag pattern detection** -- 40+ corruption and illegality indicators with deterministic scoring
- **Third-party audit integration** -- structured parsing of audit reports from 6 major certification bodies
- **Multi-format compliance reporting** -- PDF, JSON, XBRL, XML for 8 report types

### 1.4 Performance Targets

| Metric | Target |
|--------|--------|
| Single compliance check (1 supplier, 1 category) | < 500ms |
| Full 8-category assessment (1 supplier) | < 5s |
| Batch assessment (1,000 suppliers) | < 60s |
| Legal framework database query | < 200ms |
| Document verification | < 2s per document |
| Certification validation | < 1s per certificate |
| Red flag scan (1 supplier) | < 3s |
| Report generation (full compliance report) | < 10s |
| API p99 latency | < 800ms |
| Redis cache hit rate target | > 70% |

### 1.5 Development Estimates

| Phase | Scope | Duration | Engineers |
|-------|-------|----------|-----------|
| Phase 1 | Core engines (1-3), models, config, provenance | 2 weeks | 2 |
| Phase 2 | Engines 4-7, API routes, auth integration | 2 weeks | 2 |
| Phase 3 | DB migration, reference data, integration | 1 week | 2 |
| Phase 4 | Testing (unit + integration), monitoring | 1 week | 2 |
| **Total** | **Complete agent** | **6 weeks** | **2 engineers** |

### 1.6 Estimated Output

- ~42 files (agent code + API + reference data)
- ~40K lines of code
- ~500+ tests
- V111 database migration
- 1 Grafana dashboard

---

## 2. Module Structure

### 2.1 Directory Layout

```
greenlang/agents/eudr/legal_compliance_verifier/
    __init__.py
    config.py                              # Centralized configuration (GL_EUDR_LCV_ prefix)
    models.py                              # Pydantic v2 data models (30+ models)
    metrics.py                             # 20 Prometheus metrics (gl_eudr_lcv_ prefix)
    provenance.py                          # SHA-256 provenance tracking
    setup.py                               # Facade orchestrating all 7 engines
    #
    # === 7 Engines (top-level modules) ===
    #
    legal_framework_database_engine.py     # Engine 1: Country-specific legal frameworks
    document_verification_engine.py        # Engine 2: Permit/license/certificate verification
    certification_scheme_validator.py      # Engine 3: FSC/RSPO/PEFC/RA/ISCC validation
    red_flag_detection_engine.py           # Engine 4: Corruption & illegality indicators
    country_compliance_checker.py          # Engine 5: Per-country legal requirement verification
    third_party_audit_engine.py            # Engine 6: Audit report parsing & verification
    compliance_reporting_engine.py         # Engine 7: Multi-format report generation
    #
    # === API Layer ===
    #
    api/
        __init__.py
        router.py                          # Main router (/v1/eudr-lcv), sub-router aggregation
        dependencies.py                    # FastAPI dependencies (service injection, auth)
        schemas.py                         # API-specific request/response schemas
        framework_routes.py                # Legal framework CRUD & query (5 endpoints)
        document_routes.py                 # Document verification & management (5 endpoints)
        certification_routes.py            # Certification scheme validation (4 endpoints)
        red_flag_routes.py                 # Red flag detection & alerts (4 endpoints)
        compliance_routes.py               # Country compliance checking (5 endpoints)
        audit_routes.py                    # Third-party audit integration (4 endpoints)
        report_routes.py                   # Compliance report generation (4 endpoints)
        batch_routes.py                    # Batch processing operations (3 endpoints)
        admin_routes.py                    # Admin & health endpoints (3 endpoints)
    #
    # === Reference Data ===
    #
    reference_data/
        __init__.py
        legal_frameworks.py                # 27-country legal framework database (8 categories)
        certification_schemes.py           # FSC/PEFC/RSPO/RA/ISCC rule definitions
        red_flag_patterns.py               # 40+ red flag indicator definitions
        legislation_categories.py          # EUDR Article 2(40) category taxonomy
```

### 2.2 Test Directory Layout

```
tests/agents/eudr/legal_compliance_verifier/
    __init__.py
    conftest.py                            # Shared fixtures, mock services, test data
    test_legal_framework_database_engine.py
    test_document_verification_engine.py
    test_certification_scheme_validator.py
    test_red_flag_detection_engine.py
    test_country_compliance_checker.py
    test_third_party_audit_engine.py
    test_compliance_reporting_engine.py
    test_models.py
    test_config.py
    test_provenance.py
```

### 2.3 Deployment Artifacts

```
deployment/
    database/migrations/sql/
        V111__agent_eudr_legal_compliance_verifier.sql
    monitoring/dashboards/
        eudr-legal-compliance-verifier.json
```

---

## 3. Engine Specifications

### 3.1 Engine 1: Legal Framework Database Engine

**File:** `legal_framework_database_engine.py`
**Purpose:** Maintains and queries country-specific legal frameworks for all 8 EUDR Article 2(40) legislation categories across 27 commodity-producing countries.

**Responsibilities:**
- Store and index legal frameworks by country, category, and commodity
- Query applicable laws for a given country-commodity-category triple
- Track law enactment dates, amendment history, and current enforcement status
- Integrate with external legal databases (FAO LEX, ECOLEX)
- Map laws to EUDR requirements with deterministic applicability scoring
- Detect legal framework gaps (countries with incomplete legislation)

**Key Data Sources (6):**
1. FAO FAOLEX -- food, agriculture, and forestry legislation (194 countries)
2. ECOLEX (IUCN/UNEP/FAO) -- environmental law (195 countries)
3. ILO NATLEX -- labour legislation (187 countries)
4. National legislation portals (27 EUDR countries)
5. EU Official Journal (EUDR text, delegated acts, implementing acts)
6. WTO Legal Texts -- trade and customs regulations

**Countries Covered (27):**
Brazil, Indonesia, Colombia, Peru, Cote d'Ivoire, Ghana, Cameroon, Democratic Republic of Congo, Republic of Congo, Gabon, Malaysia, Papua New Guinea, Ecuador, Bolivia, Paraguay, Honduras, Guatemala, Nicaragua, Myanmar, Laos, Vietnam, Thailand, India, Ethiopia, Uganda, Tanzania, Nigeria.

**Legislation Categories (8, per Article 2(40)):**

| # | Category | Subcategories | Laws per Country (avg) |
|---|----------|---------------|----------------------|
| 1 | Land use rights | Tenure, ownership, concessions, zoning | 8-15 |
| 2 | Environmental protection | EIA, pollution, biodiversity, water | 10-20 |
| 3 | Forest-related rules | Harvesting permits, reforestation, REDD+ | 5-12 |
| 4 | Third-party rights | Indigenous, customary tenure, FPIC | 3-8 |
| 5 | Labour rights | ILO conventions, child labour, OSH | 6-10 |
| 6 | Tax and royalty | Forest taxes, export duties, royalties | 4-8 |
| 7 | Trade and customs | Import/export, CITES, sanctions | 5-10 |
| 8 | Anti-corruption | Bribery, procurement, transparency | 3-6 |

**Zero-Hallucination Approach:**
- All legal framework data stored as structured records with citation references
- Applicability determination uses deterministic rule matching (country + commodity + category)
- No LLM used for legal interpretation -- all compliance rules are pre-coded lookup tables
- Every query result includes provenance hash linking to source legislation

**Estimated Lines of Code:** 2,800-3,200

### 3.2 Engine 2: Document Verification Engine

**File:** `document_verification_engine.py`
**Purpose:** Verifies permits, licenses, certificates, and other legal documents required for EUDR compliance.

**Responsibilities:**
- Verify document authenticity via checksums, signatures, issuer validation
- Check document validity periods (issue date, expiry, renewal status)
- Monitor expiring documents with configurable advance warning (30/60/90 days)
- Validate document type against country-specific requirements
- Cross-reference documents against issuing authority registries
- Track document chains (original -> renewal -> amendment)

**Document Types Verified (12):**
1. Forest harvesting permits/concession licenses
2. Environmental impact assessment (EIA) approvals
3. Land title deeds and ownership certificates
4. Export permits and phytosanitary certificates
5. CITES permits (for regulated timber species)
6. Labour compliance certificates
7. Tax clearance certificates
8. Certificate of origin documents
9. Transport and chain-of-custody documents
10. Indigenous community consent records (FPIC)
11. Reforestation obligation compliance certificates
12. Anti-corruption declaration/compliance certificates

**Verification Pipeline:**
```
Document Input -> Format Validation -> Issuer Verification -> Validity Check
    -> Cross-Reference Check -> Expiry Monitoring -> Compliance Score -> Provenance
```

**Validity States (6):**
- `VALID` -- document current and verified
- `EXPIRED` -- document past expiry date
- `EXPIRING_SOON` -- within configurable warning period
- `SUSPENDED` -- temporarily invalidated by issuing authority
- `REVOKED` -- permanently invalidated
- `UNVERIFIABLE` -- cannot confirm authenticity

**Zero-Hallucination Approach:**
- Document validity is a deterministic date comparison (no LLM)
- Issuer verification uses pre-registered authority lookup tables
- Checksum/signature verification uses cryptographic functions
- All verification results include step-by-step audit trail

**Estimated Lines of Code:** 3,000-3,500

### 3.3 Engine 3: Certification Scheme Validator

**File:** `certification_scheme_validator.py`
**Purpose:** Validates certifications from recognized sustainability schemes (FSC, PEFC, RSPO, Rainforest Alliance, ISCC) against EUDR requirements.

**Certification Schemes Supported (5 + sub-schemes):**

| Scheme | Sub-schemes | Commodities | CoC Model |
|--------|-------------|-------------|-----------|
| FSC | FM, CoC, CW | Wood, paper, rubber | Transfer, Percentage, Credit |
| PEFC | SFM, CoC | Wood, paper | Physical separation, Percentage |
| RSPO | P&C, SCC, IS | Palm oil | Identity Preserved, Segregated, Mass Balance |
| Rainforest Alliance | SA, CoC | Cocoa, coffee, tea | Segregated, Mass Balance |
| ISCC | ISCC EU, ISCC PLUS | Soy, palm, biomass | Physical segregation, Mass balance |

**Validation Checks per Certificate (10):**
1. Certificate number format validation (scheme-specific regex)
2. Issuing certification body (CB) accreditation verification
3. Certificate scope validation (commodities, sites, operations)
4. Certificate validity period verification
5. Chain-of-custody model compliance
6. Annual surveillance audit status
7. Non-conformity/corrective action status
8. Suspended/withdrawn certificate check against scheme databases
9. EUDR-equivalence mapping (which EUDR requirements the cert satisfies)
10. Multi-site certificate scope validation

**External API Integrations:**
- FSC Certificate Database API (info.fsc.org)
- PEFC Certificate Search (pefc.org)
- RSPO PalmTrace API (rspo.org)
- Rainforest Alliance Certification Portal
- ISCC Certification Database

**Zero-Hallucination Approach:**
- Certificate validation rules are deterministic pattern matching
- EUDR-equivalence mappings are pre-defined lookup tables (not LLM-generated)
- All scheme-specific rules codified from official scheme standards
- Verification results include direct links to scheme database records

**Estimated Lines of Code:** 3,200-3,800

### 3.4 Engine 4: Red Flag Detection Engine

**File:** `red_flag_detection_engine.py`
**Purpose:** Detects indicators of illegal activity, corruption, and non-compliance using 40+ deterministic red flag patterns.

**Red Flag Categories (6):**

| Category | Indicators | Weight Range |
|----------|-----------|--------------|
| Corruption & Bribery | 8 patterns | 0.10-0.25 |
| Illegal Logging | 7 patterns | 0.15-0.30 |
| Land Rights Violations | 6 patterns | 0.10-0.25 |
| Labour Violations | 6 patterns | 0.10-0.20 |
| Tax Evasion | 5 patterns | 0.05-0.15 |
| Document Fraud | 8 patterns | 0.10-0.25 |

**Red Flag Indicators (40):**

*Corruption & Bribery (8):*
1. Supplier located in country with CPI < 30
2. Concession awarded without competitive tender
3. Permit issued in < 50% of standard processing time
4. Multiple permits from same official in short timeframe
5. Politically exposed person (PEP) in ownership chain
6. Supplier sanctioned by OFAC/EU/UN sanctions lists
7. Beneficial ownership obscured through shell companies
8. Facilitation payment patterns in transaction records

*Illegal Logging (7):*
9. Harvest volume exceeds concession permit limits
10. Species harvested not listed in concession permit
11. Harvest location outside permitted boundaries
12. Transport documents inconsistent with harvest records
13. CITES-listed species without CITES permit
14. Night-time satellite activity in forest concession
15. Road construction in previously unlogged forest

*Land Rights Violations (6):*
16. Production on disputed land (overlapping claims)
17. No FPIC documentation for indigenous territory overlap
18. Active land rights litigation involving supplier
19. Forced displacement reported in sourcing area
20. Customary tenure not recognized despite evidence
21. Community grievance filed against supplier

*Labour Violations (6):*
22. ILO core convention violation reports for supplier
23. Child labour indicators (education enrollment gaps)
24. Forced labour indicators (debt bondage, withheld documents)
25. OSH violation citations from labour inspectorate
26. Below minimum wage payment patterns
27. Excessive working hours (> 60h/week patterns)

*Tax Evasion (5):*
28. Transfer pricing anomalies (price < 70% market rate)
29. Royalty underpayment patterns
30. Tax clearance certificate expired or missing
31. Revenue inconsistent with declared production volume
32. Export value significantly below import declaration

*Document Fraud (8):*
33. Document dates inconsistent (issue after expiry)
34. Issuing authority not in registered authority database
35. Certificate number fails format validation
36. Multiple documents with sequential serial numbers from same batch
37. Digital signature verification failure
38. Document metadata inconsistent with content
39. Duplicate document submitted for different shipments
40. Document template mismatch with known issuing authority format

**Scoring Methodology:**
- Each red flag has a base severity weight (0.05 to 0.30)
- Country-specific multiplier applied (1.0 to 2.0 based on CPI/WGI)
- Commodity-specific multiplier applied (1.0 to 1.5 based on commodity risk)
- Aggregate score = weighted sum, normalized to 0-100
- Risk classification: LOW (0-25), MODERATE (26-50), HIGH (51-75), CRITICAL (76-100)
- All scoring is deterministic -- identical inputs always produce identical outputs

**Zero-Hallucination Approach:**
- Red flag patterns are deterministic rule evaluations (threshold comparisons, pattern matching)
- No LLM used for risk classification -- all thresholds are pre-configured
- Scoring formula: `score = sum(flag_weight * country_multiplier * commodity_multiplier) / max_possible * 100`
- Every triggered flag includes the specific data point and threshold that triggered it

**Estimated Lines of Code:** 3,500-4,000

### 3.5 Engine 5: Country Compliance Checker

**File:** `country_compliance_checker.py`
**Purpose:** Performs per-country legal requirement verification across all 8 legislation categories for a given supplier-commodity pair.

**Responsibilities:**
- Determine applicable legal requirements for country-commodity pair
- Check supplier compliance against each requirement
- Generate compliance gap analysis (requirements met vs. unmet)
- Produce country-specific compliance scorecards (0-100 per category)
- Track compliance status changes over time
- Integrate country risk data from EUDR-016 (Country Risk Evaluator)

**Compliance Assessment Process:**
```
1. Identify country + commodity -> Retrieve applicable legal requirements
2. For each of 8 categories:
   a. List specific legal requirements (from Engine 1)
   b. Check available evidence (documents from Engine 2, certs from Engine 3)
   c. Apply deterministic compliance rules
   d. Score: COMPLIANT / PARTIALLY_COMPLIANT / NON_COMPLIANT / INSUFFICIENT_DATA
3. Calculate category scores (0-100) and overall compliance score
4. Generate gap analysis listing unmet requirements
5. Produce compliance assessment with provenance chain
```

**Compliance States (4):**
- `COMPLIANT` (score 80-100) -- all requirements met with evidence
- `PARTIALLY_COMPLIANT` (score 50-79) -- some requirements met, gaps identified
- `NON_COMPLIANT` (score 0-49) -- critical requirements unmet
- `INSUFFICIENT_DATA` -- cannot assess due to missing information

**Country-Specific Rule Sets:**
Each country has a rule set defining:
- Mandatory documents per commodity type
- Required certifications and permits
- Minimum compliance thresholds per category
- Country-specific legal peculiarities (e.g., Brazil IBAMA requirements, Indonesia SVLK)
- Enforcement intensity classification (HIGH / MODERATE / LOW)

**Zero-Hallucination Approach:**
- Compliance determination is rule-based: requirement checklist with boolean verification
- Scoring: `category_score = (requirements_met / total_requirements) * 100`
- No LLM interpretation of legal requirements
- All rules traceable to specific legislation citations

**Estimated Lines of Code:** 3,000-3,500

### 3.6 Engine 6: Third-Party Audit Integration Engine

**File:** `third_party_audit_engine.py`
**Purpose:** Parses, extracts findings from, and verifies third-party audit reports from certification bodies and independent auditors.

**Supported Audit Report Sources (6):**
1. FSC audit reports (FM, CoC, CW evaluations)
2. PEFC audit reports (SFM, CoC assessments)
3. RSPO audit reports (P&C, SCC assessments)
4. Independent EUDR due diligence audits
5. Government forestry inspection reports
6. ISO 14001 / ISO 45001 audit reports

**Audit Report Processing Pipeline:**
```
Audit Report Upload -> Format Detection -> Structure Extraction
    -> Finding Classification -> Severity Assessment -> Evidence Mapping
    -> Verification Cross-Check -> Compliance Impact -> Provenance
```

**Finding Categories (5):**
- `MAJOR_NON_CONFORMITY` -- requires corrective action, potential suspension
- `MINOR_NON_CONFORMITY` -- requires corrective action within timeframe
- `OBSERVATION` -- area for improvement, no corrective action required
- `POSITIVE_PRACTICE` -- good practice noted
- `NOT_APPLICABLE` -- requirement not applicable to scope

**Structured Extraction Fields (per finding):**
- Finding reference number
- Applicable requirement (standard clause or legal reference)
- Description of non-conformity
- Evidence observed
- Root cause (if identified)
- Corrective action required
- Corrective action deadline
- Follow-up status (OPEN / IN_PROGRESS / CLOSED / OVERDUE)

**Zero-Hallucination Approach:**
- Audit finding classification uses structured templates (not free-text LLM analysis)
- Severity scoring is deterministic based on finding category and affected requirement
- LLM may be used ONLY for entity extraction from unstructured report text (non-critical path)
- All extracted findings require human confirmation before compliance impact assessment
- Critical path calculations (compliance scoring) never use LLM output

**Estimated Lines of Code:** 2,800-3,200

### 3.7 Engine 7: Compliance Reporting Engine

**File:** `compliance_reporting_engine.py`
**Purpose:** Generates compliance reports in multiple formats for regulatory submission, internal governance, and audit purposes.

**Report Types (8):**

| # | Report Type | Format | Audience |
|---|-------------|--------|----------|
| 1 | Full Compliance Assessment | PDF, JSON | Regulators, auditors |
| 2 | Category-Specific Compliance | PDF, JSON | Compliance teams |
| 3 | Supplier Compliance Scorecard | PDF, JSON, HTML | Procurement, suppliers |
| 4 | Red Flag Summary | PDF, JSON | Risk management |
| 5 | Document Verification Status | PDF, JSON | Document management |
| 6 | Certification Validity Report | PDF, JSON | Certification managers |
| 7 | Country Legal Framework Summary | PDF, JSON | Legal teams |
| 8 | EUDR Due Diligence Statement Annex | PDF, XBRL, XML | EU regulators |

**Report Generation Pipeline:**
```
Report Request -> Data Aggregation -> Template Selection -> Data Injection
    -> Compliance Score Calculation -> Visualization Generation
    -> Format Rendering -> Digital Signature -> Provenance Hash -> Output
```

**Multi-Language Support (5):**
- English (EN), French (FR), German (DE), Spanish (ES), Portuguese (PT)

**Zero-Hallucination Approach:**
- All report data sourced from deterministic engine outputs
- LLM may be used ONLY for narrative section generation (executive summary text)
- All numeric values, scores, and compliance determinations are engine-calculated
- Narrative sections clearly marked as AI-generated when LLM is used
- Report includes complete provenance chain (SHA-256 hash of all input data)

**Estimated Lines of Code:** 3,000-3,500

---

## 4. Data Models (Pydantic v2)

### 4.1 Enumerations (12)

```python
class LegislationCategory(str, Enum):
    """EUDR Article 2(40) legislation categories."""
    LAND_USE_RIGHTS = "land_use_rights"
    ENVIRONMENTAL_PROTECTION = "environmental_protection"
    FOREST_RELATED_RULES = "forest_related_rules"
    THIRD_PARTY_RIGHTS = "third_party_rights"
    LABOUR_RIGHTS = "labour_rights"
    TAX_AND_ROYALTY = "tax_and_royalty"
    TRADE_AND_CUSTOMS = "trade_and_customs"
    ANTI_CORRUPTION = "anti_corruption"

class ComplianceStatus(str, Enum):
    """Compliance determination states."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    INSUFFICIENT_DATA = "insufficient_data"

class DocumentValidityStatus(str, Enum):
    """Document validity states."""
    VALID = "valid"
    EXPIRED = "expired"
    EXPIRING_SOON = "expiring_soon"
    SUSPENDED = "suspended"
    REVOKED = "revoked"
    UNVERIFIABLE = "unverifiable"

class CertificationScheme(str, Enum):
    """Supported certification schemes."""
    FSC_FM = "fsc_fm"
    FSC_COC = "fsc_coc"
    FSC_CW = "fsc_cw"
    PEFC_SFM = "pefc_sfm"
    PEFC_COC = "pefc_coc"
    RSPO_PC = "rspo_pc"
    RSPO_SCC = "rspo_scc"
    RSPO_IS = "rspo_is"
    RA_SA = "ra_sa"
    RA_COC = "ra_coc"
    ISCC_EU = "iscc_eu"
    ISCC_PLUS = "iscc_plus"

class RedFlagSeverity(str, Enum):
    """Red flag severity classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class RedFlagCategory(str, Enum):
    """Red flag indicator categories."""
    CORRUPTION_BRIBERY = "corruption_bribery"
    ILLEGAL_LOGGING = "illegal_logging"
    LAND_RIGHTS_VIOLATION = "land_rights_violation"
    LABOUR_VIOLATION = "labour_violation"
    TAX_EVASION = "tax_evasion"
    DOCUMENT_FRAUD = "document_fraud"

class AuditFindingCategory(str, Enum):
    """Audit finding classification."""
    MAJOR_NON_CONFORMITY = "major_non_conformity"
    MINOR_NON_CONFORMITY = "minor_non_conformity"
    OBSERVATION = "observation"
    POSITIVE_PRACTICE = "positive_practice"
    NOT_APPLICABLE = "not_applicable"

class AuditFindingStatus(str, Enum):
    """Corrective action follow-up status."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    CLOSED = "closed"
    OVERDUE = "overdue"

class ReportType(str, Enum):
    """Compliance report types."""
    FULL_ASSESSMENT = "full_assessment"
    CATEGORY_SPECIFIC = "category_specific"
    SUPPLIER_SCORECARD = "supplier_scorecard"
    RED_FLAG_SUMMARY = "red_flag_summary"
    DOCUMENT_STATUS = "document_status"
    CERTIFICATION_VALIDITY = "certification_validity"
    COUNTRY_FRAMEWORK = "country_framework"
    DDS_ANNEX = "dds_annex"

class ReportFormat(str, Enum):
    """Report output formats."""
    PDF = "pdf"
    JSON = "json"
    HTML = "html"
    XBRL = "xbrl"
    XML = "xml"

class RiskLevel(str, Enum):
    """General risk level classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class CommodityType(str, Enum):
    """EUDR Article 1 regulated commodities."""
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"
```

### 4.2 Core Models (10)

```python
class LegalFramework(BaseModel):
    """Country-specific legal framework record for one legislation category."""
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    country_code: str = Field(..., min_length=2, max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code")
    category: LegislationCategory
    law_name: str = Field(..., max_length=500)
    law_reference: str = Field(..., max_length=200,
        description="Official gazette/statute reference number")
    enacted_date: date
    last_amended_date: Optional[date] = None
    enforcement_status: str = Field(default="active",
        pattern="^(active|suspended|repealed|pending)$")
    applicable_commodities: List[CommodityType] = Field(default_factory=list)
    source_database: str = Field(...,
        pattern="^(faolex|ecolex|natlex|national_portal|eu_oj|wto)$")
    source_url: Optional[str] = None
    requirements: List[str] = Field(default_factory=list,
        description="List of specific legal requirements")
    provenance_hash: Optional[str] = Field(None, max_length=64)
    metadata: Optional[Dict[str, Any]] = None
    tenant_id: uuid.UUID
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ComplianceDocument(BaseModel):
    """A permit, license, certificate, or legal document for verification."""
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    supplier_id: uuid.UUID
    document_type: str = Field(..., max_length=100)
    document_number: str = Field(..., max_length=200)
    issuing_authority: str = Field(..., max_length=300)
    issuing_country: str = Field(..., min_length=2, max_length=3)
    issue_date: date
    expiry_date: Optional[date] = None
    validity_status: DocumentValidityStatus = DocumentValidityStatus.VALID
    verification_score: Decimal = Field(default=Decimal("0"),
        ge=Decimal("0"), le=Decimal("100"))
    verification_details: Optional[Dict[str, Any]] = None
    s3_document_key: Optional[str] = Field(None, max_length=500,
        description="S3 object key for stored document")
    checksum_sha256: Optional[str] = Field(None, max_length=64)
    legislation_category: LegislationCategory
    linked_framework_id: Optional[uuid.UUID] = None
    provenance_hash: Optional[str] = Field(None, max_length=64)
    tenant_id: uuid.UUID
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class CertificationRecord(BaseModel):
    """Certification scheme record for a supplier or site."""
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    supplier_id: uuid.UUID
    scheme: CertificationScheme
    certificate_number: str = Field(..., max_length=100)
    certification_body: str = Field(..., max_length=300)
    cb_accreditation_number: Optional[str] = Field(None, max_length=100)
    scope_description: str = Field(default="", max_length=2000)
    covered_commodities: List[CommodityType] = Field(default_factory=list)
    covered_sites: List[str] = Field(default_factory=list)
    coc_model: Optional[str] = Field(None, max_length=50)
    issue_date: date
    expiry_date: date
    last_audit_date: Optional[date] = None
    next_audit_date: Optional[date] = None
    validity_status: DocumentValidityStatus = DocumentValidityStatus.VALID
    non_conformities_open: int = Field(default=0, ge=0)
    eudr_equivalence_score: Decimal = Field(default=Decimal("0"),
        ge=Decimal("0"), le=Decimal("100"),
        description="Percentage of EUDR requirements satisfied by this cert")
    eudr_categories_covered: List[LegislationCategory] = Field(
        default_factory=list)
    provenance_hash: Optional[str] = Field(None, max_length=64)
    metadata: Optional[Dict[str, Any]] = None
    tenant_id: uuid.UUID
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class RedFlagAlert(BaseModel):
    """A triggered red flag indicator for a supplier."""
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    supplier_id: uuid.UUID
    country_code: str = Field(..., min_length=2, max_length=3)
    flag_code: str = Field(..., max_length=10,
        description="Red flag identifier (RF-001 through RF-040)")
    flag_category: RedFlagCategory
    flag_description: str = Field(..., max_length=500)
    severity: RedFlagSeverity
    base_weight: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    country_multiplier: Decimal = Field(default=Decimal("1.0"),
        ge=Decimal("0.5"), le=Decimal("3.0"))
    commodity_multiplier: Decimal = Field(default=Decimal("1.0"),
        ge=Decimal("0.5"), le=Decimal("2.0"))
    weighted_score: Decimal = Field(default=Decimal("0"),
        ge=Decimal("0"), le=Decimal("100"))
    triggering_data: Dict[str, Any] = Field(default_factory=dict,
        description="Specific data point and threshold that triggered the flag")
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    provenance_hash: Optional[str] = Field(None, max_length=64)
    tenant_id: uuid.UUID
    created_at: datetime = Field(default_factory=_utcnow)


class AuditReport(BaseModel):
    """Third-party audit report with extracted findings."""
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    supplier_id: uuid.UUID
    audit_type: str = Field(..., max_length=100,
        description="FSC, PEFC, RSPO, EUDR DD, government, ISO")
    auditor_organization: str = Field(..., max_length=300)
    lead_auditor: Optional[str] = Field(None, max_length=200)
    audit_date: date
    report_date: date
    scope: str = Field(default="", max_length=2000)
    overall_conclusion: str = Field(default="",
        pattern="^(conformant|minor_nc|major_nc|suspended|withdrawn|)$")
    major_non_conformities: int = Field(default=0, ge=0)
    minor_non_conformities: int = Field(default=0, ge=0)
    observations: int = Field(default=0, ge=0)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    corrective_action_deadline: Optional[date] = None
    follow_up_audit_date: Optional[date] = None
    s3_report_key: Optional[str] = Field(None, max_length=500)
    provenance_hash: Optional[str] = Field(None, max_length=64)
    metadata: Optional[Dict[str, Any]] = None
    tenant_id: uuid.UUID
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ComplianceAssessment(BaseModel):
    """Full 8-category compliance assessment for a supplier-commodity pair."""
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    supplier_id: uuid.UUID
    country_code: str = Field(..., min_length=2, max_length=3)
    commodity: CommodityType
    assessment_date: datetime = Field(default_factory=_utcnow)
    overall_status: ComplianceStatus
    overall_score: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    category_scores: Dict[str, Decimal] = Field(default_factory=dict,
        description="Score per LegislationCategory (0-100)")
    category_statuses: Dict[str, str] = Field(default_factory=dict,
        description="ComplianceStatus per LegislationCategory")
    requirements_total: int = Field(default=0, ge=0)
    requirements_met: int = Field(default=0, ge=0)
    requirements_unmet: int = Field(default=0, ge=0)
    requirements_insufficient_data: int = Field(default=0, ge=0)
    gap_analysis: List[Dict[str, Any]] = Field(default_factory=list,
        description="List of unmet requirements with remediation guidance")
    red_flag_count: int = Field(default=0, ge=0)
    red_flag_score: Decimal = Field(default=Decimal("0"),
        ge=Decimal("0"), le=Decimal("100"))
    documents_verified: int = Field(default=0, ge=0)
    certifications_validated: int = Field(default=0, ge=0)
    risk_level: RiskLevel = RiskLevel.MODERATE
    provenance_hash: Optional[str] = Field(None, max_length=64)
    chain_hash: Optional[str] = Field(None, max_length=64,
        description="SHA-256 chain hash linking all input provenance hashes")
    metadata: Optional[Dict[str, Any]] = None
    tenant_id: uuid.UUID
    created_at: datetime = Field(default_factory=_utcnow)


class LegalRequirement(BaseModel):
    """A specific legal requirement derived from a legal framework."""
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    framework_id: uuid.UUID
    country_code: str = Field(..., min_length=2, max_length=3)
    category: LegislationCategory
    requirement_code: str = Field(..., max_length=50)
    description: str = Field(..., max_length=1000)
    applicable_commodities: List[CommodityType] = Field(default_factory=list)
    mandatory: bool = True
    evidence_types: List[str] = Field(default_factory=list,
        description="Document types that satisfy this requirement")
    verification_method: str = Field(default="document_check",
        pattern="^(document_check|certification_check|field_audit|self_declaration|database_query)$")
    provenance_hash: Optional[str] = Field(None, max_length=64)
    tenant_id: uuid.UUID


class ComplianceReport(BaseModel):
    """Generated compliance report metadata."""
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    assessment_id: uuid.UUID
    report_type: ReportType
    report_format: ReportFormat
    language: str = Field(default="en", pattern="^(en|fr|de|es|pt)$")
    s3_report_key: str = Field(..., max_length=500)
    file_size_bytes: int = Field(default=0, ge=0)
    generated_at: datetime = Field(default_factory=_utcnow)
    digital_signature: Optional[str] = Field(None, max_length=512)
    provenance_hash: Optional[str] = Field(None, max_length=64)
    tenant_id: uuid.UUID


class AuditLogEntry(BaseModel):
    """Audit trail entry for all LCV operations."""
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    entity_type: str = Field(..., max_length=50)
    entity_id: uuid.UUID
    action: str = Field(..., max_length=50)
    actor: str = Field(..., max_length=200)
    details: Optional[Dict[str, Any]] = None
    provenance_hash: Optional[str] = Field(None, max_length=64)
    chain_hash: Optional[str] = Field(None, max_length=64)
    tenant_id: uuid.UUID
    created_at: datetime = Field(default_factory=_utcnow)
```

### 4.3 Request Models (12)

```python
class QueryLegalFrameworkRequest(BaseModel):
    country_code: str = Field(..., min_length=2, max_length=3)
    category: Optional[LegislationCategory] = None
    commodity: Optional[CommodityType] = None
    include_repealed: bool = False

class VerifyDocumentRequest(BaseModel):
    supplier_id: uuid.UUID
    document_type: str = Field(..., max_length=100)
    document_number: str = Field(..., max_length=200)
    issuing_authority: str = Field(..., max_length=300)
    issuing_country: str = Field(..., min_length=2, max_length=3)
    issue_date: date
    expiry_date: Optional[date] = None
    legislation_category: LegislationCategory
    s3_document_key: Optional[str] = None

class ValidateCertificationRequest(BaseModel):
    supplier_id: uuid.UUID
    scheme: CertificationScheme
    certificate_number: str = Field(..., max_length=100)
    certification_body: Optional[str] = None

class ScanRedFlagsRequest(BaseModel):
    supplier_id: uuid.UUID
    country_code: str = Field(..., min_length=2, max_length=3)
    commodity: CommodityType
    include_categories: Optional[List[RedFlagCategory]] = None

class AssessComplianceRequest(BaseModel):
    supplier_id: uuid.UUID
    country_code: str = Field(..., min_length=2, max_length=3)
    commodity: CommodityType
    categories: Optional[List[LegislationCategory]] = None
    include_red_flags: bool = True
    include_certifications: bool = True

class SubmitAuditReportRequest(BaseModel):
    supplier_id: uuid.UUID
    audit_type: str = Field(..., max_length=100)
    auditor_organization: str = Field(..., max_length=300)
    audit_date: date
    report_date: date
    s3_report_key: str = Field(..., max_length=500)

class GenerateReportRequest(BaseModel):
    assessment_id: uuid.UUID
    report_type: ReportType
    report_format: ReportFormat = ReportFormat.PDF
    language: str = Field(default="en", pattern="^(en|fr|de|es|pt)$")

class BatchAssessmentRequest(BaseModel):
    supplier_ids: List[uuid.UUID] = Field(..., max_length=1000)
    commodity: CommodityType
    categories: Optional[List[LegislationCategory]] = None
    include_red_flags: bool = True

class AcknowledgeRedFlagRequest(BaseModel):
    flag_id: uuid.UUID
    acknowledged_by: str = Field(..., max_length=200)
    justification: Optional[str] = Field(None, max_length=1000)

class UpdateFrameworkRequest(BaseModel):
    country_code: str = Field(..., min_length=2, max_length=3)
    category: LegislationCategory
    law_name: str = Field(..., max_length=500)
    law_reference: str = Field(..., max_length=200)
    enacted_date: date
    source_database: str
    requirements: List[str] = Field(default_factory=list)

class ExpiringDocumentsRequest(BaseModel):
    days_ahead: int = Field(default=30, ge=1, le=365)
    country_code: Optional[str] = None
    document_type: Optional[str] = None

class CountryComplianceRequest(BaseModel):
    country_code: str = Field(..., min_length=2, max_length=3)
    commodity: CommodityType
    supplier_ids: Optional[List[uuid.UUID]] = None
```

### 4.4 Response Models (12)

```python
class LegalFrameworkResponse(BaseModel):
    frameworks: List[LegalFramework]
    total_count: int
    country_code: str
    categories_covered: List[str]
    provenance_hash: str

class DocumentVerificationResponse(BaseModel):
    document: ComplianceDocument
    verification_passed: bool
    verification_steps: List[Dict[str, Any]]
    warnings: List[str]
    provenance_hash: str

class CertificationValidationResponse(BaseModel):
    certification: CertificationRecord
    validation_passed: bool
    validation_checks: List[Dict[str, Any]]
    eudr_coverage_summary: Dict[str, bool]
    provenance_hash: str

class RedFlagScanResponse(BaseModel):
    supplier_id: uuid.UUID
    flags_triggered: List[RedFlagAlert]
    total_flags: int
    aggregate_score: Decimal
    risk_level: RiskLevel
    category_breakdown: Dict[str, int]
    provenance_hash: str

class ComplianceAssessmentResponse(BaseModel):
    assessment: ComplianceAssessment
    frameworks_checked: int
    documents_verified: int
    certifications_validated: int
    red_flags_detected: int
    gap_analysis: List[Dict[str, Any]]
    recommendations: List[str]
    provenance_hash: str

class AuditReportResponse(BaseModel):
    report: AuditReport
    findings_extracted: int
    compliance_impact: Dict[str, Any]
    provenance_hash: str

class ComplianceReportResponse(BaseModel):
    report: ComplianceReport
    download_url: str
    provenance_hash: str

class BatchAssessmentResponse(BaseModel):
    job_id: uuid.UUID
    total_suppliers: int
    completed: int
    failed: int
    results: List[ComplianceAssessmentResponse]
    provenance_hash: str

class ExpiringDocumentsResponse(BaseModel):
    documents: List[ComplianceDocument]
    total_expiring: int
    by_category: Dict[str, int]
    by_country: Dict[str, int]

class CountryComplianceSummaryResponse(BaseModel):
    country_code: str
    commodity: CommodityType
    framework_completeness: Decimal
    average_supplier_score: Decimal
    category_averages: Dict[str, Decimal]
    non_compliant_count: int
    total_assessed: int
    provenance_hash: str

class HealthCheckResponse(BaseModel):
    status: str
    version: str
    engines: Dict[str, str]
    database_connected: bool
    redis_connected: bool
    uptime_seconds: float
    provenance_chain_valid: bool

class AdminStatsResponse(BaseModel):
    total_frameworks: int
    total_documents: int
    total_certifications: int
    total_assessments: int
    total_red_flags: int
    total_audit_reports: int
    countries_covered: int
    assessments_last_24h: int
```

---

## 5. API Endpoint Specification

### 5.1 Route Summary (37 endpoints across 9 groups)

**API Prefix:** `/v1/eudr-lcv`

#### Group 1: Legal Framework Routes (`framework_routes.py`) -- 5 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| GET | `/frameworks/{country_code}` | `eudr-lcv:framework:read` | Get legal frameworks for a country |
| GET | `/frameworks/{country_code}/{category}` | `eudr-lcv:framework:read` | Get frameworks for country + category |
| POST | `/frameworks` | `eudr-lcv:framework:write` | Create/update legal framework record |
| GET | `/frameworks/search` | `eudr-lcv:framework:read` | Search frameworks by commodity, keyword |
| GET | `/frameworks/coverage` | `eudr-lcv:framework:read` | Get framework coverage matrix (country x category) |

#### Group 2: Document Verification Routes (`document_routes.py`) -- 5 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/documents/verify` | `eudr-lcv:document:verify` | Verify a compliance document |
| GET | `/documents/{document_id}` | `eudr-lcv:document:read` | Get document verification details |
| GET | `/documents/supplier/{supplier_id}` | `eudr-lcv:document:read` | List all documents for a supplier |
| GET | `/documents/expiring` | `eudr-lcv:document:read` | List expiring documents |
| DELETE | `/documents/{document_id}` | `eudr-lcv:document:delete` | Remove a document record |

#### Group 3: Certification Routes (`certification_routes.py`) -- 4 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/certifications/validate` | `eudr-lcv:certification:validate` | Validate a certification |
| GET | `/certifications/supplier/{supplier_id}` | `eudr-lcv:certification:read` | List certifications for supplier |
| GET | `/certifications/{cert_id}` | `eudr-lcv:certification:read` | Get certification details |
| GET | `/certifications/schemes` | `eudr-lcv:certification:read` | List supported schemes and EUDR mapping |

#### Group 4: Red Flag Routes (`red_flag_routes.py`) -- 4 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/red-flags/scan` | `eudr-lcv:red-flag:scan` | Scan supplier for red flags |
| GET | `/red-flags/supplier/{supplier_id}` | `eudr-lcv:red-flag:read` | List red flags for supplier |
| POST | `/red-flags/{flag_id}/acknowledge` | `eudr-lcv:red-flag:acknowledge` | Acknowledge a red flag |
| GET | `/red-flags/summary` | `eudr-lcv:red-flag:read` | Red flag summary across all suppliers |

#### Group 5: Compliance Assessment Routes (`compliance_routes.py`) -- 5 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/compliance/assess` | `eudr-lcv:compliance:assess` | Run full compliance assessment |
| GET | `/compliance/{assessment_id}` | `eudr-lcv:compliance:read` | Get assessment details |
| GET | `/compliance/supplier/{supplier_id}` | `eudr-lcv:compliance:read` | List assessments for supplier |
| GET | `/compliance/country/{country_code}` | `eudr-lcv:compliance:read` | Country compliance summary |
| GET | `/compliance/gaps/{assessment_id}` | `eudr-lcv:compliance:read` | Get gap analysis for assessment |

#### Group 6: Audit Integration Routes (`audit_routes.py`) -- 4 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/audits/submit` | `eudr-lcv:audit:write` | Submit audit report for processing |
| GET | `/audits/{audit_id}` | `eudr-lcv:audit:read` | Get audit report with findings |
| GET | `/audits/supplier/{supplier_id}` | `eudr-lcv:audit:read` | List audit reports for supplier |
| GET | `/audits/{audit_id}/findings` | `eudr-lcv:audit:read` | Get extracted findings for audit |

#### Group 7: Report Routes (`report_routes.py`) -- 4 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/reports/generate` | `eudr-lcv:report:generate` | Generate compliance report |
| GET | `/reports/{report_id}` | `eudr-lcv:report:read` | Get report metadata |
| GET | `/reports/{report_id}/download` | `eudr-lcv:report:download` | Download report file |
| GET | `/reports/assessment/{assessment_id}` | `eudr-lcv:report:read` | List reports for assessment |

#### Group 8: Batch Processing Routes (`batch_routes.py`) -- 3 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/batch/assess` | `eudr-lcv:batch:execute` | Batch compliance assessment |
| GET | `/batch/{job_id}` | `eudr-lcv:batch:read` | Get batch job status |
| GET | `/batch/{job_id}/results` | `eudr-lcv:batch:read` | Get batch job results |

#### Group 9: Admin & Health Routes (`admin_routes.py`) -- 3 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| GET | `/health` | (public) | Health check |
| GET | `/stats` | `eudr-lcv:admin:read` | Service statistics |
| POST | `/cache/invalidate` | `eudr-lcv:admin:manage` | Invalidate Redis cache |

**Total: 37 endpoints**

### 5.2 RBAC Permissions (20)

```
eudr-lcv:framework:read
eudr-lcv:framework:write
eudr-lcv:document:read
eudr-lcv:document:verify
eudr-lcv:document:delete
eudr-lcv:certification:read
eudr-lcv:certification:validate
eudr-lcv:red-flag:read
eudr-lcv:red-flag:scan
eudr-lcv:red-flag:acknowledge
eudr-lcv:compliance:read
eudr-lcv:compliance:assess
eudr-lcv:audit:read
eudr-lcv:audit:write
eudr-lcv:report:read
eudr-lcv:report:generate
eudr-lcv:report:download
eudr-lcv:batch:read
eudr-lcv:batch:execute
eudr-lcv:admin:manage
```

---

## 6. Database Schema (V111)

### 6.1 Overview

- **Migration:** `V111__agent_eudr_legal_compliance_verifier.sql`
- **Tables:** 15 (11 regular + 4 TimescaleDB hypertables)
- **Table prefix:** `gl_eudr_lcv_`
- **Estimated indexes:** ~160

### 6.2 Table Definitions

#### Regular Tables (11)

**1. `gl_eudr_lcv_legal_frameworks`** -- Country-specific legal framework records

| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK, DEFAULT gen_random_uuid() |
| country_code | VARCHAR(3) | NOT NULL |
| category | VARCHAR(40) | NOT NULL, CHECK IN 8 categories |
| law_name | VARCHAR(500) | NOT NULL |
| law_reference | VARCHAR(200) | NOT NULL |
| enacted_date | DATE | NOT NULL |
| last_amended_date | DATE | |
| enforcement_status | VARCHAR(20) | DEFAULT 'active' |
| applicable_commodities | JSONB | DEFAULT '[]' |
| source_database | VARCHAR(20) | NOT NULL |
| source_url | VARCHAR(1000) | |
| requirements | JSONB | DEFAULT '[]' |
| provenance_hash | VARCHAR(64) | |
| metadata | JSONB | |
| tenant_id | UUID | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** country_code, category, (country_code, category), enforcement_status, applicable_commodities (GIN), source_database, tenant_id, provenance_hash, created_at, updated_at, (country_code, category, enforcement_status)

**2. `gl_eudr_lcv_legal_requirements`** -- Specific legal requirements per framework

| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK |
| framework_id | UUID | NOT NULL, FK -> legal_frameworks |
| country_code | VARCHAR(3) | NOT NULL |
| category | VARCHAR(40) | NOT NULL |
| requirement_code | VARCHAR(50) | NOT NULL, UNIQUE per country |
| description | VARCHAR(1000) | NOT NULL |
| applicable_commodities | JSONB | DEFAULT '[]' |
| mandatory | BOOLEAN | DEFAULT TRUE |
| evidence_types | JSONB | DEFAULT '[]' |
| verification_method | VARCHAR(30) | DEFAULT 'document_check' |
| provenance_hash | VARCHAR(64) | |
| tenant_id | UUID | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** framework_id, country_code, category, requirement_code, mandatory, (country_code, category, mandatory), tenant_id

**3. `gl_eudr_lcv_compliance_documents`** -- Uploaded/verified compliance documents

| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK |
| supplier_id | UUID | NOT NULL |
| document_type | VARCHAR(100) | NOT NULL |
| document_number | VARCHAR(200) | NOT NULL |
| issuing_authority | VARCHAR(300) | NOT NULL |
| issuing_country | VARCHAR(3) | NOT NULL |
| issue_date | DATE | NOT NULL |
| expiry_date | DATE | |
| validity_status | VARCHAR(20) | DEFAULT 'valid' |
| verification_score | DECIMAL(5,2) | DEFAULT 0 |
| verification_details | JSONB | |
| s3_document_key | VARCHAR(500) | |
| checksum_sha256 | VARCHAR(64) | |
| legislation_category | VARCHAR(40) | NOT NULL |
| linked_framework_id | UUID | FK -> legal_frameworks |
| provenance_hash | VARCHAR(64) | |
| tenant_id | UUID | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** supplier_id, document_type, issuing_country, validity_status, expiry_date, legislation_category, (supplier_id, validity_status), (expiry_date, validity_status), tenant_id, provenance_hash

**4. `gl_eudr_lcv_certification_records`** -- Certification scheme records

| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK |
| supplier_id | UUID | NOT NULL |
| scheme | VARCHAR(20) | NOT NULL |
| certificate_number | VARCHAR(100) | NOT NULL |
| certification_body | VARCHAR(300) | NOT NULL |
| cb_accreditation_number | VARCHAR(100) | |
| scope_description | TEXT | |
| covered_commodities | JSONB | DEFAULT '[]' |
| covered_sites | JSONB | DEFAULT '[]' |
| coc_model | VARCHAR(50) | |
| issue_date | DATE | NOT NULL |
| expiry_date | DATE | NOT NULL |
| last_audit_date | DATE | |
| next_audit_date | DATE | |
| validity_status | VARCHAR(20) | DEFAULT 'valid' |
| non_conformities_open | INTEGER | DEFAULT 0 |
| eudr_equivalence_score | DECIMAL(5,2) | DEFAULT 0 |
| eudr_categories_covered | JSONB | DEFAULT '[]' |
| provenance_hash | VARCHAR(64) | |
| metadata | JSONB | |
| tenant_id | UUID | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** supplier_id, scheme, certificate_number, certification_body, validity_status, expiry_date, (supplier_id, scheme), (scheme, validity_status), tenant_id, provenance_hash

**5. `gl_eudr_lcv_red_flag_alerts`** -- Triggered red flag indicators

| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK |
| supplier_id | UUID | NOT NULL |
| country_code | VARCHAR(3) | NOT NULL |
| flag_code | VARCHAR(10) | NOT NULL |
| flag_category | VARCHAR(30) | NOT NULL |
| flag_description | VARCHAR(500) | NOT NULL |
| severity | VARCHAR(10) | NOT NULL |
| base_weight | DECIMAL(5,4) | NOT NULL |
| country_multiplier | DECIMAL(5,2) | DEFAULT 1.0 |
| commodity_multiplier | DECIMAL(5,2) | DEFAULT 1.0 |
| weighted_score | DECIMAL(5,2) | DEFAULT 0 |
| triggering_data | JSONB | NOT NULL |
| acknowledged | BOOLEAN | DEFAULT FALSE |
| acknowledged_by | VARCHAR(200) | |
| acknowledged_at | TIMESTAMPTZ | |
| provenance_hash | VARCHAR(64) | |
| tenant_id | UUID | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** supplier_id, country_code, flag_code, flag_category, severity, acknowledged, (supplier_id, acknowledged), (severity, acknowledged), tenant_id, created_at

**6. `gl_eudr_lcv_audit_reports`** -- Third-party audit report records

| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK |
| supplier_id | UUID | NOT NULL |
| audit_type | VARCHAR(100) | NOT NULL |
| auditor_organization | VARCHAR(300) | NOT NULL |
| lead_auditor | VARCHAR(200) | |
| audit_date | DATE | NOT NULL |
| report_date | DATE | NOT NULL |
| scope | TEXT | |
| overall_conclusion | VARCHAR(20) | |
| major_non_conformities | INTEGER | DEFAULT 0 |
| minor_non_conformities | INTEGER | DEFAULT 0 |
| observations | INTEGER | DEFAULT 0 |
| corrective_action_deadline | DATE | |
| follow_up_audit_date | DATE | |
| s3_report_key | VARCHAR(500) | |
| provenance_hash | VARCHAR(64) | |
| metadata | JSONB | |
| tenant_id | UUID | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** supplier_id, audit_type, auditor_organization, overall_conclusion, audit_date, (supplier_id, audit_type), tenant_id, provenance_hash

**7. `gl_eudr_lcv_audit_findings`** -- Individual findings extracted from audit reports

| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK |
| audit_report_id | UUID | NOT NULL, FK -> audit_reports |
| finding_reference | VARCHAR(50) | NOT NULL |
| finding_category | VARCHAR(30) | NOT NULL |
| applicable_requirement | VARCHAR(200) | |
| description | TEXT | NOT NULL |
| evidence_observed | TEXT | |
| root_cause | TEXT | |
| corrective_action_required | TEXT | |
| corrective_action_deadline | DATE | |
| follow_up_status | VARCHAR(20) | DEFAULT 'open' |
| closed_date | DATE | |
| provenance_hash | VARCHAR(64) | |
| tenant_id | UUID | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** audit_report_id, finding_category, follow_up_status, (audit_report_id, finding_category), (follow_up_status, corrective_action_deadline), tenant_id

**8. `gl_eudr_lcv_compliance_reports`** -- Generated report metadata

| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK |
| assessment_id | UUID | NOT NULL, FK -> assessments hypertable |
| report_type | VARCHAR(30) | NOT NULL |
| report_format | VARCHAR(10) | NOT NULL |
| language | VARCHAR(5) | DEFAULT 'en' |
| s3_report_key | VARCHAR(500) | NOT NULL |
| file_size_bytes | BIGINT | DEFAULT 0 |
| digital_signature | VARCHAR(512) | |
| provenance_hash | VARCHAR(64) | |
| tenant_id | UUID | NOT NULL |
| generated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** assessment_id, report_type, report_format, language, (assessment_id, report_type), tenant_id, generated_at

**9. `gl_eudr_lcv_issuing_authorities`** -- Registered issuing authority lookup

| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK |
| country_code | VARCHAR(3) | NOT NULL |
| authority_name | VARCHAR(300) | NOT NULL |
| authority_code | VARCHAR(50) | UNIQUE |
| document_types | JSONB | DEFAULT '[]' |
| verification_url | VARCHAR(1000) | |
| active | BOOLEAN | DEFAULT TRUE |
| tenant_id | UUID | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** country_code, authority_code, active, (country_code, active), tenant_id

**10. `gl_eudr_lcv_certification_bodies`** -- Registered certification body lookup

| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK |
| cb_name | VARCHAR(300) | NOT NULL |
| accreditation_number | VARCHAR(100) | |
| schemes_accredited | JSONB | DEFAULT '[]' |
| accreditation_body | VARCHAR(200) | |
| countries_active | JSONB | DEFAULT '[]' |
| active | BOOLEAN | DEFAULT TRUE |
| tenant_id | UUID | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** cb_name, accreditation_number, active, schemes_accredited (GIN), tenant_id

**11. `gl_eudr_lcv_batch_jobs`** -- Batch processing job tracking

| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | PK |
| job_type | VARCHAR(30) | NOT NULL |
| status | VARCHAR(20) | DEFAULT 'pending' |
| total_items | INTEGER | DEFAULT 0 |
| completed_items | INTEGER | DEFAULT 0 |
| failed_items | INTEGER | DEFAULT 0 |
| started_at | TIMESTAMPTZ | |
| completed_at | TIMESTAMPTZ | |
| error_details | JSONB | |
| provenance_hash | VARCHAR(64) | |
| tenant_id | UUID | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** job_type, status, (status, created_at), tenant_id

#### TimescaleDB Hypertables (4)

**12. `gl_eudr_lcv_compliance_assessments`** -- Full compliance assessments (hypertable on assessed_at)

| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | DEFAULT gen_random_uuid() |
| supplier_id | UUID | NOT NULL |
| country_code | VARCHAR(3) | NOT NULL |
| commodity | VARCHAR(20) | NOT NULL |
| overall_status | VARCHAR(30) | NOT NULL |
| overall_score | DECIMAL(5,2) | NOT NULL |
| category_scores | JSONB | NOT NULL |
| category_statuses | JSONB | NOT NULL |
| requirements_total | INTEGER | DEFAULT 0 |
| requirements_met | INTEGER | DEFAULT 0 |
| requirements_unmet | INTEGER | DEFAULT 0 |
| requirements_insufficient_data | INTEGER | DEFAULT 0 |
| gap_analysis | JSONB | DEFAULT '[]' |
| red_flag_count | INTEGER | DEFAULT 0 |
| red_flag_score | DECIMAL(5,2) | DEFAULT 0 |
| documents_verified | INTEGER | DEFAULT 0 |
| certifications_validated | INTEGER | DEFAULT 0 |
| risk_level | VARCHAR(10) | DEFAULT 'moderate' |
| provenance_hash | VARCHAR(64) | |
| chain_hash | VARCHAR(64) | |
| metadata | JSONB | |
| tenant_id | UUID | NOT NULL |
| assessed_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| PRIMARY KEY (id, assessed_at) | | |

- Chunk interval: 90 days
- Retention policy: 5 years (EUDR Article 31)
- Continuous aggregate: `gl_eudr_lcv_monthly_compliance_summary` (monthly avg scores by country+commodity)

**13. `gl_eudr_lcv_document_verification_log`** -- Document verification event log (hypertable on verified_at)

| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | DEFAULT gen_random_uuid() |
| document_id | UUID | NOT NULL |
| supplier_id | UUID | NOT NULL |
| verification_type | VARCHAR(30) | NOT NULL |
| result | VARCHAR(20) | NOT NULL |
| details | JSONB | |
| provenance_hash | VARCHAR(64) | |
| tenant_id | UUID | NOT NULL |
| verified_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| PRIMARY KEY (id, verified_at) | | |

- Chunk interval: 30 days
- Retention policy: 5 years

**14. `gl_eudr_lcv_red_flag_history`** -- Red flag scan results over time (hypertable on scanned_at)

| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | DEFAULT gen_random_uuid() |
| supplier_id | UUID | NOT NULL |
| country_code | VARCHAR(3) | NOT NULL |
| commodity | VARCHAR(20) | NOT NULL |
| total_flags | INTEGER | DEFAULT 0 |
| aggregate_score | DECIMAL(5,2) | DEFAULT 0 |
| risk_level | VARCHAR(10) | |
| category_breakdown | JSONB | |
| provenance_hash | VARCHAR(64) | |
| tenant_id | UUID | NOT NULL |
| scanned_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| PRIMARY KEY (id, scanned_at) | | |

- Chunk interval: 30 days
- Retention policy: 5 years

**15. `gl_eudr_lcv_audit_log`** -- Comprehensive audit trail (hypertable on created_at)

| Column | Type | Constraints |
|--------|------|-------------|
| id | UUID | DEFAULT gen_random_uuid() |
| entity_type | VARCHAR(50) | NOT NULL |
| entity_id | UUID | NOT NULL |
| action | VARCHAR(50) | NOT NULL |
| actor | VARCHAR(200) | NOT NULL |
| details | JSONB | |
| provenance_hash | VARCHAR(64) | |
| chain_hash | VARCHAR(64) | |
| tenant_id | UUID | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| PRIMARY KEY (id, created_at) | | |

- Chunk interval: 30 days
- Retention policy: 5 years (EUDR Article 31)

### 6.3 Continuous Aggregates (2)

**1. `gl_eudr_lcv_monthly_compliance_summary`**
```sql
SELECT
    time_bucket('1 month', assessed_at) AS month,
    country_code,
    commodity,
    tenant_id,
    AVG(overall_score) AS avg_score,
    COUNT(*) AS assessment_count,
    COUNT(*) FILTER (WHERE overall_status = 'compliant') AS compliant_count,
    COUNT(*) FILTER (WHERE overall_status = 'non_compliant') AS non_compliant_count
FROM gl_eudr_lcv_compliance_assessments
GROUP BY 1, 2, 3, 4;
```

**2. `gl_eudr_lcv_weekly_red_flag_summary`**
```sql
SELECT
    time_bucket('1 week', scanned_at) AS week,
    country_code,
    tenant_id,
    AVG(aggregate_score) AS avg_risk_score,
    SUM(total_flags) AS total_flags,
    COUNT(*) AS scans_performed
FROM gl_eudr_lcv_red_flag_history
GROUP BY 1, 2, 3;
```

---

## 7. Prometheus Metrics (20)

**Prefix:** `gl_eudr_lcv_`

### Counters (10)

| # | Metric Name | Labels | Description |
|---|-------------|--------|-------------|
| 1 | `gl_eudr_lcv_framework_queries_total` | country_code, category | Legal framework queries |
| 2 | `gl_eudr_lcv_document_verifications_total` | country_code, result | Document verifications performed |
| 3 | `gl_eudr_lcv_certification_validations_total` | scheme, result | Certification validations performed |
| 4 | `gl_eudr_lcv_red_flag_scans_total` | country_code, commodity | Red flag scans executed |
| 5 | `gl_eudr_lcv_red_flags_triggered_total` | flag_category, severity | Individual red flags triggered |
| 6 | `gl_eudr_lcv_compliance_assessments_total` | country_code, commodity, status | Compliance assessments completed |
| 7 | `gl_eudr_lcv_audit_reports_processed_total` | audit_type | Audit reports processed |
| 8 | `gl_eudr_lcv_reports_generated_total` | report_type, format | Compliance reports generated |
| 9 | `gl_eudr_lcv_batch_jobs_total` | job_type, status | Batch processing jobs |
| 10 | `gl_eudr_lcv_api_errors_total` | operation, status_code | API errors by operation |

### Histograms (5)

| # | Metric Name | Buckets | Description |
|---|-------------|---------|-------------|
| 11 | `gl_eudr_lcv_compliance_check_duration_seconds` | 0.1, 0.25, 0.5, 1, 2, 5, 10 | Single compliance check latency |
| 12 | `gl_eudr_lcv_full_assessment_duration_seconds` | 0.5, 1, 2, 5, 10, 30 | Full 8-category assessment latency |
| 13 | `gl_eudr_lcv_document_verification_duration_seconds` | 0.1, 0.5, 1, 2, 5 | Document verification latency |
| 14 | `gl_eudr_lcv_red_flag_scan_duration_seconds` | 0.1, 0.5, 1, 3, 5 | Red flag scan latency |
| 15 | `gl_eudr_lcv_report_generation_duration_seconds` | 1, 2, 5, 10, 30 | Report generation latency |

### Gauges (5)

| # | Metric Name | Description |
|---|-------------|-------------|
| 16 | `gl_eudr_lcv_countries_covered` | Number of countries with legal framework data |
| 17 | `gl_eudr_lcv_active_red_flags` | Number of unacknowledged red flags |
| 18 | `gl_eudr_lcv_expiring_documents_30d` | Documents expiring within 30 days |
| 19 | `gl_eudr_lcv_non_compliant_suppliers` | Suppliers with non-compliant status |
| 20 | `gl_eudr_lcv_cache_hit_ratio` | Redis cache hit ratio (0.0-1.0) |

---

## 8. Integration Architecture

### 8.1 Internal Agent Integrations (6)

| Agent | Integration Direction | Data Exchange |
|-------|----------------------|---------------|
| EUDR-001 Supply Chain Mapping Master | LCV reads from | Supplier IDs, country codes, commodity types, supply chain graph |
| EUDR-016 Country Risk Evaluator | LCV reads from | Country risk scores, governance ratings, CPI/WGI data |
| EUDR-021 Indigenous Rights Checker | LCV reads from | FPIC status, indigenous territory overlap, rights violation alerts |
| EUDR-022 Protected Area Validator | LCV reads from | Protected area compliance, buffer zone violations |
| EUDR-012 Document Authentication | LCV reads from | Document authenticity scores, tamper detection results |
| AGENT-DATA-003 ERP/Finance Connector | LCV reads from | Financial records, tax declarations, royalty payments |

### 8.2 External Integrations (6)

| System | Protocol | Purpose | Rate Limit |
|--------|----------|---------|------------|
| FAO FAOLEX API | REST/JSON | Legal framework data for 194 countries | 100 req/min |
| ECOLEX API | REST/JSON | Environmental law database | 60 req/min |
| FSC Certificate Database | REST/JSON | FSC certificate validation | 120 req/min |
| RSPO PalmTrace | REST/JSON | RSPO certificate validation | 60 req/min |
| PEFC Certificate Search | REST/JSON | PEFC certificate validation | 60 req/min |
| ISCC Certification DB | REST/JSON | ISCC certificate validation | 60 req/min |

### 8.3 Infrastructure Integrations

| Component | Usage |
|-----------|-------|
| PostgreSQL 14+ (TimescaleDB) | Primary data store (15 tables, 4 hypertables) |
| Redis 7+ | Caching: framework lookups (TTL 24h), cert validation (TTL 1h), assessment results (TTL 30m) |
| S3 (AWS/MinIO) | Document storage, audit reports, generated reports |
| SEC-001 JWT Auth | JWT RS256 token validation on all protected endpoints |
| SEC-002 RBAC | 20 fine-grained permissions with `eudr-lcv:` prefix |
| OBS-001 Prometheus | 20 metrics (10 counters, 5 histograms, 5 gauges) |
| OBS-003 OpenTelemetry | Distributed tracing across engine calls |

---

## 9. Security Architecture

### 9.1 Authentication

- All endpoints (except `/health`) require JWT authentication via SEC-001
- JWT RS256 token validation with configurable JWKS endpoint
- API key support for machine-to-machine integrations
- Token expiry: 1 hour (configurable via `GL_EUDR_LCV_TOKEN_EXPIRY_S`)

### 9.2 Authorization (RBAC)

- 20 permissions registered in SEC-002 RBAC with `eudr-lcv:` prefix
- Role mapping follows GreenLang standard roles:
  - `admin`: all 20 permissions
  - `compliance_manager`: read + assess + generate + acknowledge
  - `compliance_analyst`: read + scan + assess
  - `auditor`: read + download
  - `viewer`: read-only permissions

### 9.3 Data Security

- All documents encrypted at rest via SEC-003 (AES-256-GCM)
- TLS 1.3 in transit via SEC-004
- PII detection on uploaded documents via SEC-011
- Audit trail for all data access via SEC-005
- Secrets (API keys for external integrations) managed via SEC-006 (HashiCorp Vault)

### 9.4 Rate Limiting

- Default: 100 requests/minute per tenant
- Batch endpoints: 10 requests/minute per tenant
- Admin endpoints: 20 requests/minute per tenant
- Configurable via `GL_EUDR_LCV_RATE_LIMIT_*` environment variables

---

## 10. Performance and Scalability

### 10.1 Caching Strategy (Redis)

| Cache Layer | Key Pattern | TTL | Purpose |
|-------------|-------------|-----|---------|
| Legal framework | `lcv:fw:{country}:{category}` | 24 hours | Framework lookups (changes rarely) |
| Legal requirements | `lcv:req:{country}:{commodity}:{category}` | 24 hours | Requirement checklists |
| Certification validation | `lcv:cert:{scheme}:{cert_number}` | 1 hour | External API result caching |
| Red flag patterns | `lcv:rf:patterns` | 24 hours | Red flag rule definitions |
| Country compliance | `lcv:cc:{country}:{commodity}` | 30 minutes | Country summary scores |
| Assessment results | `lcv:assess:{assessment_id}` | 30 minutes | Recent assessment caching |
| Issuing authorities | `lcv:auth:{country}` | 24 hours | Authority lookup table |

**Expected cache hit rate:** >70% (framework and requirement queries are highly cacheable)

### 10.2 Horizontal Scaling

- Stateless engine design -- all state in PostgreSQL/Redis
- Kubernetes Horizontal Pod Autoscaler (HPA) based on CPU utilization (target: 70%)
- Minimum replicas: 2, maximum: 8
- Batch processing uses async task queue (Redis-backed)

### 10.3 Database Performance

- TimescaleDB hypertables auto-partition time-series data
- Continuous aggregates for dashboard queries (no real-time aggregation)
- Connection pooling: 10 connections per pod (configurable)
- Read replicas for report generation queries

---

## 11. Configuration

### 11.1 Environment Variables (GL_EUDR_LCV_ prefix)

```yaml
# Database
GL_EUDR_LCV_DATABASE_URL: postgresql+asyncpg://...
GL_EUDR_LCV_POOL_SIZE: 10
GL_EUDR_LCV_POOL_TIMEOUT_S: 30
GL_EUDR_LCV_POOL_RECYCLE_S: 3600

# Redis
GL_EUDR_LCV_REDIS_URL: redis://...
GL_EUDR_LCV_REDIS_TTL_S: 86400
GL_EUDR_LCV_REDIS_KEY_PREFIX: lcv

# S3
GL_EUDR_LCV_S3_BUCKET: gl-eudr-lcv-documents
GL_EUDR_LCV_S3_REGION: eu-west-1

# Compliance thresholds
GL_EUDR_LCV_COMPLIANT_THRESHOLD: 80
GL_EUDR_LCV_PARTIAL_THRESHOLD: 50
GL_EUDR_LCV_RED_FLAG_CRITICAL_THRESHOLD: 75
GL_EUDR_LCV_RED_FLAG_HIGH_THRESHOLD: 50
GL_EUDR_LCV_DOCUMENT_EXPIRY_WARNING_DAYS: 30

# Batch processing
GL_EUDR_LCV_BATCH_MAX_SIZE: 1000
GL_EUDR_LCV_BATCH_CONCURRENCY: 10
GL_EUDR_LCV_BATCH_TIMEOUT_S: 120

# External APIs
GL_EUDR_LCV_FAOLEX_API_URL: https://faolex.fao.org/api/v1
GL_EUDR_LCV_ECOLEX_API_URL: https://www.ecolex.org/api/v1
GL_EUDR_LCV_FSC_API_URL: https://info.fsc.org/api/v1
GL_EUDR_LCV_RSPO_API_URL: https://rspo.org/api/v1

# Rate limiting
GL_EUDR_LCV_RATE_LIMIT_DEFAULT: 100
GL_EUDR_LCV_RATE_LIMIT_BATCH: 10

# Provenance
GL_EUDR_LCV_PROVENANCE_ENABLED: true
GL_EUDR_LCV_RETENTION_YEARS: 5

# Logging
GL_EUDR_LCV_LOG_LEVEL: INFO
```

---

## 12. Testing Strategy

### 12.1 Coverage Targets

| Test Category | Target | Description |
|---------------|--------|-------------|
| Unit tests | 85%+ | Per-engine tests, model validation, provenance |
| Integration tests | 80%+ | Cross-engine workflows, API endpoint tests |
| Performance tests | All targets met | Latency benchmarks per engine |
| End-to-end tests | Critical paths | Full assessment pipeline |

### 12.2 Test File Breakdown

| Test File | Estimated Tests | Focus |
|-----------|----------------|-------|
| test_legal_framework_database_engine.py | 80+ | Framework CRUD, querying, coverage matrix |
| test_document_verification_engine.py | 75+ | Document verification pipeline, validity states |
| test_certification_scheme_validator.py | 70+ | 5 schemes x validation checks, EUDR mapping |
| test_red_flag_detection_engine.py | 90+ | 40 red flags, scoring, categorization |
| test_country_compliance_checker.py | 70+ | Per-country rules, gap analysis |
| test_third_party_audit_engine.py | 50+ | Report parsing, finding extraction |
| test_compliance_reporting_engine.py | 45+ | 8 report types, formats, languages |
| test_models.py | 50+ | Pydantic validation, serialization |
| test_config.py | 20+ | Configuration defaults, env overrides |
| test_provenance.py | 30+ | SHA-256 chain, verification |
| **Total** | **580+** | |

### 12.3 Key Test Scenarios

1. Full 8-category compliance assessment for Brazil soya supplier
2. Document verification with expired certificate detection
3. FSC CoC certification validation against FSC database
4. Red flag scan triggering 5+ flags with correct scoring
5. Batch assessment of 100 suppliers completing within time target
6. Report generation in all 5 formats
7. Provenance chain integrity across multi-engine workflow
8. Cache invalidation and refresh for updated legal frameworks
9. Concurrent assessment requests (load test, 50 concurrent)
10. Cross-agent integration: assessment using EUDR-016 country risk data

---

## 13. Deployment Architecture

### 13.1 Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY greenlang/ greenlang/
EXPOSE 8023
CMD ["uvicorn", "greenlang.agents.eudr.legal_compliance_verifier.api.router:app",
     "--host", "0.0.0.0", "--port", "8023"]
```

### 13.2 Kubernetes

- Deployment: 2-8 replicas (HPA)
- Service: ClusterIP on port 8023
- Health probes: `/v1/eudr-lcv/health` (liveness + readiness)
- Resource limits: 512Mi memory, 500m CPU per pod
- Resource requests: 256Mi memory, 250m CPU per pod
- ConfigMap: non-secret configuration
- Secret: database URLs, API keys (from Vault)

### 13.3 Grafana Dashboard

- File: `deployment/monitoring/dashboards/eudr-legal-compliance-verifier.json`
- Panels: 20 (matching 20 Prometheus metrics)
- Sections: Overview, Compliance Assessments, Document Verification, Red Flags, Certifications, Batch Processing, Performance

---

## 14. Risks and Mitigations

| # | Risk | Impact | Probability | Mitigation |
|---|------|--------|-------------|------------|
| 1 | External legal DB APIs unavailable | Cannot refresh frameworks | Medium | Cache with 24h TTL; pre-loaded reference data for 27 countries |
| 2 | Legal framework changes not captured | Stale compliance rules | Medium | Scheduled weekly refresh; manual update workflow; version tracking |
| 3 | Certification scheme API changes | Validation failures | Low | API version pinning; fallback to manual validation |
| 4 | Large batch jobs exceeding timeout | Incomplete assessments | Low | Chunked processing; async job queue; configurable timeout |
| 5 | Red flag false positives | Alert fatigue | Medium | Tunable thresholds; acknowledgment workflow; scoring calibration |
| 6 | Country-specific legal complexity | Incomplete rule coverage | High | Start with 10 priority countries; iterative expansion; legal expert review |
| 7 | Document storage capacity | S3 cost growth | Low | Lifecycle policies; compression; tiered storage |
| 8 | Concurrent assessment load spikes | Degraded performance | Medium | HPA auto-scaling; rate limiting; Redis caching |

---

## 15. Appendices

### A. EUDR Article 2(40) Full Text Reference

> "relevant legislation" means laws applicable in the country of production covering the following areas:
> (a) land use rights;
> (b) environmental protection;
> (c) forest-related rules, including forest management and biodiversity conservation directly related to wood harvesting;
> (d) third parties' rights;
> (e) labour rights;
> (f) human rights protected under international law;
> (g) the principle of free, prior and informed consent (FPIC) including as set out in the UN Declaration on the Rights of Indigenous Peoples;
> (h) tax, anti-corruption, trade and customs regulations.

### B. Country Priority Matrix

| Priority | Countries | Rationale |
|----------|-----------|-----------|
| P0 (Launch) | Brazil, Indonesia, Colombia, Cote d'Ivoire, Ghana, Malaysia | Top 6 EUDR commodity sources |
| P1 (Phase 2) | Peru, Cameroon, DRC, Ecuador, Papua New Guinea, Myanmar | Major sourcing regions |
| P2 (Phase 3) | Paraguay, Bolivia, Honduras, Guatemala, Nicaragua, Vietnam | Secondary sources |
| P3 (Phase 4) | Thailand, India, Ethiopia, Uganda, Tanzania, Nigeria, Laos, Gabon, Congo Rep. | Extended coverage |

### C. Certification Scheme EUDR Equivalence Matrix

| Requirement | FSC FM | FSC CoC | PEFC SFM | RSPO P&C | RA SA |
|-------------|--------|---------|----------|----------|-------|
| Land use rights | Yes | Partial | Yes | Yes | Partial |
| Environmental protection | Yes | No | Yes | Yes | Yes |
| Forest-related rules | Yes | Partial | Yes | N/A | Partial |
| Third-party rights (FPIC) | Yes | No | Partial | Yes | Yes |
| Labour rights | Yes | Partial | Yes | Yes | Yes |
| Tax and royalty | No | No | No | No | No |
| Trade and customs | No | Partial | No | No | No |
| Anti-corruption | Partial | No | Partial | Partial | Partial |

### D. Red Flag Scoring Formula

```
For each triggered flag i:
    flag_score_i = base_weight_i * country_multiplier * commodity_multiplier

Aggregate score = (SUM(flag_score_i) / max_possible_score) * 100

Where:
    max_possible_score = SUM(max_weight_i * max_country_mult * max_commodity_mult)
                         for all 40 flags

Risk classification:
    LOW:      0 <= score < 25
    MODERATE: 25 <= score < 50
    HIGH:     50 <= score < 75
    CRITICAL: 75 <= score <= 100
```

### E. Provenance Chain Design

```
Assessment Provenance Chain:
    hash_0 = SHA-256(framework_data || requirements)
    hash_1 = SHA-256(hash_0 || document_verification_results)
    hash_2 = SHA-256(hash_1 || certification_validation_results)
    hash_3 = SHA-256(hash_2 || red_flag_scan_results)
    hash_4 = SHA-256(hash_3 || country_compliance_results)
    hash_5 = SHA-256(hash_4 || audit_findings)
    chain_hash = SHA-256(hash_5 || assessment_scores)

    The chain_hash stored on the ComplianceAssessment record enables
    bit-perfect verification that all inputs produced the stated output.
```

---

*End of Architecture Specification*
