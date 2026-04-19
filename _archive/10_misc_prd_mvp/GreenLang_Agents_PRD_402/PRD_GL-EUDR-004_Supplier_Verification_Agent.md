# PRD: Supplier Verification Agent (GL-EUDR-004)

**Agent family:** EUDRTraceabilityFamily
**Layer:** Supply Chain Traceability
**Primary domains:** Supplier verification, KYS (Know Your Supplier), due diligence
**Priority:** P0 (highest)
**Doc version:** 1.0
**Last updated:** 2026-01-30 (Asia/Kolkata)

---

## 1. Executive Summary

**Supplier Verification Agent (GL-EUDR-004)** performs comprehensive verification of all suppliers in the EUDR supply chain. It validates supplier identity, legal status, certifications, and operational legitimacy to ensure the supply chain includes only verified entities.

Critical because:
- EUDR requires due diligence on all supply chain participants
- Fraudulent suppliers can introduce non-compliant commodities
- Certifications must be validated for authenticity
- Legal compliance in country of operation must be verified

---

## 2. Problem Statement

Supply chains often include:
- **Shell companies** with no real operations
- **Intermediaries** obscuring true origins
- **Fraudulent certifications** (fake FSC, RSPO, etc.)
- **Non-existent "producers"** used for laundering
- **Sanctioned entities** that must be avoided

Without supplier verification, operators cannot fulfill EUDR Article 10 requirements.

---

## 3. Goals and Non-Goals

### 3.1 Goals (must deliver)

1. **Identity verification**
   - Legal entity validation
   - Tax ID verification
   - Business registration check
   - Ownership structure mapping

2. **Certification verification**
   - FSC/PEFC verification (wood)
   - RSPO verification (palm oil)
   - Rainforest Alliance (cocoa, coffee)
   - Certificate authenticity check
   - Expiry monitoring

3. **Operational verification**
   - Physical facility existence
   - Production capacity validation
   - Geographic footprint matching

4. **Compliance screening**
   - Sanctions list screening
   - Debarred party lists
   - Environmental violation history
   - Legal dispute history

### 3.2 Non-Goals

- Credit/financial verification
- Quality assessment
- Price negotiation
- Contract management

---

## 4. Verification Framework

### 4.1 Verification Levels

| Level | Name | Checks | Required For |
|---|---|---|---|
| 1 | Basic | Entity exists, tax ID valid | All suppliers |
| 2 | Standard | + Certifications, ownership | Tier 1-2 suppliers |
| 3 | Enhanced | + Site verification, capacity | High-risk suppliers |
| 4 | Comprehensive | + Third-party audit, deep ownership | Critical suppliers |

### 4.2 Verification Categories

```python
class VerificationCategory(Enum):
    IDENTITY = "identity"
    LEGAL = "legal"
    CERTIFICATION = "certification"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    FINANCIAL = "financial"

class VerificationCheck:
    category: VerificationCategory
    check_name: str
    check_method: str  # AUTOMATED, MANUAL, HYBRID
    data_sources: List[str]
    weight: float  # 0.0 to 1.0
    required: bool
```

---

## 5. Data Model

```sql
-- Supplier Verification Records
CREATE TABLE supplier_verifications (
    verification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    verification_level INTEGER NOT NULL,  -- 1-4

    -- Status
    status VARCHAR(50) DEFAULT 'PENDING',
    overall_score DECIMAL(5,2),
    risk_rating VARCHAR(20),

    -- Timestamps
    initiated_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    expires_at DATE,
    next_review_date DATE,

    -- Results by category
    identity_verified BOOLEAN,
    identity_score DECIMAL(5,2),
    identity_details JSONB,

    legal_verified BOOLEAN,
    legal_score DECIMAL(5,2),
    legal_details JSONB,

    certification_verified BOOLEAN,
    certification_score DECIMAL(5,2),
    certification_details JSONB,

    operational_verified BOOLEAN,
    operational_score DECIMAL(5,2),
    operational_details JSONB,

    compliance_verified BOOLEAN,
    compliance_score DECIMAL(5,2),
    compliance_details JSONB,

    -- Issues found
    issues JSONB DEFAULT '[]',
    blocking_issues INTEGER DEFAULT 0,

    -- Evidence
    documents JSONB DEFAULT '[]',
    verification_sources JSONB DEFAULT '[]',

    -- Audit
    verified_by VARCHAR(100),
    review_notes TEXT,

    CONSTRAINT valid_level CHECK (verification_level BETWEEN 1 AND 4)
);

-- Certification Records
CREATE TABLE supplier_certifications (
    certification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    certification_type VARCHAR(100) NOT NULL,
    certification_body VARCHAR(255),
    certificate_number VARCHAR(100),
    issue_date DATE,
    expiry_date DATE,
    scope_description TEXT,
    commodities TEXT[],
    facilities UUID[],
    verification_status VARCHAR(50) DEFAULT 'UNVERIFIED',
    verification_date TIMESTAMP,
    verification_method VARCHAR(50),
    certificate_document_id UUID,
    is_valid BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Sanctions Screening Results
CREATE TABLE sanctions_screenings (
    screening_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID NOT NULL,
    screening_date TIMESTAMP DEFAULT NOW(),
    lists_checked TEXT[],
    match_found BOOLEAN DEFAULT FALSE,
    matches JSONB DEFAULT '[]',
    risk_score DECIMAL(5,2),
    screened_by VARCHAR(100),
    next_screening_date DATE
);

-- Verification Check Results
CREATE TABLE verification_checks (
    check_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_id UUID REFERENCES supplier_verifications(verification_id),
    category VARCHAR(50) NOT NULL,
    check_name VARCHAR(100) NOT NULL,
    check_method VARCHAR(50) NOT NULL,
    result VARCHAR(50) NOT NULL,  -- PASS, FAIL, INCONCLUSIVE, NOT_APPLICABLE
    score DECIMAL(5,2),
    evidence JSONB,
    issues JSONB DEFAULT '[]',
    checked_at TIMESTAMP DEFAULT NOW(),
    data_source VARCHAR(100)
);

-- Indexes
CREATE INDEX idx_verifications_supplier ON supplier_verifications(supplier_id);
CREATE INDEX idx_verifications_status ON supplier_verifications(status);
CREATE INDEX idx_certifications_supplier ON supplier_certifications(supplier_id);
CREATE INDEX idx_certifications_expiry ON supplier_certifications(expiry_date);
CREATE INDEX idx_sanctions_supplier ON sanctions_screenings(supplier_id);
```

---

## 6. Functional Requirements

### 6.1 Identity Verification
- **FR-001 (P0):** Verify legal entity name against registries
- **FR-002 (P0):** Validate tax ID / VAT number
- **FR-003 (P0):** Check business registration status
- **FR-004 (P1):** Map ownership structure (beneficial owners)
- **FR-005 (P1):** Verify authorized representatives

### 6.2 Certification Verification
- **FR-010 (P0):** Verify FSC certificates via FSC database API
- **FR-011 (P0):** Verify RSPO certificates via RSPO API
- **FR-012 (P0):** Check certificate validity and scope
- **FR-013 (P0):** Monitor certificate expiry
- **FR-014 (P1):** Verify Rainforest Alliance certificates
- **FR-015 (P1):** Detect fraudulent certificates

### 6.3 Compliance Screening
- **FR-020 (P0):** Screen against OFAC SDN list
- **FR-021 (P0):** Screen against EU sanctions lists
- **FR-022 (P0):** Check World Bank debarred list
- **FR-023 (P1):** Screen against environmental violation databases
- **FR-024 (P1):** Check for illegal logging convictions

### 6.4 Operational Verification
- **FR-030 (P0):** Verify facility addresses exist
- **FR-031 (P1):** Validate production capacity claims
- **FR-032 (P1):** Cross-reference plot locations with claimed facilities
- **FR-033 (P2):** Satellite verification of facility existence

### 6.5 Continuous Monitoring
- **FR-040 (P0):** Schedule periodic re-verification
- **FR-041 (P0):** Alert on certificate expiry
- **FR-042 (P0):** Alert on sanctions list additions
- **FR-043 (P1):** Monitor news for adverse events

---

## 7. Verification Engine

```python
class SupplierVerificationEngine:
    """
    Multi-source supplier verification engine.
    """

    def __init__(self):
        self.fsc_api = FSCCertificateAPI()
        self.rspo_api = RSPOCertificateAPI()
        self.sanctions_screener = SanctionsScreener()
        self.business_registry = BusinessRegistryService()

    async def verify_supplier(
        self,
        supplier_id: UUID,
        level: int = 2
    ) -> VerificationResult:
        """
        Perform comprehensive supplier verification.
        """
        supplier = await get_supplier(supplier_id)
        checks = []

        # Level 1: Basic checks
        checks.append(await self._verify_tax_id(supplier))
        checks.append(await self._verify_business_registration(supplier))
        checks.append(await self._screen_sanctions(supplier))

        if level >= 2:
            # Level 2: Standard checks
            checks.extend(await self._verify_certifications(supplier))
            checks.append(await self._verify_ownership(supplier))

        if level >= 3:
            # Level 3: Enhanced checks
            checks.append(await self._verify_facilities(supplier))
            checks.append(await self._verify_capacity(supplier))

        if level >= 4:
            # Level 4: Comprehensive checks
            checks.append(await self._verify_third_party_audit(supplier))
            checks.append(await self._deep_ownership_check(supplier))

        return self._compile_result(supplier_id, checks, level)

    async def verify_fsc_certificate(
        self,
        certificate_number: str
    ) -> CertificationVerification:
        """
        Verify FSC certificate via official API.
        """
        try:
            cert_data = await self.fsc_api.lookup(certificate_number)

            if not cert_data:
                return CertificationVerification(
                    verified=False,
                    status="NOT_FOUND",
                    message="Certificate not found in FSC database"
                )

            is_valid = (
                cert_data.status == "valid" and
                cert_data.expiry_date > date.today()
            )

            return CertificationVerification(
                verified=True,
                is_valid=is_valid,
                status=cert_data.status,
                holder=cert_data.organization_name,
                scope=cert_data.scope,
                expiry_date=cert_data.expiry_date,
                certification_body=cert_data.cb_name,
                data_source="FSC Public Certificate Database"
            )

        except APIError as e:
            return CertificationVerification(
                verified=False,
                status="API_ERROR",
                message=str(e)
            )

    async def _screen_sanctions(
        self,
        supplier: Supplier
    ) -> VerificationCheck:
        """
        Screen against sanctions lists.
        """
        screening_result = await self.sanctions_screener.screen(
            name=supplier.name,
            country=supplier.country_code,
            aliases=supplier.known_aliases,
            tax_id=supplier.tax_id
        )

        if screening_result.matches:
            return VerificationCheck(
                category="COMPLIANCE",
                check_name="sanctions_screening",
                result="FAIL",
                score=0.0,
                issues=[
                    Issue(
                        severity="BLOCKING",
                        code="SANCTIONS_MATCH",
                        message=f"Match found on {match.list_name}",
                        details=match.to_dict()
                    )
                    for match in screening_result.matches
                ]
            )

        return VerificationCheck(
            category="COMPLIANCE",
            check_name="sanctions_screening",
            result="PASS",
            score=100.0,
            evidence={
                "lists_checked": screening_result.lists_checked,
                "screening_date": datetime.now().isoformat()
            }
        )

    def _compile_result(
        self,
        supplier_id: UUID,
        checks: List[VerificationCheck],
        level: int
    ) -> VerificationResult:
        """
        Compile individual checks into overall verification result.
        """
        # Calculate category scores
        category_scores = defaultdict(list)
        for check in checks:
            if check.score is not None:
                category_scores[check.category].append(check.score)

        # Calculate overall score
        category_weights = {
            "IDENTITY": 0.25,
            "LEGAL": 0.20,
            "CERTIFICATION": 0.20,
            "COMPLIANCE": 0.25,
            "OPERATIONAL": 0.10
        }

        overall_score = 0
        for category, scores in category_scores.items():
            avg_score = sum(scores) / len(scores) if scores else 0
            weight = category_weights.get(category, 0.1)
            overall_score += avg_score * weight

        # Determine risk rating
        blocking_issues = [
            c for c in checks
            if c.result == "FAIL" and any(
                i.severity == "BLOCKING" for i in c.issues
            )
        ]

        if blocking_issues:
            risk_rating = "CRITICAL"
            status = "FAILED"
        elif overall_score >= 80:
            risk_rating = "LOW"
            status = "VERIFIED"
        elif overall_score >= 60:
            risk_rating = "MEDIUM"
            status = "VERIFIED_WITH_ISSUES"
        else:
            risk_rating = "HIGH"
            status = "FAILED"

        return VerificationResult(
            supplier_id=supplier_id,
            verification_level=level,
            status=status,
            overall_score=overall_score,
            risk_rating=risk_rating,
            checks=checks,
            blocking_issues=len(blocking_issues),
            expires_at=date.today() + timedelta(days=365)
        )
```

---

## 8. Certification APIs

### 8.1 FSC Certificate Lookup

```python
class FSCCertificateAPI:
    """
    Interface to FSC public certificate database.
    https://info.fsc.org/certificate.php
    """
    BASE_URL = "https://info.fsc.org/api/v1"

    async def lookup(self, certificate_code: str) -> Optional[FSCCertificate]:
        response = await self.client.get(
            f"{self.BASE_URL}/certificates/{certificate_code}"
        )
        if response.status_code == 404:
            return None
        return FSCCertificate.from_api(response.json())

    async def search(self, organization_name: str) -> List[FSCCertificate]:
        response = await self.client.get(
            f"{self.BASE_URL}/certificates",
            params={"organization": organization_name}
        )
        return [FSCCertificate.from_api(c) for c in response.json()]
```

### 8.2 RSPO Certificate Lookup

```python
class RSPOCertificateAPI:
    """
    Interface to RSPO certificate database.
    https://rspo.org/certification/
    """
    BASE_URL = "https://rspo.org/api/v1"

    async def lookup(self, member_number: str) -> Optional[RSPOCertificate]:
        response = await self.client.get(
            f"{self.BASE_URL}/members/{member_number}/certificates"
        )
        if response.status_code == 404:
            return None
        return RSPOCertificate.from_api(response.json())
```

---

## 9. API Specification

```yaml
paths:
  /api/v1/suppliers/{supplier_id}/verify:
    post:
      summary: Initiate supplier verification
      parameters:
        - name: supplier_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                verification_level:
                  type: integer
                  minimum: 1
                  maximum: 4
                  default: 2
      responses:
        202:
          description: Verification initiated

  /api/v1/suppliers/{supplier_id}/verification:
    get:
      summary: Get verification status
      responses:
        200:
          description: Verification result

  /api/v1/certifications/verify:
    post:
      summary: Verify a certification
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required:
                - certification_type
                - certificate_number
              properties:
                certification_type:
                  type: string
                  enum: [FSC, RSPO, PEFC, RAINFOREST_ALLIANCE]
                certificate_number:
                  type: string
      responses:
        200:
          description: Certification verification result

  /api/v1/sanctions/screen:
    post:
      summary: Screen entity against sanctions lists
```

---

## 10. Success Metrics

- **Verification Coverage:** 100% of active suppliers verified
- **Certification Accuracy:** 99% match with official databases
- **Sanctions Detection:** 100% of listed entities detected
- **Verification Speed:** <24 hours for Level 2
- **Re-verification Rate:** 100% within expiry period

---

## 11. Testing Strategy

### 11.1 Integration Tests
- FSC API integration
- RSPO API integration
- Sanctions screening integration

### 11.2 Verification Tests
- Valid certificate verification
- Expired certificate detection
- Fraudulent certificate detection
- Sanctions match detection

---

*Document Version: 1.0*
*Created: 2026-01-30*
*Status: APPROVED FOR IMPLEMENTATION*
