# PRD: Traceability Audit Agent (GL-EUDR-014)

**Agent family:** EUDRTraceabilityFamily
**Layer:** Supply Chain Traceability
**Primary domains:** Audit trails, compliance verification, evidence management
**Priority:** P0 (highest)
**Doc version:** 1.0
**Last updated:** 2026-01-30 (Asia/Kolkata)

---

## 1. Executive Summary

**Traceability Audit Agent (GL-EUDR-014)** maintains comprehensive audit trails for all EUDR traceability activities. It enables competent authorities to verify compliance by providing complete evidence chains from EU market back to origin plots, supporting regulatory audits and internal compliance reviews.

---

## 2. EUDR Audit Requirements

### 2.1 Regulatory Context

Per EUDR Article 14, operators must:
- Keep records for **5 years** after DDS submission
- Provide information to competent authorities on request
- Demonstrate traceability to production plots
- Maintain evidence of due diligence activities

### 2.2 Audit Scope

| Audit Area | Evidence Required |
|---|---|
| Origin Verification | Plot coordinates, geolocation evidence, satellite imagery |
| Deforestation Status | Forest monitoring data, risk assessments |
| Supply Chain | Chain of custody, transformation records |
| Legal Compliance | Permits, land rights, local law compliance |
| Due Diligence | Risk assessments, mitigation actions |

---

## 3. Data Model

```sql
-- Audit Trails
CREATE TABLE audit_trails (
    trail_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(100) NOT NULL,
    entity_id UUID NOT NULL,

    -- Audit Context
    audit_category VARCHAR(100) NOT NULL,
    action_type VARCHAR(100) NOT NULL,
    action_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Actor
    actor_type VARCHAR(50) NOT NULL,  -- USER, SYSTEM, AGENT, API
    actor_id VARCHAR(255) NOT NULL,
    actor_name VARCHAR(255),
    actor_organization VARCHAR(255),
    ip_address INET,

    -- Change Details
    previous_state JSONB,
    new_state JSONB,
    change_summary TEXT,

    -- Evidence
    evidence_links UUID[] DEFAULT '{}',
    document_references JSONB DEFAULT '[]',

    -- Metadata
    session_id UUID,
    request_id VARCHAR(100),
    correlation_id UUID,

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_entity_type CHECK (
        entity_type IN (
            'PLOT', 'BATCH', 'SHIPMENT', 'TRANSFORMATION',
            'SUPPLIER', 'FACILITY', 'DDS', 'RISK_ASSESSMENT',
            'CUSTODY_TRANSFER', 'DECLARATION', 'CERTIFICATE'
        )
    ),
    CONSTRAINT valid_action_type CHECK (
        action_type IN (
            'CREATE', 'UPDATE', 'DELETE', 'VERIFY', 'APPROVE',
            'REJECT', 'SUBMIT', 'EXPORT', 'IMPORT', 'LINK',
            'UNLINK', 'SPLIT', 'MERGE', 'TRANSFORM', 'TRANSFER'
        )
    )
);

-- Evidence Records
CREATE TABLE evidence_records (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    evidence_type VARCHAR(100) NOT NULL,

    -- Reference
    related_entity_type VARCHAR(100) NOT NULL,
    related_entity_id UUID NOT NULL,

    -- Evidence Details
    title VARCHAR(500) NOT NULL,
    description TEXT,
    evidence_date TIMESTAMP NOT NULL,

    -- Storage
    storage_type VARCHAR(50) NOT NULL,  -- DOCUMENT, IMAGE, DATA, EXTERNAL
    storage_location VARCHAR(1000),
    file_hash VARCHAR(128),  -- SHA-512 hash
    file_size_bytes BIGINT,
    mime_type VARCHAR(100),

    -- Verification
    is_verified BOOLEAN DEFAULT FALSE,
    verified_by VARCHAR(255),
    verified_at TIMESTAMP,
    verification_method VARCHAR(100),

    -- Retention
    retention_required_until DATE NOT NULL,
    is_archived BOOLEAN DEFAULT FALSE,

    -- Integrity
    integrity_hash VARCHAR(128) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_evidence_type CHECK (
        evidence_type IN (
            'GEOLOCATION_DATA', 'SATELLITE_IMAGE', 'CERTIFICATE',
            'PERMIT', 'INVOICE', 'TRANSPORT_DOC', 'CUSTOMS_DOC',
            'AUDIT_REPORT', 'RISK_ASSESSMENT', 'DECLARATION',
            'PHOTO', 'CONTRACT', 'CORRESPONDENCE'
        )
    )
);

-- Compliance Audits
CREATE TABLE compliance_audits (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_reference VARCHAR(100) UNIQUE NOT NULL,

    -- Audit Type
    audit_type VARCHAR(100) NOT NULL,
    audit_scope JSONB NOT NULL,

    -- Timeline
    initiated_date TIMESTAMP NOT NULL,
    target_completion DATE,
    actual_completion TIMESTAMP,

    -- Auditor
    auditor_type VARCHAR(50) NOT NULL,  -- INTERNAL, EXTERNAL, AUTHORITY
    auditor_name VARCHAR(255),
    auditor_organization VARCHAR(255),
    audit_body_accreditation VARCHAR(100),

    -- Scope
    operator_id UUID NOT NULL,
    dds_references VARCHAR(100)[],
    batch_ids UUID[],
    supplier_ids UUID[],

    -- Results
    status VARCHAR(50) DEFAULT 'INITIATED',
    overall_result VARCHAR(50),
    findings JSONB DEFAULT '[]',
    non_conformities JSONB DEFAULT '[]',
    observations JSONB DEFAULT '[]',

    -- Actions
    corrective_actions JSONB DEFAULT '[]',
    follow_up_required BOOLEAN DEFAULT FALSE,
    follow_up_date DATE,

    -- Documentation
    audit_report_id UUID,
    evidence_package_id UUID,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_audit_type CHECK (
        audit_type IN (
            'INTERNAL_REVIEW', 'SUPPLIER_AUDIT', 'CERTIFICATION_AUDIT',
            'AUTHORITY_INSPECTION', 'THIRD_PARTY_VERIFICATION', 'SPOT_CHECK'
        )
    ),
    CONSTRAINT valid_status CHECK (
        status IN (
            'INITIATED', 'PLANNING', 'IN_PROGRESS', 'REVIEW',
            'COMPLETED', 'CLOSED', 'CANCELLED'
        )
    ),
    CONSTRAINT valid_result CHECK (
        overall_result IS NULL OR overall_result IN (
            'CONFORMING', 'MINOR_NON_CONFORMITY', 'MAJOR_NON_CONFORMITY',
            'CRITICAL_NON_CONFORMITY', 'INCONCLUSIVE'
        )
    )
);

-- Audit Findings
CREATE TABLE audit_findings (
    finding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id UUID REFERENCES compliance_audits(audit_id),

    -- Finding
    finding_type VARCHAR(50) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    category VARCHAR(100) NOT NULL,

    -- Details
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    evidence_references UUID[],

    -- Related Entities
    related_entity_type VARCHAR(100),
    related_entity_id UUID,

    -- EUDR Reference
    eudr_article VARCHAR(50),
    requirement_reference VARCHAR(255),

    -- Response
    response_required BOOLEAN DEFAULT TRUE,
    response_deadline DATE,
    operator_response TEXT,

    -- Resolution
    status VARCHAR(50) DEFAULT 'OPEN',
    resolution_date TIMESTAMP,
    resolution_notes TEXT,

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_finding_type CHECK (
        finding_type IN ('NON_CONFORMITY', 'OBSERVATION', 'OPPORTUNITY')
    ),
    CONSTRAINT valid_severity CHECK (
        severity IN ('CRITICAL', 'MAJOR', 'MINOR', 'OBSERVATION')
    )
);

-- Traceability Verification
CREATE TABLE traceability_verifications (
    verification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Request
    request_type VARCHAR(100) NOT NULL,
    requested_by VARCHAR(255) NOT NULL,
    requested_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Scope
    dds_reference VARCHAR(100),
    batch_ids UUID[],
    target_date_range DATERANGE,

    -- Verification Process
    verification_method VARCHAR(100) NOT NULL,
    automated_checks JSONB DEFAULT '[]',
    manual_checks JSONB DEFAULT '[]',

    -- Results
    status VARCHAR(50) DEFAULT 'PENDING',
    completed_at TIMESTAMP,

    -- Traceability Chain
    chain_complete BOOLEAN,
    chain_gaps JSONB DEFAULT '[]',
    origin_plots_verified UUID[],
    custody_chain_verified BOOLEAN,

    -- Evidence Package
    evidence_summary JSONB,
    evidence_ids UUID[],

    -- Overall Result
    verification_result VARCHAR(50),
    confidence_score DECIMAL(5,2),
    issues_found JSONB DEFAULT '[]',

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_request_type CHECK (
        request_type IN (
            'AUTHORITY_REQUEST', 'INTERNAL_AUDIT', 'CUSTOMER_REQUEST',
            'PERIODIC_REVIEW', 'INCIDENT_INVESTIGATION'
        )
    ),
    CONSTRAINT valid_verification_result CHECK (
        verification_result IS NULL OR verification_result IN (
            'VERIFIED', 'PARTIALLY_VERIFIED', 'UNVERIFIED', 'INCONCLUSIVE'
        )
    )
);

-- Indexes
CREATE INDEX idx_audit_trails_entity ON audit_trails(entity_type, entity_id);
CREATE INDEX idx_audit_trails_timestamp ON audit_trails(action_timestamp);
CREATE INDEX idx_audit_trails_actor ON audit_trails(actor_id);
CREATE INDEX idx_audit_trails_category ON audit_trails(audit_category);
CREATE INDEX idx_evidence_entity ON evidence_records(related_entity_type, related_entity_id);
CREATE INDEX idx_evidence_type ON evidence_records(evidence_type);
CREATE INDEX idx_compliance_audits_operator ON compliance_audits(operator_id);
CREATE INDEX idx_compliance_audits_status ON compliance_audits(status);
CREATE INDEX idx_findings_audit ON audit_findings(audit_id);
CREATE INDEX idx_findings_status ON audit_findings(status);
CREATE INDEX idx_verifications_dds ON traceability_verifications(dds_reference);
```

---

## 4. Functional Requirements

### 4.1 Audit Trail Management
- **FR-001 (P0):** Capture all CRUD operations on traceability entities
- **FR-002 (P0):** Record actor identity and timestamp for every action
- **FR-003 (P0):** Store before/after state for modifications
- **FR-004 (P0):** Link audit entries to evidence documents
- **FR-005 (P0):** Support correlation across related actions

### 4.2 Evidence Management
- **FR-010 (P0):** Store evidence documents with integrity hashes
- **FR-011 (P0):** Link evidence to traceability entities
- **FR-012 (P0):** Verify evidence authenticity
- **FR-013 (P0):** Enforce 5-year retention policy
- **FR-014 (P1):** Archive evidence with retrieval capability

### 4.3 Compliance Audits
- **FR-020 (P0):** Support internal audit workflow
- **FR-021 (P0):** Track audit findings and responses
- **FR-022 (P0):** Manage corrective actions
- **FR-023 (P0):** Support authority inspection requests
- **FR-024 (P1):** Generate audit evidence packages

### 4.4 Traceability Verification
- **FR-030 (P0):** Verify complete chain from EU to origin
- **FR-031 (P0):** Identify gaps in traceability chain
- **FR-032 (P0):** Calculate traceability confidence scores
- **FR-033 (P0):** Generate verification reports
- **FR-034 (P1):** Automated verification checks

### 4.5 Regulatory Response
- **FR-040 (P0):** Package evidence for authority requests
- **FR-041 (P0):** Track response timelines
- **FR-042 (P0):** Export data in required formats
- **FR-043 (P1):** Support investigation workflows

---

## 5. Traceability Verification Engine

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, date
from decimal import Decimal
from enum import Enum


class VerificationStatus(Enum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    INCONCLUSIVE = "INCONCLUSIVE"


class ChainGapType(Enum):
    MISSING_CUSTODY_TRANSFER = "MISSING_CUSTODY_TRANSFER"
    UNVERIFIED_ORIGIN = "UNVERIFIED_ORIGIN"
    MISSING_TRANSFORMATION = "MISSING_TRANSFORMATION"
    TEMPORAL_GAP = "TEMPORAL_GAP"
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"


@dataclass
class ChainGap:
    gap_type: ChainGapType
    location_in_chain: str
    description: str
    severity: str  # CRITICAL, MAJOR, MINOR
    affected_entity_type: str
    affected_entity_id: UUID
    evidence_required: List[str]


@dataclass
class VerificationResult:
    verification_id: UUID
    status: VerificationStatus
    chain_complete: bool
    confidence_score: Decimal
    origin_plots_verified: List[UUID]
    custody_chain_verified: bool
    gaps: List[ChainGap]
    evidence_summary: Dict[str, Any]
    issues: List[Dict[str, Any]]


class TraceabilityVerificationEngine:
    """
    Verifies complete traceability chains from EU market to origin plots.
    """

    def __init__(self, db_session, evidence_service, chain_service):
        self.db = db_session
        self.evidence_service = evidence_service
        self.chain_service = chain_service

    def verify_batch_traceability(
        self,
        batch_id: UUID,
        verification_depth: str = "FULL"
    ) -> VerificationResult:
        """
        Verify complete traceability for a batch.

        Args:
            batch_id: The batch to verify
            verification_depth: FULL, SUMMARY, or QUICK

        Returns:
            VerificationResult with chain status and gaps
        """
        gaps = []
        evidence_summary = {
            "origin_evidence": [],
            "custody_evidence": [],
            "transformation_evidence": [],
            "certification_evidence": []
        }

        # Step 1: Get batch details
        batch = self.get_batch(batch_id)
        if not batch:
            return self._create_failed_result(
                "Batch not found",
                ChainGapType.MISSING_CUSTODY_TRANSFER
            )

        # Step 2: Verify origin plots
        origin_verification = self._verify_origin_plots(batch)
        gaps.extend(origin_verification.gaps)
        evidence_summary["origin_evidence"] = origin_verification.evidence

        # Step 3: Verify custody chain
        custody_verification = self._verify_custody_chain(batch)
        gaps.extend(custody_verification.gaps)
        evidence_summary["custody_evidence"] = custody_verification.evidence

        # Step 4: Verify transformations
        if batch.has_transformations:
            transform_verification = self._verify_transformations(batch)
            gaps.extend(transform_verification.gaps)
            evidence_summary["transformation_evidence"] = transform_verification.evidence

        # Step 5: Verify certifications
        cert_verification = self._verify_certifications(batch)
        evidence_summary["certification_evidence"] = cert_verification.evidence

        # Calculate confidence score
        confidence_score = self._calculate_confidence(gaps, evidence_summary)

        # Determine overall status
        status = self._determine_status(gaps, confidence_score)

        return VerificationResult(
            verification_id=UUID(self._generate_uuid()),
            status=status,
            chain_complete=len(gaps) == 0,
            confidence_score=confidence_score,
            origin_plots_verified=origin_verification.verified_plots,
            custody_chain_verified=custody_verification.is_complete,
            gaps=gaps,
            evidence_summary=evidence_summary,
            issues=self._summarize_issues(gaps)
        )

    def _verify_origin_plots(self, batch) -> dict:
        """
        Verify origin plot documentation.
        """
        verified_plots = []
        gaps = []
        evidence = []

        for plot_id in batch.origin_plots:
            plot = self.get_plot(plot_id)

            if not plot:
                gaps.append(ChainGap(
                    gap_type=ChainGapType.UNVERIFIED_ORIGIN,
                    location_in_chain="ORIGIN",
                    description=f"Plot {plot_id} not found in registry",
                    severity="CRITICAL",
                    affected_entity_type="PLOT",
                    affected_entity_id=plot_id,
                    evidence_required=["plot_registration", "geolocation_data"]
                ))
                continue

            # Check geolocation evidence
            if not self._has_valid_geolocation(plot):
                gaps.append(ChainGap(
                    gap_type=ChainGapType.MISSING_EVIDENCE,
                    location_in_chain="ORIGIN",
                    description=f"Plot {plot_id} missing valid geolocation evidence",
                    severity="CRITICAL",
                    affected_entity_type="PLOT",
                    affected_entity_id=plot_id,
                    evidence_required=["geolocation_data", "coordinate_verification"]
                ))
            else:
                evidence.append({
                    "plot_id": str(plot_id),
                    "evidence_type": "GEOLOCATION",
                    "coordinates": plot.coordinates,
                    "precision": plot.coordinate_precision
                })

            # Check deforestation verification
            if not plot.deforestation_status_verified:
                gaps.append(ChainGap(
                    gap_type=ChainGapType.MISSING_EVIDENCE,
                    location_in_chain="ORIGIN",
                    description=f"Plot {plot_id} deforestation status not verified",
                    severity="MAJOR",
                    affected_entity_type="PLOT",
                    affected_entity_id=plot_id,
                    evidence_required=["satellite_imagery", "forest_monitoring_data"]
                ))
            else:
                verified_plots.append(plot_id)
                evidence.append({
                    "plot_id": str(plot_id),
                    "evidence_type": "DEFORESTATION_CHECK",
                    "verified_date": str(plot.deforestation_verified_date),
                    "method": plot.deforestation_verification_method
                })

        return {
            "verified_plots": verified_plots,
            "gaps": gaps,
            "evidence": evidence
        }

    def _verify_custody_chain(self, batch) -> dict:
        """
        Verify unbroken chain of custody.
        """
        gaps = []
        evidence = []

        # Get all custody transfers for this batch
        transfers = self.chain_service.get_custody_transfers(batch.batch_id)

        if not transfers:
            gaps.append(ChainGap(
                gap_type=ChainGapType.MISSING_CUSTODY_TRANSFER,
                location_in_chain="CHAIN",
                description="No custody transfers recorded for batch",
                severity="CRITICAL",
                affected_entity_type="BATCH",
                affected_entity_id=batch.batch_id,
                evidence_required=["custody_transfer_records", "transport_documents"]
            ))
            return {"is_complete": False, "gaps": gaps, "evidence": evidence}

        # Verify chain continuity
        sorted_transfers = sorted(transfers, key=lambda t: t.transfer_date)

        for i, transfer in enumerate(sorted_transfers):
            # Check transfer has required evidence
            if not self._has_custody_evidence(transfer):
                gaps.append(ChainGap(
                    gap_type=ChainGapType.MISSING_EVIDENCE,
                    location_in_chain=f"TRANSFER_{i+1}",
                    description=f"Transfer {transfer.transfer_id} missing supporting documents",
                    severity="MAJOR",
                    affected_entity_type="CUSTODY_TRANSFER",
                    affected_entity_id=transfer.transfer_id,
                    evidence_required=["delivery_note", "receipt_confirmation"]
                ))
            else:
                evidence.append({
                    "transfer_id": str(transfer.transfer_id),
                    "from": transfer.from_entity,
                    "to": transfer.to_entity,
                    "date": str(transfer.transfer_date),
                    "documents": transfer.document_references
                })

            # Check for temporal gaps
            if i > 0:
                prev_transfer = sorted_transfers[i-1]
                gap_days = (transfer.transfer_date - prev_transfer.transfer_date).days

                if gap_days > 30:  # More than 30 days between transfers
                    gaps.append(ChainGap(
                        gap_type=ChainGapType.TEMPORAL_GAP,
                        location_in_chain=f"BETWEEN_TRANSFER_{i}_AND_{i+1}",
                        description=f"{gap_days} day gap between transfers",
                        severity="MINOR" if gap_days < 90 else "MAJOR",
                        affected_entity_type="BATCH",
                        affected_entity_id=batch.batch_id,
                        evidence_required=["storage_records", "warehouse_receipts"]
                    ))

        is_complete = len([g for g in gaps if g.severity in ["CRITICAL", "MAJOR"]]) == 0

        return {"is_complete": is_complete, "gaps": gaps, "evidence": evidence}

    def _verify_transformations(self, batch) -> dict:
        """
        Verify transformation records maintain traceability.
        """
        gaps = []
        evidence = []

        transformations = self.get_transformations_for_batch(batch.batch_id)

        for transform in transformations:
            # Verify input-output linkage
            if not self._verify_transformation_linkage(transform):
                gaps.append(ChainGap(
                    gap_type=ChainGapType.MISSING_TRANSFORMATION,
                    location_in_chain="TRANSFORMATION",
                    description=f"Transformation {transform.transformation_id} has incomplete batch linkage",
                    severity="MAJOR",
                    affected_entity_type="TRANSFORMATION",
                    affected_entity_id=transform.transformation_id,
                    evidence_required=["processing_record", "yield_calculation"]
                ))
            else:
                evidence.append({
                    "transformation_id": str(transform.transformation_id),
                    "type": transform.transformation_type,
                    "input_batches": [str(b) for b in transform.input_batches],
                    "output_batches": [str(b) for b in transform.output_batches],
                    "yield_ratio": str(transform.yield_ratio)
                })

            # Verify quantity balances
            if not self._verify_mass_balance(transform):
                gaps.append(ChainGap(
                    gap_type=ChainGapType.QUANTITY_MISMATCH,
                    location_in_chain="TRANSFORMATION",
                    description=f"Transformation {transform.transformation_id} has quantity mismatch",
                    severity="MAJOR",
                    affected_entity_type="TRANSFORMATION",
                    affected_entity_id=transform.transformation_id,
                    evidence_required=["mass_balance_records", "processing_yields"]
                ))

        return {"gaps": gaps, "evidence": evidence}

    def _verify_certifications(self, batch) -> dict:
        """
        Verify certification evidence.
        """
        evidence = []

        certs = self.get_certifications_for_batch(batch.batch_id)

        for cert in certs:
            if cert.is_valid and cert.expiry_date > date.today():
                evidence.append({
                    "certification_type": cert.certification_type,
                    "certificate_number": cert.certificate_number,
                    "issuer": cert.issuer,
                    "valid_until": str(cert.expiry_date)
                })

        return {"evidence": evidence}

    def _calculate_confidence(
        self,
        gaps: List[ChainGap],
        evidence_summary: Dict[str, Any]
    ) -> Decimal:
        """
        Calculate confidence score based on gaps and evidence.
        """
        base_score = Decimal("100.0")

        # Deduct for gaps
        for gap in gaps:
            if gap.severity == "CRITICAL":
                base_score -= Decimal("25.0")
            elif gap.severity == "MAJOR":
                base_score -= Decimal("10.0")
            elif gap.severity == "MINOR":
                base_score -= Decimal("5.0")

        # Ensure minimum of 0
        return max(base_score, Decimal("0.0"))

    def _determine_status(
        self,
        gaps: List[ChainGap],
        confidence_score: Decimal
    ) -> VerificationStatus:
        """
        Determine overall verification status.
        """
        critical_gaps = [g for g in gaps if g.severity == "CRITICAL"]
        major_gaps = [g for g in gaps if g.severity == "MAJOR"]

        if len(critical_gaps) > 0:
            return VerificationStatus.UNVERIFIED
        elif len(major_gaps) > 0:
            return VerificationStatus.PARTIALLY_VERIFIED
        elif confidence_score >= Decimal("90.0"):
            return VerificationStatus.VERIFIED
        elif confidence_score >= Decimal("70.0"):
            return VerificationStatus.PARTIALLY_VERIFIED
        else:
            return VerificationStatus.INCONCLUSIVE

    def _summarize_issues(self, gaps: List[ChainGap]) -> List[Dict[str, Any]]:
        """
        Summarize gaps as actionable issues.
        """
        return [
            {
                "type": gap.gap_type.value,
                "severity": gap.severity,
                "location": gap.location_in_chain,
                "description": gap.description,
                "action_required": gap.evidence_required
            }
            for gap in gaps
        ]

    # Placeholder methods for database access
    def get_batch(self, batch_id: UUID): pass
    def get_plot(self, plot_id: UUID): pass
    def get_transformations_for_batch(self, batch_id: UUID): pass
    def get_certifications_for_batch(self, batch_id: UUID): pass
    def _has_valid_geolocation(self, plot) -> bool: pass
    def _has_custody_evidence(self, transfer) -> bool: pass
    def _verify_transformation_linkage(self, transform) -> bool: pass
    def _verify_mass_balance(self, transform) -> bool: pass
    def _generate_uuid(self) -> str: pass


class AuditTrailService:
    """
    Service for recording and querying audit trails.
    """

    def record_action(
        self,
        entity_type: str,
        entity_id: UUID,
        action_type: str,
        actor_id: str,
        actor_type: str = "USER",
        previous_state: Optional[Dict] = None,
        new_state: Optional[Dict] = None,
        evidence_links: Optional[List[UUID]] = None,
        correlation_id: Optional[UUID] = None
    ) -> UUID:
        """
        Record an action in the audit trail.
        """
        change_summary = self._generate_change_summary(previous_state, new_state)

        entry = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "audit_category": self._categorize_action(entity_type, action_type),
            "action_type": action_type,
            "action_timestamp": datetime.now(),
            "actor_type": actor_type,
            "actor_id": actor_id,
            "previous_state": previous_state,
            "new_state": new_state,
            "change_summary": change_summary,
            "evidence_links": evidence_links or [],
            "correlation_id": correlation_id
        }

        return self._save_trail_entry(entry)

    def get_entity_history(
        self,
        entity_type: str,
        entity_id: UUID,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get complete audit history for an entity.
        """
        query = """
            SELECT * FROM audit_trails
            WHERE entity_type = %s AND entity_id = %s
        """
        params = [entity_type, entity_id]

        if from_date:
            query += " AND action_timestamp >= %s"
            params.append(from_date)
        if to_date:
            query += " AND action_timestamp <= %s"
            params.append(to_date)

        query += " ORDER BY action_timestamp ASC"

        return self._execute_query(query, params)

    def get_correlated_actions(self, correlation_id: UUID) -> List[Dict]:
        """
        Get all actions with the same correlation ID.
        """
        return self._execute_query(
            "SELECT * FROM audit_trails WHERE correlation_id = %s ORDER BY action_timestamp",
            [correlation_id]
        )

    def _generate_change_summary(
        self,
        previous: Optional[Dict],
        new: Optional[Dict]
    ) -> str:
        """
        Generate human-readable change summary.
        """
        if not previous and new:
            return "Entity created"
        if previous and not new:
            return "Entity deleted"
        if not previous or not new:
            return "No changes recorded"

        changes = []
        all_keys = set(previous.keys()) | set(new.keys())

        for key in all_keys:
            old_val = previous.get(key)
            new_val = new.get(key)
            if old_val != new_val:
                changes.append(f"{key}: {old_val} -> {new_val}")

        return "; ".join(changes) if changes else "No changes"

    def _categorize_action(self, entity_type: str, action_type: str) -> str:
        """
        Categorize action for filtering and reporting.
        """
        category_map = {
            "PLOT": "ORIGIN_DATA",
            "BATCH": "TRACEABILITY",
            "SHIPMENT": "LOGISTICS",
            "TRANSFORMATION": "PROCESSING",
            "SUPPLIER": "SUPPLIER_MANAGEMENT",
            "DDS": "REGULATORY",
            "RISK_ASSESSMENT": "RISK_MANAGEMENT"
        }
        return category_map.get(entity_type, "OTHER")

    # Placeholder methods
    def _save_trail_entry(self, entry: Dict) -> UUID: pass
    def _execute_query(self, query: str, params: List) -> List[Dict]: pass
```

---

## 6. API Specification

```yaml
openapi: 3.0.3
info:
  title: Traceability Audit Agent API
  version: 1.0.0

paths:
  /api/v1/audit/trails:
    get:
      summary: Query audit trails
      parameters:
        - name: entity_type
          in: query
          schema:
            type: string
        - name: entity_id
          in: query
          schema:
            type: string
            format: uuid
        - name: from_date
          in: query
          schema:
            type: string
            format: date-time
        - name: to_date
          in: query
          schema:
            type: string
            format: date-time
        - name: actor_id
          in: query
          schema:
            type: string

  /api/v1/audit/evidence:
    post:
      summary: Upload evidence document
      requestBody:
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                file:
                  type: string
                  format: binary
                related_entity_type:
                  type: string
                related_entity_id:
                  type: string
                  format: uuid
                evidence_type:
                  type: string
    get:
      summary: List evidence records

  /api/v1/audit/evidence/{evidence_id}:
    get:
      summary: Get evidence details
    delete:
      summary: Mark evidence as archived

  /api/v1/audit/verify:
    post:
      summary: Initiate traceability verification
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                batch_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                dds_reference:
                  type: string
                verification_depth:
                  type: string
                  enum: [FULL, SUMMARY, QUICK]

  /api/v1/audit/verify/{verification_id}:
    get:
      summary: Get verification result

  /api/v1/audit/compliance-audits:
    post:
      summary: Create compliance audit
    get:
      summary: List audits

  /api/v1/audit/compliance-audits/{audit_id}:
    get:
      summary: Get audit details
    patch:
      summary: Update audit status/findings

  /api/v1/audit/compliance-audits/{audit_id}/findings:
    post:
      summary: Add finding to audit
    get:
      summary: List audit findings

  /api/v1/audit/compliance-audits/{audit_id}/evidence-package:
    get:
      summary: Generate evidence package for audit
      description: Compiles all relevant evidence into downloadable package

  /api/v1/audit/authority-request:
    post:
      summary: Handle authority information request
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                request_reference:
                  type: string
                authority_name:
                  type: string
                dds_references:
                  type: array
                  items:
                    type: string
                requested_information:
                  type: array
                  items:
                    type: string
                deadline:
                  type: string
                  format: date

  /api/v1/audit/retention/status:
    get:
      summary: Get retention compliance status

  /api/v1/audit/export:
    post:
      summary: Export audit data
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                entity_type:
                  type: string
                entity_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                format:
                  type: string
                  enum: [JSON, CSV, PDF]
                include_evidence:
                  type: boolean
```

---

## 7. Non-Functional Requirements

### 7.1 Performance
- Audit trail writes: <50ms
- History queries: <500ms for 10,000 entries
- Verification: <30 seconds for full chain
- Evidence upload: <5 seconds for 50MB file

### 7.2 Retention
- All audit data retained for minimum 5 years
- Evidence documents retained for 7 years
- Automatic archival after retention period
- Retrieval from archive within 24 hours

### 7.3 Security
- Tamper-evident audit trail (hash chain)
- Evidence integrity verification via SHA-512
- Role-based access to audit data
- Encryption at rest for evidence documents

### 7.4 Availability
- 99.9% availability for audit writes
- Async processing for verification
- Graceful degradation for evidence service

---

## 8. Integration Points

| System | Integration Type | Purpose |
|---|---|---|
| GL-EUDR-007 Chain of Custody | Event subscription | Capture custody changes |
| GL-EUDR-008 Batch Tracking | Event subscription | Capture batch lifecycle |
| GL-EUDR-005 Plot Registry | Query | Verify origin plots |
| GL-EUDR-003 Commodity Traceability | Query | Verify transformations |
| DDS Submission Service | Event subscription | Capture DDS submissions |
| Document Management | API | Evidence storage |

---

## 9. Success Metrics

- **Trail Completeness:** 100% of actions logged
- **Evidence Integrity:** 100% verified hashes
- **Verification Coverage:** 100% of DDS can be traced
- **Retention Compliance:** 100% within policy
- **Authority Response:** <48 hours for standard requests

---

*Document Version: 1.0*
*Created: 2026-01-30*
*Status: APPROVED FOR IMPLEMENTATION*
