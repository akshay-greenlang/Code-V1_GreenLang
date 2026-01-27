# ADR-005: Hash-Chained Audit Events

**Date:** 2026-01-27
**Status:** Accepted
**Deciders:** Architecture Team, Governance/Compliance, Security Team
**Consulted:** Legal, External Auditors, Platform Engineering

---

## Context

### Problem Statement
GL-FOUND-X-001 (GreenLang Orchestrator) must produce tamper-evident audit trails for regulatory compliance. Auditors and regulators need assurance that:
- Audit records have not been modified after creation
- The sequence of events is verifiable
- Missing events can be detected
- The audit trail can be cryptographically verified

### Current Situation
- **Regulatory Drivers:** CBAM, CSRD, GHG Protocol require auditable calculations
- **Third-Party Audit:** External auditors verify emission calculations annually
- **Legal Hold:** Some records must be preserved for 7+ years
- **Tamper Evidence:** Current logging provides no cryptographic guarantees

### Business Impact
- **Regulatory Compliance:** Tamper-evident trails reduce audit friction
- **Legal Protection:** Cryptographic proof of record integrity
- **Customer Trust:** Enterprise customers require audit assurance
- **Certification:** Enables third-party assurance certifications (ISO 14064, etc.)

---

## Decision

### What We're Implementing
**Hash-chained audit events** at MVP, with optional full PKI signing as a future enhancement.

### Core Design

1. **Hash Chain Structure**
   - Each event includes hash of previous event
   - Creates cryptographic link between sequential events
   - Tampering breaks the chain and is detectable

2. **Event Schema**
   ```json
   {
     "event_id": "evt_abc123",
     "sequence_number": 42,
     "timestamp": "2026-01-27T10:30:00.000Z",
     "run_id": "run_xyz789",
     "event_type": "STEP_SUCCEEDED",
     "event_data": {
       "step_id": "stp_ingest_001",
       "duration_ms": 120034,
       "artifact_count": 2
     },
     "prev_event_hash": "sha256:abc123def456...",
     "event_hash": "sha256:789xyz012abc...",
     "chain_id": "chain_run_xyz789"
   }
   ```

3. **Hash Computation**
   ```python
   def compute_event_hash(event: AuditEvent) -> str:
       """Compute hash for event, including link to previous."""
       canonical = json.dumps({
           "event_id": event.event_id,
           "sequence_number": event.sequence_number,
           "timestamp": event.timestamp.isoformat(),
           "run_id": event.run_id,
           "event_type": event.event_type,
           "event_data": event.event_data,
           "prev_event_hash": event.prev_event_hash
       }, sort_keys=True, separators=(',', ':'))
       return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
   ```

4. **Chain Architecture**
   ```
   Genesis Event (prev_hash = null)
        |
        v
   +-------------+     +-------------+     +-------------+
   | Event 1     | --> | Event 2     | --> | Event 3     |
   | hash: abc   |     | prev: abc   |     | prev: def   |
   |             |     | hash: def   |     | hash: ghi   |
   +-------------+     +-------------+     +-------------+
   ```

### Chain Scopes
- **Per-Run Chain:** Events for a single run form one chain
- **Per-Namespace Chain:** Optional global chain across all runs (for high-assurance)
- **Chain Anchoring:** Periodic anchors to external timestamping service (future)

### Technology Stack
- **Hashing:** SHA-256 (FIPS 180-4 compliant)
- **Storage:** PostgreSQL (primary), S3 (backup/archive)
- **Serialization:** Canonical JSON (RFC 8785)
- **Verification:** Python library, CLI tool

### Code Location
- `greenlang/orchestrator/audit/`
  - `events.py` - Event types and schema
  - `chain.py` - Hash chain management
  - `writer.py` - Event writing with chain linking
  - `verifier.py` - Chain verification
  - `exporter.py` - Audit package export

---

## Rationale

### Why Hash Chains at MVP

**1. Basic Tamper Evidence**
- Any modification breaks the chain
- Missing events detectable via sequence gaps
- Low computational overhead

**2. No Complex Infrastructure**
- No PKI/CA infrastructure required
- No HSM or key management needed
- Can run entirely in-process

**3. Audit-Friendly**
- Simple verification algorithm
- Auditors can verify with basic tools
- Clear chain of evidence

**4. Foundation for Enhancement**
- Can add signing later without breaking compatibility
- Can add external anchoring (blockchain, TSA)
- Progressive enhancement path

**5. Industry Precedent**
- Similar to blockchain/distributed ledger concepts
- Used in Git, Certificate Transparency, etc.
- Well-understood security properties

---

## Alternatives Considered

### Alternative 1: No Cryptographic Signing
**Pros:**
- Simplest implementation
- No computational overhead
- No cryptographic complexity

**Cons:**
- No tamper evidence
- Cannot detect modifications
- Auditors must trust database integrity
- Does not meet regulatory expectations

**Why Rejected:** Does not provide required assurance for regulatory compliance. Auditors increasingly expect cryptographic guarantees.

### Alternative 2: Full PKI Signing (Every Event)
**Pros:**
- Strongest cryptographic guarantee
- Non-repudiation via digital signatures
- Industry-standard approach

**Cons:**
- Requires PKI infrastructure (CA, key management)
- HSM costs for production-grade security
- Key rotation complexity
- Higher computational overhead
- Overkill for MVP

**Why Rejected:** Infrastructure complexity too high for MVP. Hash chains provide sufficient tamper evidence for initial compliance requirements.

### Alternative 3: WORM Storage Only
**Pros:**
- Hardware-enforced immutability
- Compliance-focused storage solutions exist
- No cryptographic complexity

**Cons:**
- Vendor-specific solutions
- Does not prove sequence or completeness
- Cannot detect modifications at rest
- Expensive for high-volume data

**Why Rejected:** WORM alone does not prove chain integrity or detect missing events. Can be used in combination with hash chains.

### Alternative 4: Blockchain/DLT
**Pros:**
- Decentralized tamper evidence
- Public verifiability
- Strong integrity guarantees

**Cons:**
- Massive infrastructure complexity
- High transaction costs
- Latency for finality
- Regulatory uncertainty in some jurisdictions
- Overkill for single-organization audit

**Why Rejected:** Complexity and cost far exceed requirements. Hash chains provide equivalent guarantees for single-organization use case.

---

## Consequences

### Positive
- **Tamper Evidence:** Modifications detected via broken chain
- **Sequence Integrity:** Missing events detected via sequence gaps
- **Simple Verification:** Auditors can verify with basic tools
- **Low Overhead:** SHA-256 computation is fast
- **Future-Proof:** Foundation for enhanced signing later
- **Compliance:** Meets initial regulatory requirements

### Negative
- **Not Non-Repudiation:** Hash chains alone don't prove who created events
- **Key Compromise Risk:** If hash algorithm weakened, chain integrity affected
- **Storage Overhead:** Each event stores two hashes (prev + current)
- **Performance:** Sequential hash computation adds latency (minimal)
- **Complexity:** Developers must maintain chain integrity

### Neutral
- **Algorithm Choice:** SHA-256 widely accepted but may need upgrade path
- **Verification Tools:** Must build/maintain verification tooling

---

## Implementation Plan

### Phase 1: Event Schema (Week 1)
1. Define canonical event types (RUN_SUBMITTED, STEP_STARTED, etc.)
2. Implement event data models with Pydantic
3. Create canonical JSON serialization
4. Add sequence number generation

### Phase 2: Hash Chain (Week 2)
1. Implement hash computation function
2. Build chain writer with prev_hash linking
3. Add genesis event handling
4. Create chain storage in PostgreSQL

### Phase 3: Verification (Week 3)
1. Implement chain verification algorithm
2. Build verification CLI tool
3. Add gap detection for missing events
4. Create verification reports

### Phase 4: Export (Week 4)
1. Implement audit package export
2. Add S3 archival for completed chains
3. Create human-readable audit reports
4. Build integration tests

---

## Event Types

### Run Lifecycle Events
| Event Type | Description | Key Data |
|------------|-------------|----------|
| `RUN_SUBMITTED` | Pipeline run submitted | pipeline_hash, submitter |
| `PLAN_COMPILED` | Execution plan generated | plan_hash, step_count |
| `POLICY_EVALUATED` | Pre-run policy check | decision, reason_codes |
| `RUN_STARTED` | Execution began | start_time |
| `RUN_SUCCEEDED` | Run completed successfully | duration_ms, artifact_count |
| `RUN_FAILED` | Run failed | error_code, failed_step |
| `RUN_CANCELED` | Run was canceled | canceled_by, reason |

### Step Lifecycle Events
| Event Type | Description | Key Data |
|------------|-------------|----------|
| `STEP_READY` | Step dependencies satisfied | step_id |
| `STEP_STARTED` | Step execution began | step_id, agent_version |
| `STEP_RETRIED` | Step retry attempted | step_id, attempt, error |
| `STEP_SUCCEEDED` | Step completed successfully | step_id, duration_ms |
| `STEP_FAILED` | Step failed | step_id, error_code |
| `STEP_TIMED_OUT` | Step exceeded timeout | step_id, timeout_seconds |

### Artifact Events
| Event Type | Description | Key Data |
|------------|-------------|----------|
| `ARTIFACT_WRITTEN` | Artifact stored | artifact_id, sha256, uri |
| `ARTIFACT_READ` | Artifact accessed | artifact_id, reader_step |
| `ARTIFACT_DELETED` | Artifact deleted per retention | artifact_id, policy |

---

## Verification Algorithm

```python
def verify_chain(events: List[AuditEvent]) -> VerificationResult:
    """Verify integrity of event chain."""
    errors = []

    # Sort by sequence number
    sorted_events = sorted(events, key=lambda e: e.sequence_number)

    # Check genesis event
    if sorted_events[0].prev_event_hash is not None:
        errors.append("Genesis event has non-null prev_hash")

    # Verify chain links
    for i, event in enumerate(sorted_events):
        # Verify sequence continuity
        if i > 0:
            expected_seq = sorted_events[i-1].sequence_number + 1
            if event.sequence_number != expected_seq:
                errors.append(f"Sequence gap: {expected_seq} missing")

            # Verify hash link
            if event.prev_event_hash != sorted_events[i-1].event_hash:
                errors.append(f"Chain broken at event {event.event_id}")

        # Verify event hash
        computed_hash = compute_event_hash(event)
        if computed_hash != event.event_hash:
            errors.append(f"Hash mismatch at event {event.event_id}")

    return VerificationResult(
        valid=len(errors) == 0,
        errors=errors,
        event_count=len(sorted_events)
    )
```

---

## Compliance & Security

### Security Considerations
- **Hash Algorithm:** SHA-256 (FIPS 180-4, NIST approved)
- **Canonical Serialization:** RFC 8785 for deterministic JSON
- **Storage Security:** Events encrypted at rest
- **Access Control:** Read-only access for audit queries

### Compliance Considerations
- **Immutability:** Append-only event log
- **Completeness:** Sequence numbers detect gaps
- **Verifiability:** Auditors can independently verify
- **Retention:** Events archived per policy

### Future Enhancements (Post-MVP)
1. **Digital Signatures:** Sign chain checkpoints with platform key
2. **External Anchoring:** Publish periodic hashes to timestamping service
3. **Multi-Party Verification:** Distribute chain to multiple verifiers
4. **HSM Integration:** Hardware-backed signing for high-assurance

---

## Migration Plan

### Short-term (0-6 months)
- Deploy hash-chained audit events
- Build verification tooling
- Train auditors on verification process

### Medium-term (6-12 months)
- Add checkpoint signing with platform key
- Integrate with external timestamping service
- Build audit dashboard

### Long-term (12+ months)
- Evaluate HSM for signing keys
- Consider public anchoring (Certificate Transparency style)
- Third-party audit certification

---

## Links & References

- **PRD:** GL-FOUND-X-001 GreenLang Orchestrator
- **Related ADRs:** ADR-004 (S3 Artifacts), ADR-006 (Deterministic Plans)
- **SHA-256:** [FIPS 180-4](https://csrc.nist.gov/publications/detail/fips/180/4/final)
- **Canonical JSON:** [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785)
- **Certificate Transparency:** [RFC 6962](https://www.rfc-editor.org/rfc/rfc6962)

---

## Updates

### 2026-01-27 - Status: Accepted
ADR approved by Architecture, Governance, and Security teams. Implementation scheduled for Q1 2026.

---

**Template Version:** 1.0
**Last Updated:** 2026-01-27
**ADR Author:** Platform Architecture Team
**Reviewers:** Governance/Compliance, Security Team, External Auditors
