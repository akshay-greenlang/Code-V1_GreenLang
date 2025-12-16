# GreenLang Agent Evaluation Template

## Agent Identification

| Field | Value |
|-------|-------|
| Agent ID | GL-XXX |
| Agent Name | |
| Version | |
| Evaluation Date | |
| Evaluator | |
| Previous Score | N/A |

---

## 1. Architecture Patterns (20 Points Maximum)

### 1.1 Agent Design Architecture (8 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| Single Responsibility Pattern | 2 | | |
| Composition Pattern | 2 | | |
| Strategy Pattern | 2 | | |
| Chain of Responsibility | 2 | | |
| **Subtotal** | **8** | | |

### 1.2 State Management (4 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| Stateful Graph Architecture | 1 | | |
| Checkpointing Capability | 1 | | |
| Time-Travel Debugging | 1 | | |
| Memory Isolation | 1 | | |
| **Subtotal** | **4** | | |

### 1.3 Tool Architecture (4 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| Tool Protocol Interface | 1 | | |
| Tool Registry Implementation | 1 | | |
| Dynamic Tool Loading | 1 | | |
| Tool Documentation | 1 | | |
| **Subtotal** | **4** | | |

### 1.4 Multi-Agent Coordination (4 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| Supervisor Pattern | 1 | | |
| Peer-to-Peer Communication | 1 | | |
| Pipeline Architecture | 1 | | |
| Swarm Pattern Support | 1 | | |
| **Subtotal** | **4** | | |

**Architecture Total: ___ / 20**

---

## 2. Safety and Reliability (20 Points Maximum)

### 2.1 Functional Safety per IEC 61511 (8 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| SIL Assessment Complete | 2 | | |
| Systematic Capability (SC) | 2 | | |
| Architectural Constraints | 2 | | |
| PFDavg/PFH Documentation | 2 | | |
| **Subtotal** | **8** | | |

**Determined SIL Level:** [ ] SIL 1 [ ] SIL 2 [ ] SIL 3 [ ] N/A

### 2.2 Error Handling and Recovery (6 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| Exception Hierarchy | 1 | | |
| Retry with Backoff | 1 | | |
| Circuit Breaker | 2 | | |
| Graceful Degradation | 2 | | |
| **Subtotal** | **6** | | |

### 2.3 Operational Safety (6 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| NFPA Combustion Safety | 2 | | |
| ASME Pressure Safety | 2 | | |
| EPA Emissions Monitoring | 2 | | |
| **Subtotal** | **6** | | |

**Safety Total: ___ / 20**

---

## 3. Explainability Requirements (15 Points Maximum)

### 3.1 AI Explainability per ISO/IEC 22989 (6 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| Explainability Definition | 2 | | |
| Transparency | 2 | | |
| Decision Audit Trail | 2 | | |
| **Subtotal** | **6** | | |

### 3.2 XAI Implementation (6 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| SHAP Implementation | 2 | | |
| LIME Implementation | 2 | | |
| Explanation Latency (<400ms) | 2 | | |
| **Subtotal** | **6** | | |

**Measured Latency:** ___ ms

### 3.3 Regulatory Explainability (3 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| GDPR Compliance | 1 | | |
| Model Cards | 1 | | |
| Data Statements | 1 | | |
| **Subtotal** | **3** | | |

**Explainability Total: ___ / 15**

---

## 4. Testing Standards (15 Points Maximum)

### 4.1 Test Coverage Requirements (5 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| Unit Test Coverage (>=90%) | 2 | | |
| Integration Tests | 1 | | |
| Edge Case Coverage (30+ scenarios) | 1 | | |
| Performance Tests | 1 | | |
| **Subtotal** | **5** | | |

**Actual Coverage:** ___ %
**Number of Test Scenarios:** ___

### 4.2 AI-Specific Testing (5 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| Adversarial Testing | 1 | | |
| Bias Detection | 1 | | |
| Stress Testing | 1 | | |
| Regression Testing | 2 | | |
| **Subtotal** | **5** | | |

### 4.3 Continuous Integration (5 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| CI/CD Pipeline | 2 | | |
| Automated Retraining | 1 | | |
| Version Control | 1 | | |
| Rollback Capability | 1 | | |
| **Subtotal** | **5** | | |

**Testing Total: ___ / 15**

---

## 5. Documentation Requirements (10 Points Maximum)

### 5.1 Model Documentation (4 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| Model Card | 2 | | |
| Datasheet for Datasets | 1 | | |
| System Card | 1 | | |
| **Subtotal** | **4** | | |

### 5.2 API Documentation (3 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| OpenAPI Specification | 1 | | |
| Code Examples | 1 | | |
| Error Documentation | 1 | | |
| **Subtotal** | **3** | | |

### 5.3 Operational Documentation (3 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| Deployment Guide | 1 | | |
| Runbook | 1 | | |
| Architecture Diagram | 1 | | |
| **Subtotal** | **3** | | |

**Documentation Total: ___ / 10**

---

## 6. Integration Patterns (10 Points Maximum)

### 6.1 Enterprise Integration (4 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| ISA-95 Level Integration | 2 | | |
| API-First Design | 1 | | |
| Asynchronous Messaging | 1 | | |
| **Subtotal** | **4** | | |

### 6.2 Industrial Protocol Support (3 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| OPC UA Integration | 1 | | |
| SCADA Compatibility | 1 | | |
| IEC 61131 Alignment | 1 | | |
| **Subtotal** | **3** | | |

### 6.3 Data Integration (3 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| RAG Implementation | 1 | | |
| Vector Database | 1 | | |
| Data Pipeline | 1 | | |
| **Subtotal** | **3** | | |

**Integration Total: ___ / 10**

---

## 7. Zero-Hallucination Requirements (10 Points Maximum)

### 7.1 Calculation Integrity (5 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| Deterministic Tools | 2 | | |
| Provenance Tracking | 1 | | |
| Source Citation | 1 | | |
| Cross-Validation | 1 | | |
| **Subtotal** | **5** | | |

**LLM Temperature Setting:** ___
**Seed Value:** ___

### 7.2 Hallucination Prevention Techniques (3 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| RAG Implementation | 1 | | |
| Guardrails | 1 | | |
| Self-Familiarity Check | 1 | | |
| **Subtotal** | **3** | | |

**Measured Hallucination Rate:** ___ %

### 7.3 Domain-Specific Validation (2 Points)

| Criterion | Points Available | Score | Evidence |
|-----------|-----------------|-------|----------|
| Industry Validation | 1 | | |
| Regulatory Validation | 1 | | |
| **Subtotal** | **2** | | |

**Zero-Hallucination Total: ___ / 10**

---

## Final Score Summary

| Category | Maximum | Score | % |
|----------|---------|-------|---|
| Architecture Patterns | 20 | | |
| Safety and Reliability | 20 | | |
| Explainability | 15 | | |
| Testing Standards | 15 | | |
| Documentation | 10 | | |
| Integration Patterns | 10 | | |
| Zero-Hallucination | 10 | | |
| **TOTAL** | **100** | | |

---

## Certification Decision

| Score Range | Rating | Decision |
|-------------|--------|----------|
| 95-100 | Excellent | [ ] Production Ready - Tier 1 |
| 90-94 | Very Good | [ ] Production Ready - Tier 2 |
| 85-89 | Good | [ ] Limited Production |
| 80-84 | Acceptable | [ ] Staging/UAT Only |
| <80 | Needs Improvement | [ ] Development Only |

**Final Decision:** ________________________

---

## Improvement Recommendations

### Critical (Must Fix Before Production)

1.
2.
3.

### High Priority

1.
2.
3.

### Medium Priority

1.
2.
3.

### Low Priority

1.
2.
3.

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Evaluator | | | |
| Technical Lead | | | |
| Quality Assurance | | | |
| Standards Committee | | | |

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | | Initial evaluation | |

---

## Attachments

- [ ] Test Coverage Report
- [ ] Performance Benchmark Results
- [ ] Security Assessment
- [ ] Architecture Diagram
- [ ] Model Card
- [ ] Datasheet for Datasets
