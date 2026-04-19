# GreenLang Complete Architecture Overview

**Generated:** February 2, 2026
**Status:** Enterprise-Grade Climate Operating System

---

## Executive Summary

GreenLang is an enterprise-grade **Climate Operating System** designed as the "LangChain for Climate Intelligence." The platform provides comprehensive infrastructure for building climate-intelligent applications with focus on regulatory compliance, zero-hallucination calculations, and enterprise scalability.

**Key Metrics:**
- Total Python Files: **12,476**
- Lines of Code: **155,142** (core)
- Total Classes: **9,514**
- Operational Agents: **47-59**
- Emission Factors: **1,000+**
- Test Coverage: **85%**
- Code Grade: **A+ (95/100)**

---

## 1. Layered Architecture

```
+-------------------------------------------------------------------------+
|                        Applications Layer                                 |
|   GL-CSRD-APP  |  GL-CBAM-APP  |  GL-VCCI-APP  |  GL-EUDR-APP           |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                   Agent Framework (47-59 Agents)                         |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                      Intelligence Layer                                   |
|   LLM Providers (OpenAI/Anthropic)  |  RAG Engine  |  Budget Tracker    |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                       Execution Layer                                     |
|   Orchestrator  |  Workflow Engine  |  Async Executor  |  Policy Engine |
+-------------------------------------------------------------------------+
                                    |
+-------------------------------------------------------------------------+
|                     Core Infrastructure                                   |
|   Calculation Engine  |  Emission Factors  |  Auth/Security  |  Cache   |
+-------------------------------------------------------------------------+
```

---

## 2. Agent Pipeline Designs

### GL-CSRD-APP Pipeline (6 Agents)

```
IntakeAgent → CalculatorAgent → MaterialityAgent
      ↓             ↓                 ↓
AggregatorAgent → ReportingAgent → AuditAgent
```

### GL-CBAM-APP Pipeline (3 Agents)

```
ShipmentIntakeAgent → EmissionsCalculatorAgent → ReportingPackagerAgent
```

### GL-VCCI-APP Pipeline (5 Agents)

```
ValueChainIntakeAgent → Scope3CalculatorAgent → HotspotAnalysisAgent
              ↓                                         ↓
      Scope3ReportingAgent  ←  SupplierEngagementAgent
```

---

## 3. Technology Stack

### Core Framework
- Python: 3.10, 3.11, 3.12
- FastAPI: 0.104.0+
- Pydantic: 2.12.5

### Data Processing
- pandas: 2.1.4
- numpy: 1.26.3
- PostgreSQL: 14+
- Redis: 5.0.1

### AI/ML
- Anthropic Claude (claude-3-opus/sonnet)
- OpenAI GPT-4-turbo
- sentence-transformers
- FAISS

### Security
- PyJWT: 2.8.0
- cryptography: 46.0.3
- AES-256 encryption
- mTLS

---

## 4. Data Flow Pattern

```
External Data (ERP, CSV, API)
        ↓
   Intake Agent (Validation)
        ↓
   Process Agent (Calculation)
        ↓
   Analyze Agent (Compliance)
        ↓
   Report Agent (Multi-format)
        ↓
   Audit Agent (Provenance)
        ↓
   SHA-256 Hash Chain
```

---

## 5. Security Architecture

### Zero-Trust Network
- DMZ: WAF, API Gateway, Load Balancers
- Application: K8s clusters, Istio service mesh
- Data: AES-256 encryption, DB monitoring
- Management: PAM, bastion hosts, MFA

### Encryption Standards
- At Rest: AES-256-GCM (AWS KMS)
- In Transit: TLS 1.3, mTLS (Istio)

### RBAC Model
- Super Admin (all permissions, MFA required)
- Platform Admin (platform/user management)
- Data Analyst (read access, dashboards)
- Auditor (audit/compliance read-only)

---

## 6. Scalability

### Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Parse Time | <10ms | 8ms |
| Compile Time | <50ms | 40ms |
| Throughput | 10K ops/sec | 12K ops/sec |
| Latency P99 | <100ms | 95ms |

### 4-Tier Caching

| Tier | Technology | TTL | Hit Rate |
|------|------------|-----|----------|
| L1 | Redis Cluster | 1hr | 80% |
| L2 | PostgreSQL | 24hr | 60% |
| L3 | S3/GCS | 30d | 40% |
| L4 | CDN | 7d | 90% |

**Cost Impact:** $400/agent → $135/agent (66% reduction)

### Multi-Cloud

| Cloud | Share | Regions |
|-------|-------|---------|
| AWS | 60% | us-east-1, eu-west-1 |
| GCP | 30% | us-central1, europe-west1 |
| Azure | 10% | East US, West Europe |

---

## 7. Proven Design Patterns

| Pattern | Purpose | Implementation |
|---------|---------|----------------|
| Zero-Hallucination | Deterministic calculations | No LLM in calculation path |
| Agent Pipeline | Modular processing | 3-6 agents per application |
| Provenance Tracking | Audit compliance | SHA-256 hash chains |
| 4-Tier Caching | Cost optimization | 66% cost reduction |
| Service Mesh | Security/Resilience | Istio with mTLS |
| Event Sourcing | Complete audit trail | Kafka event store |
| Circuit Breaker | Fault tolerance | pybreaker |
| Rate Limiting | API protection | Per-user quotas |

---

## 8. Deployment Architecture

### Kubernetes HPA
- Min replicas: 3
- Max replicas: 100
- Target CPU: 70%

### CI/CD Pipeline
1. Lint (Ruff, Black, MyPy)
2. Test (pytest, 85% coverage)
3. Security (Bandit, SAST, DAST)
4. Build (Docker image)
5. Deploy (K8s rollout)

---

## 9. Monitoring

### Metrics (Prometheus)
- agent_execution_time_seconds
- agent_success_rate
- calculation_throughput_per_second
- api_request_latency_seconds

### Logging (Structured)
- structlog with JSON output
- Request correlation IDs
- Provenance tracking

### Tracing (OpenTelemetry)
- Distributed tracing
- Service dependency mapping

---

*Document maintained by GreenLang Development Team*
*Last updated: February 2, 2026*
