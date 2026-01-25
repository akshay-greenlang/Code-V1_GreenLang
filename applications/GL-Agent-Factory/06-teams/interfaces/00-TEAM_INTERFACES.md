# Agent Factory: Team Interfaces

**Version:** 1.0
**Date:** 2025-12-03
**Program:** Agent Factory

---

## Overview

This document defines how teams collaborate, including handoff points, integration points, meeting cadence, and communication protocols. Clear interfaces reduce friction and enable teams to work autonomously while maintaining program cohesion.

---

## Interface Catalog

| Interface | Provider Team | Consumer Team | Type | Frequency |
|-----------|---------------|---------------|------|-----------|
| **Model API** | ML Platform | AI/Agent | API | Real-time |
| **Evaluation Harness** | ML Platform | AI/Agent, Climate Science | SDK | Real-time |
| **Validation Hooks** | Climate Science | AI/Agent | SDK | Real-time |
| **Agent SDK** | AI/Agent, Platform | All Agent Developers | SDK | Real-time |
| **Agent Registry API** | Platform | AI/Agent, DevOps | API | Real-time |
| **Data Contracts** | Data Engineering | AI/Agent, Platform | Schema | Static |
| **Data Pipelines** | Data Engineering | AI/Agent | API | Batch/Streaming |
| **CI/CD Pipelines** | DevOps | All Teams | Pipeline | On-demand |
| **Observability Stack** | DevOps | All Teams | Monitoring | Real-time |

---

## ML Platform Team Interfaces

### → AI/Agent Team

**Interface:** Model API (for agent code generation)

**Description:** AI/Agent Team calls ML Platform's Model API to generate agent code from AgentSpec files.

**Contract:**
```python
# Request
POST /v1/models/generate
{
  "model_id": "claude-sonnet-4-5",
  "prompt": "Generate Python function to calculate CBAM emissions...",
  "temperature": 0.0,  # Deterministic
  "max_tokens": 4000,
  "metadata": {
    "agent_id": "GL-CBAM-APP",
    "request_id": "req_123456"
  }
}

# Response
{
  "model_id": "claude-sonnet-4-5",
  "generated_text": "def calculate_emissions(...):",
  "tokens_used": 650,
  "latency_ms": 2300,
  "metadata": {
    "request_id": "req_123456"
  }
}
```

**SLA:**
- Availability: 99.95%
- Latency (p95): <3 seconds
- Rate limit: 100 requests/minute (per agent)

**Handoff Points:**
- **Input:** AgentSpec file, prompt template
- **Output:** Generated code snippet
- **Error Handling:** Retry 3× with exponential backoff; fallback to template-based generation

**Meeting Cadence:**
- Daily: Slack updates on model availability
- Weekly: Tech sync (model performance, quality issues)
- Bi-Weekly: Sprint planning

**Integration Testing:**
- Shared test suite in `tests/integration/model_api/`
- Contract tests (Pact) to validate API compatibility
- Performance tests (Locust) for load testing

---

**Interface:** Evaluation Harness (for agent validation)

**Description:** AI/Agent Team uses ML Platform's evaluation harness to validate generated agents against golden tests.

**Contract:**
```python
# greenlang_eval SDK

from greenlang_eval import GoldenTestRunner

# Run golden tests on generated agent
runner = GoldenTestRunner(suite_name="cbam_steel")
result = runner.run(agent_code="path/to/agent.py")

# Result
{
  "suite_name": "cbam_steel",
  "total_tests": 100,
  "passed": 98,
  "failed": 2,
  "pass_rate": 0.98,
  "results": [...]
}
```

**SLA:**
- Test execution time: <5 minutes for 100 tests
- Availability: 99.9%

**Handoff Points:**
- **Input:** Agent code, test suite name
- **Output:** Test results (pass/fail, detailed errors)
- **Error Handling:** Fail fast on critical test failures

**Meeting Cadence:**
- Weekly: Review failed tests, discuss edge cases
- Monthly: Test suite expansion planning

---

### → Climate Science Team

**Interface:** Golden Test Infrastructure

**Description:** Climate Science Team contributes domain-specific golden tests to ML Platform's test suite.

**Contract:**
```yaml
# Golden test contribution process

1. Climate Science creates test case (YAML format)
2. Submit PR to golden-tests repo
3. ML Platform reviews for technical correctness
4. Merge and deploy to evaluation harness
```

**SLA:**
- Test review time: <3 business days
- Test deployment time: <1 day after merge

**Handoff Points:**
- **Input:** Golden test YAML file
- **Output:** Test deployed to evaluation harness
- **Error Handling:** Validation errors returned in PR comments

**Meeting Cadence:**
- Weekly: Review new test cases
- Monthly: Test quality review

---

## AI/Agent Team Interfaces

### → ML Platform Team

(See ML Platform → AI/Agent above)

---

### → Climate Science Team

**Interface:** Validation Hooks Integration

**Description:** AI/Agent Team integrates Climate Science's validation hooks into Agent SDK for runtime validation.

**Contract:**
```python
# greenlang_sdk integration with validation hooks

from greenlang_sdk import Agent
from greenlang_validation import CBAMValidator

class GLCBAMApp(Agent):
    def __init__(self):
        super().__init__(agent_id="GL-CBAM-APP")
        self.validator = CBAMValidator()

    def execute(self, input_data):
        # Process data
        result = self.calculate_emissions(input_data)

        # Validate output
        validation_result = self.validator.validate(result)
        if not validation_result.is_valid:
            raise ValidationError(validation_result.errors)

        return result
```

**SLA:**
- Validation latency: <1 second per agent call
- Validation accuracy: 100% (no false negatives)

**Handoff Points:**
- **Input:** Agent output (emissions, reports, etc.)
- **Output:** Validation result (pass/fail, errors, warnings)
- **Error Handling:** ValidationError raised on failure

**Meeting Cadence:**
- Weekly: Review validation failures
- Bi-Weekly: Validation framework improvements

---

### → Platform Team

**Interface:** Agent Registry API (for agent publishing)

**Description:** AI/Agent Team publishes generated agents to Platform's registry.

**Contract:**
```python
# POST /v1/registry/agents/{agent_id}/versions
{
  "version": "1.0.0",
  "agentspec_url": "https://...",
  "source_code_url": "s3://...",
  "docker_image_url": "ghcr.io/...",
  "quality_score": 95.0,
  "test_coverage": 92.5
}

# Response
{
  "id": "uuid-456",
  "agent_id": "GL-CBAM-APP",
  "version": "1.0.0",
  "certification_status": "pending",
  "created_at": "2025-12-03T10:30:00Z"
}
```

**SLA:**
- API availability: 99.95%
- API latency (p95): <200ms

**Handoff Points:**
- **Input:** Agent artifacts (code, Docker image, metadata)
- **Output:** Agent registered in registry
- **Error Handling:** Rollback on registration failure

**Meeting Cadence:**
- Weekly: Integration sync
- Bi-Weekly: Registry planning

---

### → Data Engineering Team

**Interface:** Data Contracts (for agent data schemas)

**Description:** AI/Agent Team uses Data Engineering's data contracts for agent input/output validation.

**Contract:**
```python
# Agent uses Pydantic models from data contracts

from greenlang_data.contracts import CBAMShipment

class GLCBAMApp(Agent):
    def execute(self, input_data: dict):
        # Validate input against contract
        shipment = CBAMShipment(**input_data)  # Raises ValidationError if invalid

        # Process
        result = self.calculate_emissions(shipment)
        return result
```

**SLA:**
- Contract stability: No breaking changes without 30-day notice
- Contract documentation: 100% coverage

**Handoff Points:**
- **Input:** Raw data (CSV, JSON, API)
- **Output:** Validated data (Pydantic model)
- **Error Handling:** ValidationError with detailed field-level errors

**Meeting Cadence:**
- Weekly: Data schema sync
- Monthly: Contract review

---

## Climate Science Team Interfaces

### → AI/Agent Team

(See AI/Agent → Climate Science above)

---

### → ML Platform Team

(See ML Platform → Climate Science above)

---

### → Data Engineering Team

**Interface:** Emission Factor Data

**Description:** Climate Science provides emission factor database to Data Engineering for enrichment pipelines.

**Contract:**
```yaml
# Emission factor API

GET /v1/emission-factors?cn_code=72081000&country=CN&production_route=bf_bof

Response:
{
  "emission_factor_tco2_per_tonne": 2.1,
  "source": "IEA Steel Technology Roadmap 2024",
  "reference_url": "https://iea.org/reports/steel-2024",
  "effective_date": "2024-01-01",
  "confidence": "high"
}
```

**SLA:**
- API availability: 99.9%
- API latency (p95): <100ms
- Data update frequency: Quarterly

**Handoff Points:**
- **Input:** Product (CN code), country, production route
- **Output:** Emission factor with provenance
- **Error Handling:** Return default value if not found; log warning

**Meeting Cadence:**
- Monthly: Data quality review
- Quarterly: Emission factor updates

---

## Platform Team Interfaces

### → AI/Agent Team

(See AI/Agent → Platform above)

---

### → Data Engineering Team

**Interface:** Agent Registry Database

**Description:** Platform provides PostgreSQL database for Data Engineering to store pipeline metadata.

**Contract:**
```sql
-- Shared database schema

-- Agents table (Platform owns)
CREATE TABLE agents (...);

-- Agent usage metrics (Data Engineering writes)
CREATE TABLE agent_usage (
    agent_version_id UUID REFERENCES agent_versions(id),
    execution_count BIGINT,
    ...
);
```

**SLA:**
- Database availability: 99.95%
- Query latency (p95): <50ms

**Handoff Points:**
- **Input:** Agent usage data
- **Output:** Stored in agent_usage table
- **Error Handling:** Retry on transient DB errors

**Meeting Cadence:**
- Weekly: Database performance review
- Monthly: Schema evolution planning

---

### → DevOps Team

**Interface:** Application Deployment Specs

**Description:** Platform provides Helm charts and Kubernetes manifests to DevOps for deployment.

**Contract:**
```yaml
# Deployment handoff checklist

Platform Team provides:
- [ ] Helm chart in helm/ directory
- [ ] values.yaml for each environment (dev, staging, prod)
- [ ] README with deployment instructions
- [ ] Health check endpoints (/health, /ready)
- [ ] Resource requirements (CPU, memory)
- [ ] Environment variables documented

DevOps Team validates:
- [ ] Helm chart lints successfully
- [ ] Dry-run deployment succeeds
- [ ] Health checks pass
- [ ] Monitoring dashboards configured
```

**SLA:**
- Deployment review time: <2 business days
- Deployment success rate: >95%

**Handoff Points:**
- **Input:** Helm chart, Docker image
- **Output:** Application deployed to Kubernetes
- **Error Handling:** Automated rollback on health check failure

**Meeting Cadence:**
- Weekly: Deployment planning
- Bi-Weekly: Post-deployment review

---

## Data Engineering Team Interfaces

### → AI/Agent Team

(See AI/Agent → Data Engineering above)

---

### → Climate Science Team

(See Climate Science → Data Engineering above)

---

### → Platform Team

(See Platform → Data Engineering above)

---

### → DevOps Team

**Interface:** Data Pipeline Deployment

**Description:** Data Engineering provides Airflow DAGs to DevOps for deployment to Airflow cluster.

**Contract:**
```python
# DAG handoff checklist

Data Engineering provides:
- [ ] DAG Python file in dags/ directory
- [ ] Dependencies in requirements.txt
- [ ] DAG documentation (docstrings)
- [ ] Test suite for DAG
- [ ] Environment variables documented

DevOps Team validates:
- [ ] DAG passes Airflow validation
- [ ] Dependencies installable
- [ ] Tests pass
- [ ] Monitoring configured (DAG run alerts)
```

**SLA:**
- DAG deployment time: <1 business day
- Pipeline uptime: 99.9%

**Handoff Points:**
- **Input:** Airflow DAG file
- **Output:** DAG deployed and scheduled
- **Error Handling:** Email/Slack alerts on DAG failures

**Meeting Cadence:**
- Weekly: Pipeline health review
- Monthly: Capacity planning

---

## DevOps Team Interfaces

### → All Teams

**Interface:** CI/CD Pipelines

**Description:** DevOps provides GitHub Actions workflows for all teams to build, test, and deploy code.

**Contract:**
```yaml
# CI/CD workflow (provided by DevOps)

# All teams get:
- Linting (Black, Flake8, Pylint)
- Unit testing (pytest)
- Security scanning (Bandit, Safety)
- Docker build and push
- Deployment to staging (on merge to develop)
- Deployment to prod (on merge to main)

# Team responsibilities:
- Write tests (>85% coverage)
- Fix linting errors
- Fix security vulnerabilities
- Provide Dockerfile and Helm chart
```

**SLA:**
- Pipeline execution time: <10 minutes
- Pipeline success rate: >90%
- Pipeline uptime: 99.9%

**Handoff Points:**
- **Input:** Code commit (GitHub)
- **Output:** Deployed application (Kubernetes)
- **Error Handling:** Slack notification on failure; auto-rollback on prod failures

**Meeting Cadence:**
- Weekly: Pipeline performance review
- Monthly: CI/CD improvements

---

**Interface:** Observability Stack

**Description:** DevOps provides Prometheus, Grafana, and ELK for all teams to monitor their services.

**Contract:**
```python
# Monitoring requirements (all services must provide)

# Prometheus metrics endpoint
GET /metrics
# Returns metrics in Prometheus format

# Grafana dashboards
# Each team creates dashboard in Grafana

# Structured logging (JSON format)
{
  "timestamp": "2025-12-03T10:00:00Z",
  "level": "INFO",
  "service": "agent-factory",
  "message": "Agent generated",
  "agent_id": "GL-CBAM-APP",
  "latency_ms": 45000
}
```

**SLA:**
- Metrics retention: 90 days
- Log retention: 30 days
- Dashboard availability: 99.9%

**Handoff Points:**
- **Input:** Service metrics and logs
- **Output:** Dashboards and alerts
- **Error Handling:** Alerts sent to team Slack channel

**Meeting Cadence:**
- Weekly: SRE review (all teams)
- Monthly: Observability improvements

---

## Communication Protocols

### Synchronous Communication

**Slack Channels:**
```
#agent-factory-all            - Program-wide announcements
#agent-factory-ml-platform    - ML Platform team
#agent-factory-ai-agents      - AI/Agent team
#agent-factory-climate-science - Climate Science team
#agent-factory-platform       - Platform team
#agent-factory-data           - Data Engineering team
#agent-factory-devops         - DevOps team
#agent-factory-tech-leads     - Tech leads only
#agent-factory-incidents      - Production incidents
```

**Slack Best Practices:**
- Tag individuals (@person) for urgent issues
- Tag teams (@ml-platform) for team-wide questions
- Use threads to keep channels organized
- Use emoji reactions for quick acknowledgments

**Meetings:**
- **Daily Standups:** Team-level (15 min)
- **Weekly Integration Sync:** All tech leads (60 min)
- **Bi-Weekly Sprint Planning:** All teams (90 min)
- **Monthly All-Hands:** Entire program (60 min)

---

### Asynchronous Communication

**GitHub:**
- **Pull Requests:** For code reviews (24-hour review SLA)
- **Issues:** For bug tracking and feature requests
- **Discussions:** For RFCs and architectural discussions

**Confluence:**
- **RFCs (Request for Comments):** Architectural proposals (3-day review period)
- **ADRs (Architecture Decision Records):** Approved architectural decisions
- **PRDs (Product Requirements Docs):** Product specifications

**Notion:**
- **Product Roadmap:** Quarterly and annual plans
- **OKRs:** Team objectives and key results
- **Meeting Notes:** All meeting notes archived

---

## Handoff Checklists

### Agent Generation → Certification (AI/Agent → Climate Science)

**Handoff Trigger:** Agent code generated and passing unit tests

**AI/Agent Team provides:**
- [ ] Agent source code (GitHub repo)
- [ ] AgentSpec file
- [ ] Test results (unit tests, integration tests)
- [ ] Quality metrics (code coverage, linting score)
- [ ] Documentation (README, API docs)

**Climate Science Team validates:**
- [ ] Runs golden test suite (must pass >95%)
- [ ] Reviews validation hook integration
- [ ] Checks regulatory compliance
- [ ] Certifies agent or requests changes

**SLA:** Certification review within 2 business days

**Communication:** Slack `#agent-factory-ai-agents` + `#agent-factory-climate-science`

---

### Agent Certification → Registry (Climate Science → Platform)

**Handoff Trigger:** Agent certified

**Climate Science Team provides:**
- [ ] Certification status (passed/failed)
- [ ] Certification ID (e.g., "CERT-CBAM-001")
- [ ] Certification certificate (PDF)
- [ ] Quality score (0-100)

**Platform Team validates:**
- [ ] Updates agent registry with certification status
- [ ] Publishes certificate to registry
- [ ] Notifies AI/Agent Team

**SLA:** Registry update within 1 business day

**Communication:** Slack `#agent-factory-platform`

---

### Agent Registry → Deployment (Platform → DevOps)

**Handoff Trigger:** Certified agent ready for production

**Platform Team provides:**
- [ ] Agent ID and version
- [ ] Docker image URL
- [ ] Helm chart
- [ ] Deployment instructions

**DevOps Team validates:**
- [ ] Dry-run deployment succeeds
- [ ] Health checks configured
- [ ] Monitoring dashboards created
- [ ] Deploy to staging first, then prod

**SLA:** Deployment to staging within 1 business day, prod within 3 business days

**Communication:** Slack `#agent-factory-devops`

---

## Integration Testing Strategy

### Cross-Team Integration Tests

**Location:** `tests/integration/` in shared repo

**Ownership:** Each interface has a designated owner team

**Execution:** Nightly automated run + on-demand for PRs

**Example:**
```python
# tests/integration/model_api/test_agent_generation.py

def test_model_api_generates_valid_code():
    """Test ML Platform Model API integration with Agent Factory."""

    # Call Model API (ML Platform)
    response = model_api.generate(
        prompt="Generate CBAM calculation function",
        model_id="claude-sonnet-4-5"
    )

    # Validate response
    assert response.status_code == 200
    assert "def calculate_emissions" in response.generated_text

    # Validate code quality (AI/Agent Team)
    quality_score = code_quality_checker.analyze(response.generated_text)
    assert quality_score > 90
```

**Failure Protocol:**
1. Test fails → Slack alert to owning teams
2. Teams debug within 4 hours
3. Root cause identified within 1 business day
4. Fix deployed within 2 business days

---

## Interface Versioning

### API Versioning Strategy

**Format:** Semantic versioning (v1.0.0, v2.0.0)

**Breaking Changes:**
- Require major version bump (v1 → v2)
- 30-day deprecation notice
- Backward compatibility maintained for 1 major version

**Non-Breaking Changes:**
- Minor version bump (v1.0 → v1.1)
- Immediate deployment
- No deprecation notice required

**Example:**
```
v1.0.0 (current)
  ↓
v1.1.0 (add optional field)  ← Non-breaking
  ↓
v2.0.0 (remove field)        ← Breaking (30-day notice)
  ↓
v2.1.0 (add new endpoint)    ← Non-breaking
```

---

## Conflict Resolution

### Escalation Path

**Level 1: Team-to-Team (Response: <4 hours)**
- Tech leads from both teams discuss
- Document decision in Slack thread
- Examples: API contract disagreements, schema conflicts

**Level 2: Tech Lead Council (Response: <1 day)**
- All tech leads vote (majority wins)
- Engineering Lead has tie-breaking vote
- Document decision in RFC
- Examples: Cross-team architecture changes

**Level 3: Engineering Lead (Response: <2 days)**
- Engineering Lead makes final decision
- Document in ADR (Architecture Decision Record)
- Examples: Major technology changes, system redesigns

**Level 4: Executive (Response: <3 days)**
- Product Manager + Engineering Lead decide
- Document in program docs
- Examples: Scope changes, timeline adjustments

---

## Appendix: Interface SLA Summary

| Interface | Provider | Consumer | Availability | Latency (p95) | Response Time |
|-----------|----------|----------|--------------|---------------|---------------|
| Model API | ML Platform | AI/Agent | 99.95% | <3 sec | Real-time |
| Evaluation Harness | ML Platform | AI/Agent | 99.9% | <5 min | Real-time |
| Validation Hooks | Climate Science | AI/Agent | 99.9% | <1 sec | Real-time |
| Agent Registry API | Platform | AI/Agent | 99.95% | <200ms | Real-time |
| Data Contracts | Data Eng | AI/Agent | N/A | N/A | Static |
| CI/CD Pipelines | DevOps | All Teams | 99.9% | <10 min | On-demand |
| Observability | DevOps | All Teams | 99.9% | <1 sec | Real-time |

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-03 | GL Product Manager | Initial team interfaces |

---

**Approvals:**

- Engineering Lead: ___________________
- All Tech Leads: ___________________
