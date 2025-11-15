# Testing Strategy for Agent Factory 5.0

## Executive Summary

This comprehensive testing strategy ensures **99.99% uptime** and **production-ready quality** for the GreenLang Agent Factory platform. We target **85%+ unit test coverage**, **100% critical path integration testing**, and **zero critical bugs in production**.

### Effort Estimates
- **Phase 1 (Production Readiness):** 3-4 weeks
- **Phase 2 (Intelligence Testing):** 2-3 weeks
- **Phase 3 (Excellence Testing):** 2 weeks
- **Phase 4 (Operations Testing):** 2 weeks
- **Total:** 9-11 weeks for comprehensive test suite implementation

### Quality Targets
| Metric | Target | Measurement |
|--------|--------|-------------|
| Unit Test Coverage | ≥85% | pytest-cov |
| Integration Test Coverage | 100% critical paths | Manual tracking |
| E2E Test Coverage | Top 20 user journeys | Playwright/Selenium |
| Performance (P95 Latency) | <500ms | Locust/K6 |
| Performance (P99 Latency) | <1000ms | Locust/K6 |
| Availability | 99.99% | Uptime monitoring |
| Security Vulnerabilities | 0 critical/high | Snyk, OWASP ZAP |
| Bug Escape Rate | <1% | Production monitoring |
| Concurrent Agents | 10,000+ | Load testing |
| Throughput | 1,000+ agents/sec | Benchmarking |

---

## Test Pyramid

```
                    /\
                   /  \
                  / E2E \          10% - Top 20 user journeys (~100 tests)
                 /______\          - Full workflow testing
                /        \         - UI + API + Database
               / Integration\      30% - Critical paths + APIs (~500 tests)
              /____________\       - Agent pipelines
             /              \      - Database operations
            /   Unit Tests   \     60% - Core business logic (~3,000 tests)
           /__________________\    - Agent methods
                                   - Utilities & helpers
                                   - Fast feedback (<5ms each)
```

**Test Distribution:**
- **Unit Tests (60%):** ~3,000 tests, <5ms each, run on every commit
- **Integration Tests (30%):** ~500 tests, <100ms each, run on PR merge
- **E2E Tests (10%):** ~100 tests, <5s each, run pre-deployment
- **Performance Tests:** Run nightly + pre-release
- **Security Tests:** Run nightly + on dependency updates
- **Chaos Tests:** Run monthly in staging

---

## Phase 1: Production Readiness Testing

### 1.1 Unit Tests (Target: 85%+ Coverage)

#### Test Structure

```
tests/
├── unit/
│   ├── core/
│   │   ├── test_llm_integration.py          [CREATED] ✓
│   │   ├── test_agent_registry.py
│   │   ├── test_dependency_injection.py
│   │   ├── test_configuration.py
│   │   └── test_observability.py
│   ├── agents/
│   │   ├── test_base_agent.py
│   │   ├── test_agent_factory.py
│   │   ├── test_agent_lifecycle.py
│   │   └── test_agent_validation.py
│   ├── api/
│   │   ├── test_auth.py
│   │   ├── test_endpoints.py
│   │   ├── test_middleware.py
│   │   └── test_rate_limiting.py
│   └── utils/
│       ├── test_validators.py
│       ├── test_formatters.py
│       └── test_helpers.py
```

#### Key Test Files

**`tests/unit/core/test_llm_integration.py`** [CREATED]
- ✓ Real Anthropic API integration tests
- ✓ Failover to OpenAI tests
- ✓ Rate limiting enforcement
- ✓ Retry logic with exponential backoff
- ✓ Circuit breaker pattern
- ✓ Token usage tracking
- ✓ Authentication failures
- ✓ Timeout handling
- ✓ Concurrent request handling
- Coverage: 90%+

**Test Checklist:**

**Core System Tests:**
- [ ] LLM Integration (Anthropic + OpenAI)
  - [x] API connection and authentication
  - [x] Request/response handling
  - [x] Failover logic
  - [x] Rate limiting
  - [x] Retry mechanisms
  - [x] Circuit breaker
  - [x] Token tracking
  - [x] Error handling
  - [x] Timeout handling
  - [x] Concurrent requests

- [ ] Agent Registry
  - [ ] Agent registration
  - [ ] Agent discovery
  - [ ] Version management
  - [ ] Dependency resolution
  - [ ] Circular dependency detection

- [ ] Dependency Injection
  - [ ] Service registration
  - [ ] Dependency resolution
  - [ ] Singleton lifecycle
  - [ ] Transient lifecycle
  - [ ] Scoped lifecycle

- [ ] Configuration Management
  - [ ] Environment-specific configs
  - [ ] Secret management
  - [ ] Config validation
  - [ ] Hot reload

- [ ] Observability
  - [ ] Metrics collection
  - [ ] Distributed tracing
  - [ ] Logging
  - [ ] Health checks

**Agent Tests:**
- [ ] Base Agent
  - [ ] Initialization
  - [ ] Process method
  - [ ] Error handling
  - [ ] Validation
  - [ ] Provenance tracking

- [ ] Agent Factory
  - [ ] Agent creation
  - [ ] Agent caching
  - [ ] Agent lifecycle
  - [ ] Resource cleanup

- [ ] Agent Validation
  - [ ] Input validation
  - [ ] Output validation
  - [ ] Schema enforcement
  - [ ] Type checking

**API Tests:**
- [ ] Authentication
  - [ ] Login/logout
  - [ ] JWT token generation
  - [ ] Token validation
  - [ ] Token refresh
  - [ ] Session management

- [ ] Authorization
  - [ ] RBAC enforcement
  - [ ] Permission checks
  - [ ] Resource ownership
  - [ ] Tenant isolation

- [ ] Endpoints
  - [ ] GET /agents
  - [ ] POST /agents/execute
  - [ ] GET /agents/{id}/status
  - [ ] POST /agents/create
  - [ ] DELETE /agents/{id}

- [ ] Middleware
  - [ ] Request validation
  - [ ] Response formatting
  - [ ] Error handling
  - [ ] CORS
  - [ ] Compression

- [ ] Rate Limiting
  - [ ] Per-user limits
  - [ ] Per-IP limits
  - [ ] Sliding window
  - [ ] Token bucket

**Utilities Tests:**
- [ ] Validators
  - [ ] Email validation
  - [ ] Phone validation
  - [ ] Date validation
  - [ ] Custom validators

- [ ] Formatters
  - [ ] JSON formatting
  - [ ] CSV formatting
  - [ ] Date formatting
  - [ ] Number formatting

- [ ] Helpers
  - [ ] String utilities
  - [ ] Math utilities
  - [ ] Collection utilities

### 1.2 Integration Tests (Target: 100% Critical Paths)

**`tests/integration/test_agent_pipeline.py`** [CREATED]
- ✓ Multi-agent orchestration
- ✓ Data flow between agents
- ✓ Pipeline error handling
- ✓ Agent dependency resolution
- ✓ Parallel execution
- ✓ Timeout handling
- ✓ Retry logic
- ✓ Database integration
- ✓ End-to-end scenarios (CBAM, Scope 3)

**Integration Test Checklist:**

- [x] Agent Pipeline Integration
  - [x] Single agent execution
  - [x] Multi-agent orchestration
  - [x] Dependency graph resolution
  - [x] Parallel execution
  - [x] Sequential execution
  - [x] Data flow between agents
  - [x] Error handling and retries
  - [x] Timeout management
  - [x] Database persistence

- [ ] Database Integration
  - [ ] PostgreSQL connection
  - [ ] TimescaleDB hypertables
  - [ ] CRUD operations
  - [ ] Transactions
  - [ ] Connection pooling
  - [ ] Query optimization

- [ ] Cache Integration
  - [ ] Redis connection
  - [ ] Cache hit/miss
  - [ ] Cache invalidation
  - [ ] TTL management
  - [ ] Distributed caching

- [ ] Message Queue Integration
  - [ ] RabbitMQ connection
  - [ ] Message publishing
  - [ ] Message consumption
  - [ ] Dead letter queue
  - [ ] Message retry

- [ ] External API Integration
  - [ ] ERP connectors (SAP, Oracle)
  - [ ] Shipping providers (FedEx, UPS)
  - [ ] Carbon databases
  - [ ] Third-party APIs

- [ ] File Storage Integration
  - [ ] S3 upload/download
  - [ ] File versioning
  - [ ] Pre-signed URLs
  - [ ] Large file handling

### 1.3 End-to-End Tests (Target: Top 20 User Journeys)

**Test Structure:**
```
tests/e2e/
├── user_journeys/
│   ├── test_cbam_compliance_flow.py
│   ├── test_scope3_calculation_flow.py
│   ├── test_agent_creation_flow.py
│   ├── test_report_generation_flow.py
│   └── test_data_import_flow.py
├── ui/
│   ├── test_dashboard.py
│   ├── test_agent_builder.py
│   └── test_reports.py
└── api/
    ├── test_rest_api.py
    └── test_graphql_api.py
```

**Top 20 User Journeys:**

1. **CBAM Compliance Calculation**
   - Import shipment data
   - Classify products
   - Calculate embedded emissions
   - Generate CBAM report
   - Submit to EU portal

2. **Scope 3 Emissions Calculation**
   - Configure categories
   - Import activity data
   - Calculate emissions
   - Generate disclosure report

3. **Agent Creation via UI**
   - Navigate to Agent Builder
   - Define agent logic
   - Test agent
   - Deploy agent
   - Monitor execution

4. **Agent Creation via API**
   - Authenticate
   - POST agent definition
   - Test agent
   - Monitor via API

5. **Data Import from ERP**
   - Configure ERP connector
   - Map fields
   - Run import
   - Validate data
   - Review errors

6. **Dashboard Analytics**
   - View key metrics
   - Filter by date range
   - Drill down into details
   - Export data

7. **Report Generation**
   - Select report type
   - Configure parameters
   - Generate report
   - Download PDF/Excel

8. **User Management**
   - Create user account
   - Assign roles
   - Set permissions
   - Deactivate user

9. **Multi-Tenant Onboarding**
   - Create tenant
   - Configure settings
   - Import initial data
   - Invite users

10. **Audit Trail Review**
    - View audit logs
    - Filter by user/action
    - Export for compliance
    - Investigate incidents

11. **Performance Monitoring**
    - View system metrics
    - Analyze agent performance
    - Identify bottlenecks
    - Configure alerts

12. **Error Investigation**
    - View error logs
    - Trace execution
    - Reproduce issue
    - Verify fix

13. **Data Quality Review**
    - Run quality checks
    - Review validation results
    - Correct errors
    - Re-validate

14. **API Integration Setup**
    - Generate API key
    - Test connection
    - Configure webhooks
    - Monitor usage

15. **Backup and Restore**
    - Trigger backup
    - Verify backup
    - Restore from backup
    - Validate data integrity

16. **System Upgrade**
    - Deploy new version
    - Run migrations
    - Verify functionality
    - Rollback if needed

17. **Load Testing in Staging**
    - Configure load test
    - Execute test
    - Analyze results
    - Optimize performance

18. **Security Audit**
    - Run vulnerability scan
    - Review findings
    - Apply patches
    - Re-scan

19. **Disaster Recovery Drill**
    - Simulate failure
    - Execute DR plan
    - Verify recovery
    - Document lessons

20. **Compliance Audit**
    - Generate compliance report
    - Review controls
    - Provide evidence
    - Address findings

---

## Phase 2: Intelligence Testing (Agent Factory AI)

### 2.1 Natural Language Agent Generation

**Test Scenarios:**
- [ ] Simple agent from prompt
  - Input: "Create an agent that calculates diesel emissions"
  - Expected: Functional emission calculator agent

- [ ] Complex multi-step agent
  - Input: "Create a CBAM compliance agent that imports shipments, classifies products, and calculates emissions"
  - Expected: Multi-step pipeline agent

- [ ] Agent with validation rules
  - Input: "Create an agent that validates shipment data for completeness"
  - Expected: Validation agent with comprehensive checks

- [ ] Agent with external integrations
  - Input: "Create an agent that fetches carbon factors from DEFRA API"
  - Expected: Agent with API integration

**Quality Metrics:**
- Agent generation success rate: >95%
- Generated code quality score: >8/10
- Agent test coverage: >80%
- Agent performance: <100ms execution

### 2.2 Agent Optimization

**Test Scenarios:**
- [ ] Performance optimization
  - Slow agent → AI suggests caching
  - Expected: 50%+ performance improvement

- [ ] Code quality improvement
  - Complex agent → AI refactors
  - Expected: Reduced cyclomatic complexity

- [ ] Test coverage improvement
  - Under-tested agent → AI generates tests
  - Expected: 85%+ coverage

- [ ] Security hardening
  - Vulnerable agent → AI adds validation
  - Expected: Zero vulnerabilities

### 2.3 Agent Evolution

**Test Scenarios:**
- [ ] Requirement changes
  - Input: "Update emission calculator to support new fuel type"
  - Expected: Agent modified correctly

- [ ] Bug fixes
  - Input: "Fix validation error in shipment agent"
  - Expected: Bug fixed, tests updated

- [ ] Feature additions
  - Input: "Add PDF export to report generator"
  - Expected: New feature added seamlessly

---

## Phase 3: Excellence Testing (Developer Experience)

### 3.1 Local Development Environment

**Test Scenarios:**
- [ ] First-time setup (<5 minutes)
  - Clone repo
  - Run setup script
  - Verify all services running
  - Run test suite

- [ ] Hot reload
  - Modify agent code
  - Verify auto-reload
  - Test changes immediately

- [ ] Debug experience
  - Set breakpoint
  - Step through code
  - Inspect variables
  - Modify and continue

- [ ] Test execution
  - Run single test
  - Run test suite
  - View coverage report
  - Debug failing test

### 3.2 Documentation Quality

**Test Scenarios:**
- [ ] API documentation accuracy
  - Test all documented endpoints
  - Verify examples work
  - Check parameter descriptions

- [ ] Tutorial completeness
  - Follow getting started guide
  - Complete all tutorials
  - Verify all code samples work

- [ ] Reference documentation
  - Search for feature
  - Find relevant docs
  - Understand usage
  - Implement successfully

### 3.3 Error Messages

**Test Scenarios:**
- [ ] Validation errors
  - Trigger validation error
  - Verify clear message
  - Verify suggested fix

- [ ] Runtime errors
  - Trigger runtime error
  - Verify stack trace
  - Verify error context

- [ ] Configuration errors
  - Provide invalid config
  - Verify helpful message
  - Verify how to fix

---

## Phase 4: Operations Testing

### 4.1 Multi-Region Testing

**Test Scenarios:**
- [ ] US-East-1 deployment
  - Deploy application
  - Run health checks
  - Verify functionality
  - Measure latency

- [ ] EU-West-1 deployment
  - Deploy application
  - Verify GDPR compliance
  - Test data residency
  - Measure latency

- [ ] Asia-Pacific deployment
  - Deploy application
  - Verify localization
  - Test connectivity
  - Measure latency

- [ ] Cross-region failover
  - Simulate region failure
  - Verify automatic failover
  - Verify data consistency
  - Measure recovery time

**Latency Targets by Region:**
| User Location | Primary Region | Target Latency (P95) |
|---------------|----------------|----------------------|
| US East | us-east-1 | <100ms |
| US West | us-west-2 | <150ms |
| Europe | eu-west-1 | <100ms |
| Asia | ap-southeast-1 | <150ms |

### 4.2 Disaster Recovery Testing

**Test Scenarios:**

**Database Failover:**
- [ ] Primary database failure
  - Simulate database crash
  - Verify automatic failover to replica
  - Verify zero data loss
  - Measure RTO (Recovery Time Objective): <5 minutes
  - Measure RPO (Recovery Point Objective): <1 minute

**Region Failover:**
- [ ] Complete region outage
  - Simulate AWS region failure
  - Verify DNS failover to backup region
  - Verify application availability
  - Verify data replication
  - Measure RTO: <15 minutes

**Backup and Restore:**
- [ ] Database backup
  - Trigger automated backup
  - Verify backup integrity
  - Restore to test environment
  - Verify data completeness
  - Measure restore time

- [ ] Point-in-time recovery
  - Identify recovery point
  - Execute PITR
  - Verify data state
  - Measure recovery time

**Data Loss Prevention:**
- [ ] Accidental deletion
  - Simulate user error
  - Recover from backup
  - Verify no data loss

- [ ] Corruption detection
  - Inject corrupted data
  - Detect via checksums
  - Recover from backup
  - Verify integrity

### 4.3 Chaos Engineering

**Monthly Chaos Experiments:**

**Experiment 1: Random Pod Termination**
```yaml
kind: ChaosExperiment
name: pod-delete
spec:
  target: greenlang-api
  action: delete-random-pod
  frequency: every 10 minutes
  duration: 1 hour
  expected_result: zero downtime
```

**Experiment 2: Network Latency**
```yaml
kind: ChaosExperiment
name: network-latency
spec:
  target: greenlang-db
  action: inject-latency
  latency: 500ms
  duration: 30 minutes
  expected_result: graceful degradation
```

**Experiment 3: CPU Throttling**
```yaml
kind: ChaosExperiment
name: cpu-stress
spec:
  target: greenlang-worker
  action: cpu-stress
  cpu_percent: 80
  duration: 1 hour
  expected_result: auto-scaling triggered
```

**Experiment 4: Memory Pressure**
```yaml
kind: ChaosExperiment
name: memory-stress
spec:
  target: greenlang-agent-executor
  action: memory-stress
  memory_percent: 90
  duration: 30 minutes
  expected_result: OOM kill + restart
```

**Experiment 5: Database Connection Loss**
```yaml
kind: ChaosExperiment
name: db-disconnect
spec:
  target: greenlang-api
  action: block-db-connections
  duration: 5 minutes
  expected_result: connection pool recovery
```

**Chaos Experiment Checklist:**
- [ ] Define blast radius (% of traffic affected)
- [ ] Set up monitoring and alerts
- [ ] Define rollback criteria
- [ ] Execute experiment
- [ ] Monitor system behavior
- [ ] Validate expected results
- [ ] Document findings
- [ ] Implement improvements

---

## Performance Testing

### 4.1 Load Testing

**`tests/performance/test_load_stress.py`** [CREATED]

**Load Test Scenarios:**

**Scenario 1: Baseline Load**
- Concurrent Users: 100
- Duration: 10 minutes
- Target: <200ms P95 latency

**Scenario 2: Peak Load**
- Concurrent Users: 1,000
- Duration: 30 minutes
- Target: <500ms P95 latency

**Scenario 3: High Load**
- Concurrent Users: 5,000
- Duration: 1 hour
- Target: <1000ms P95 latency

**Scenario 4: Maximum Load**
- Concurrent Users: 10,000
- Duration: 1 hour
- Target: System remains stable

**Load Test Metrics:**
| Metric | Target | Measurement |
|--------|--------|-------------|
| Throughput | 1,000+ req/s | Locust |
| P50 Latency | <100ms | Locust |
| P95 Latency | <500ms | Locust |
| P99 Latency | <1000ms | Locust |
| Error Rate | <1% | Locust |
| CPU Usage | <70% | Prometheus |
| Memory Usage | <80% | Prometheus |
| Database Connections | <80% of pool | pgBouncer |

### 4.2 Stress Testing

**Objective:** Find the breaking point

**Method:**
1. Start with 100 concurrent users
2. Increase by 100 every 2 minutes
3. Continue until error rate >5%
4. Record breaking point
5. Analyze bottlenecks

**Expected Results:**
- Breaking point: >10,000 concurrent users
- Graceful degradation (not crash)
- Clear error messages
- System recovers automatically

### 4.3 Endurance Testing

**24-Hour Endurance Test:**
- Sustained load: 500 concurrent users
- Duration: 24 hours
- Target metrics:
  - Uptime: 100%
  - Success rate: >99.9%
  - No memory leaks
  - No resource exhaustion
  - No performance degradation

**Memory Leak Detection:**
- Sample memory every 5 minutes
- Calculate linear regression
- Acceptable growth: <1MB per hour
- Alert if growth >10MB per hour

### 4.4 Spike Testing

**Sudden Traffic Surge:**
- Baseline: 100 users
- Spike to: 2,000 users in 30 seconds
- Duration: 5 minutes
- Return to baseline
- Target: >95% success rate during spike

---

## Security Testing

### 5.1 SAST (Static Application Security Testing)

**Tools:**
- **Bandit** (Python security linter)
- **Semgrep** (Static analysis)
- **SonarQube** (Code quality + security)

**Configuration:**

**`.banditrc`**
```ini
[bandit]
exclude = /tests/,/venv/
tests = B201,B301,B302,B303,B304,B305,B306,B307,B308,B309,B310,B311,B312,B313,B314,B315,B316,B317,B318,B319,B320,B321,B322,B323,B324,B325,B326,B327,B401,B402,B403,B404,B405,B406,B407,B408,B409,B410,B411,B412,B413,B501,B502,B503,B504,B505,B506,B507,B601,B602,B603,B604,B605,B606,B607,B608,B609,B610,B611,B701,B702,B703
```

**Run Bandit:**
```bash
bandit -r greenlang_core/ -f json -o bandit_report.json
```

**Semgrep Rules:**
```yaml
rules:
  - id: hardcoded-secret
    pattern: |
      password = "..."
    message: Hardcoded secret detected
    severity: ERROR

  - id: sql-injection
    pattern: |
      execute($SQL)
    message: Potential SQL injection
    severity: ERROR
```

**Run Semgrep:**
```bash
semgrep --config=auto greenlang_core/
```

### 5.2 DAST (Dynamic Application Security Testing)

**Tools:**
- **OWASP ZAP** (Automated security scanner)
- **Burp Suite** (Manual penetration testing)

**OWASP ZAP Configuration:**

**`zap-baseline-scan.yaml`**
```yaml
env:
  contexts:
    - name: greenlang-api
      urls:
        - https://api.greenlang.example.com
      includePaths:
        - "https://api.greenlang.example.com/api/v1/.*"
      authentication:
        type: bearer
        bearer_token: ${ZAP_AUTH_TOKEN}
  parameters:
    failOnError: true
    failOnWarning: false
    progressToStdout: true
  rules:
    - id: 10202 # Absence of Anti-CSRF tokens
      threshold: LOW
    - id: 10021 # X-Content-Type-Options header missing
      threshold: LOW
```

**Run ZAP Scan:**
```bash
docker run -v $(pwd):/zap/wrk/:rw \
  -t owasp/zap2docker-stable \
  zap-baseline.py \
  -t https://api.greenlang.example.com \
  -c zap-baseline-scan.yaml \
  -r zap-report.html
```

### 5.3 Dependency Scanning

**Tools:**
- **Snyk** (Vulnerability database)
- **Safety** (Python package vulnerabilities)
- **Dependabot** (Automated updates)

**Run Safety:**
```bash
safety check --full-report --json > safety_report.json
```

**Run Snyk:**
```bash
snyk test --json > snyk_report.json
snyk monitor # Monitor project continuously
```

### 5.4 Container Scanning

**Tools:**
- **Trivy** (Container vulnerability scanner)
- **Clair** (Container analysis)

**Run Trivy:**
```bash
trivy image greenlang/agent-factory:latest \
  --severity HIGH,CRITICAL \
  --format json \
  --output trivy_report.json
```

### 5.5 Secret Scanning

**Tools:**
- **Gitleaks** (Git secret scanner)
- **TruffleHog** (Git history scanner)

**Gitleaks Configuration:**

**`.gitleaks.toml`**
```toml
title = "GreenLang Secret Detection"

[[rules]]
id = "aws-access-key"
description = "AWS Access Key"
regex = '''(A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}'''

[[rules]]
id = "anthropic-api-key"
description = "Anthropic API Key"
regex = '''sk-ant-[a-zA-Z0-9-]{95}'''

[[rules]]
id = "openai-api-key"
description = "OpenAI API Key"
regex = '''sk-[a-zA-Z0-9]{48}'''

[[rules]]
id = "private-key"
description = "Private Key"
regex = '''-----BEGIN (RSA|OPENSSH|DSA|EC|PGP) PRIVATE KEY-----'''
```

**Run Gitleaks:**
```bash
gitleaks detect --source . --report-path gitleaks_report.json
```

### 5.6 Penetration Testing

**Quarterly External Penetration Test:**

**Scope:**
- Public APIs
- Web application
- Authentication/authorization
- Data access controls
- Multi-tenancy isolation

**Test Scenarios:**
1. SQL Injection
2. XSS (Reflected, Stored, DOM-based)
3. CSRF
4. Authentication bypass
5. Session hijacking
6. Privilege escalation
7. IDOR (Insecure Direct Object Reference)
8. API abuse
9. Rate limit bypass
10. Data exfiltration

**Deliverables:**
- Executive summary
- Technical findings
- Proof of concepts
- Remediation recommendations
- Re-test results

---

## Compliance Testing

### 6.1 GDPR Compliance Tests

**`tests/compliance/test_gdpr.py`**

**Test Scenarios:**

**Right to Access:**
- [ ] User can download all personal data
- [ ] Data export includes all systems
- [ ] Export completes within 30 days
- [ ] Format is machine-readable (JSON)

**Right to Erasure:**
- [ ] User can request account deletion
- [ ] All personal data is deleted
- [ ] Deletion completes within 30 days
- [ ] Backups are handled appropriately

**Right to Rectification:**
- [ ] User can update personal data
- [ ] Updates propagate to all systems
- [ ] Audit trail is maintained

**Right to Portability:**
- [ ] User can export data in standard format
- [ ] Data can be imported to another system
- [ ] Export includes all relevant data

**Consent Management:**
- [ ] User can grant/revoke consent
- [ ] Consent is recorded with timestamp
- [ ] Consent is version-controlled
- [ ] Processing stops when consent revoked

**Data Minimization:**
- [ ] Only necessary data is collected
- [ ] Data retention policies enforced
- [ ] Old data is automatically deleted

**Privacy by Design:**
- [ ] Default settings are privacy-friendly
- [ ] PII is encrypted at rest
- [ ] PII is encrypted in transit
- [ ] Access logs for PII are maintained

### 6.2 SOC 2 Control Testing

**Control Categories:**

**CC1: Control Environment**
- [ ] Code of conduct
- [ ] Security policies
- [ ] Employee training
- [ ] Background checks

**CC2: Communication and Information**
- [ ] Security awareness training
- [ ] Incident response procedures
- [ ] Change management process

**CC3: Risk Assessment**
- [ ] Annual risk assessment
- [ ] Threat modeling
- [ ] Vulnerability management

**CC4: Monitoring Activities**
- [ ] Continuous monitoring
- [ ] SIEM alerts
- [ ] Performance monitoring
- [ ] Availability monitoring

**CC5: Control Activities**
- [ ] Access controls
- [ ] Change controls
- [ ] Backup controls
- [ ] Encryption controls

**CC6: Logical and Physical Access**
- [ ] Multi-factor authentication
- [ ] Role-based access control
- [ ] Access reviews
- [ ] Physical security

**CC7: System Operations**
- [ ] Incident management
- [ ] Problem management
- [ ] Change management
- [ ] Capacity management

**CC8: Change Management**
- [ ] Development standards
- [ ] Code review
- [ ] Testing requirements
- [ ] Deployment procedures

**CC9: Risk Mitigation**
- [ ] Disaster recovery plan
- [ ] Business continuity plan
- [ ] Incident response plan
- [ ] Vendor management

### 6.3 Audit Log Validation

**Test Scenarios:**
- [ ] All authentication events logged
- [ ] All authorization decisions logged
- [ ] All data access logged
- [ ] All configuration changes logged
- [ ] All admin actions logged
- [ ] Logs are immutable
- [ ] Logs are encrypted
- [ ] Logs are retained for required period
- [ ] Logs can be searched and filtered
- [ ] Logs can be exported for audit

### 6.4 Data Residency Enforcement

**Test Scenarios:**
- [ ] EU data stays in EU
- [ ] US data stays in US
- [ ] Data location documented
- [ ] Cross-border transfers require approval
- [ ] Data location can be verified
- [ ] Compliance reports available

---

## Test Automation

### 7.1 CI/CD Pipeline Integration

**GitHub Actions Workflow:**

**`.github/workflows/test.yml`**
```yaml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run unit tests
        run: |
          pytest tests/unit/ \
            --cov=greenlang_core \
            --cov-report=xml \
            --cov-report=html \
            --cov-fail-under=85 \
            --junitxml=junit.xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: unit

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: timescale/timescaledb:latest-pg15
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run integration tests
        run: |
          pytest tests/integration/ \
            --junitxml=junit.xml
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/greenlang
          REDIS_URL: redis://localhost:6379

  security-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Bandit
        run: |
          pip install bandit
          bandit -r greenlang_core/ -f json -o bandit_report.json

      - name: Run Safety
        run: |
          pip install safety
          safety check --json > safety_report.json

      - name: Run Snyk
        uses: snyk/actions/python@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}

      - name: Run Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install playwright
          playwright install

      - name: Start application
        run: |
          docker-compose up -d
          ./scripts/wait-for-app.sh

      - name: Run E2E tests
        run: |
          pytest tests/e2e/ \
            --headed \
            --video=retain-on-failure

      - name: Upload test artifacts
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: e2e-artifacts
          path: test-results/

  performance-tests:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Run performance tests
        run: |
          pip install locust
          locust -f tests/performance/locustfile.py \
            --headless \
            --users 1000 \
            --spawn-rate 10 \
            --run-time 5m \
            --host https://staging.greenlang.example.com \
            --html performance_report.html

      - name: Upload performance report
        uses: actions/upload-artifact@v3
        with:
          name: performance-report
          path: performance_report.html
```

### 7.2 Pre-commit Hooks

**`.pre-commit-config.yaml`**
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=120']

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-ll', '-i']

  - repo: local
    hooks:
      - id: pytest-unit
        name: Run unit tests
        entry: pytest tests/unit/ --exitfirst
        language: system
        pass_filenames: false
        always_run: true
```

### 7.3 Test Execution Schedule

| Test Type | Trigger | Frequency | Duration |
|-----------|---------|-----------|----------|
| Unit Tests | Every commit | Continuous | ~2 min |
| Integration Tests | PR merge | On-demand | ~10 min |
| E2E Tests | Pre-deployment | Daily + On-demand | ~30 min |
| Performance Tests | Nightly | Daily | ~1 hour |
| Security Scans | Nightly | Daily | ~15 min |
| Load Tests | Weekly | Weekly | ~2 hours |
| Stress Tests | Monthly | Monthly | ~4 hours |
| Chaos Tests | Monthly | Monthly | ~8 hours |
| Penetration Tests | Quarterly | Quarterly | 1 week |

---

## Test Data Management

### 8.1 Test Data Generation

**Test Data Factory:**

**`tests/fixtures/data_factory.py`**
```python
"""
Test data factory for generating realistic test data.
"""

from faker import Faker
import random
from datetime import datetime, timedelta

faker = Faker()

class TestDataFactory:
    """Generate test data for GreenLang testing."""

    @staticmethod
    def generate_shipment(num_records=100):
        """Generate test shipment data."""
        shipments = []
        for _ in range(num_records):
            shipment = {
                'shipment_id': faker.uuid4(),
                'product_category': random.choice(['cement', 'steel', 'aluminum']),
                'weight_tonnes': round(random.uniform(0.1, 100.0), 2),
                'origin_country': faker.country_code(),
                'destination_country': 'US',
                'import_date': faker.date_between(start_date='-1y', end_date='today'),
                'supplier_name': faker.company(),
                'hs_code': f"{random.randint(2500, 2900)}.{random.randint(10, 99)}"
            }
            shipments.append(shipment)
        return shipments

    @staticmethod
    def generate_emission_factor(fuel_type, region):
        """Generate emission factor data."""
        # Based on real-world emission factors
        factors = {
            ('diesel', 'US'): 2.68,  # kg CO2e per liter
            ('gasoline', 'US'): 2.31,
            ('natural_gas', 'US'): 1.93,
            ('coal', 'US'): 3.45,
        }
        return factors.get((fuel_type, region), 2.0)
```

### 8.2 Test Database Management

**Database Setup Script:**

**`tests/scripts/setup_test_db.sh`**
```bash
#!/bin/bash

# Setup test database with sample data

set -e

echo "Creating test database..."
docker exec -it greenlang-db psql -U postgres -c "DROP DATABASE IF EXISTS greenlang_test;"
docker exec -it greenlang-db psql -U postgres -c "CREATE DATABASE greenlang_test;"

echo "Running migrations..."
alembic upgrade head --database-url postgresql://postgres:test@localhost:5432/greenlang_test

echo "Loading test data..."
python tests/scripts/load_test_data.py

echo "Test database ready!"
```

### 8.3 Test Data Cleanup

**Pytest Fixtures:**

**`tests/conftest.py`**
```python
"""
Pytest configuration and fixtures.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope='session')
def db_engine():
    """Create database engine for tests."""
    engine = create_engine('postgresql://postgres:test@localhost:5432/greenlang_test')
    yield engine
    engine.dispose()

@pytest.fixture(scope='function')
def db_session(db_engine):
    """Create database session for each test."""
    Session = sessionmaker(bind=db_engine)
    session = Session()

    yield session

    # Cleanup after test
    session.rollback()
    session.close()

@pytest.fixture(autouse=True)
def reset_database(db_session):
    """Reset database state between tests."""
    # Truncate all tables
    db_session.execute('TRUNCATE TABLE emissions CASCADE')
    db_session.execute('TRUNCATE TABLE shipments CASCADE')
    db_session.commit()
```

---

## Quality Gates

### 9.1 Code Quality Gates

**Pass Criteria:**
- [ ] Unit test coverage ≥85%
- [ ] Integration test coverage: 100% critical paths
- [ ] No critical/high security vulnerabilities
- [ ] Code quality score ≥8/10 (SonarQube)
- [ ] Zero high-severity bugs
- [ ] Cyclomatic complexity <10
- [ ] Duplication <3%

### 9.2 Performance Quality Gates

**Pass Criteria:**
- [ ] P95 latency <500ms
- [ ] P99 latency <1000ms
- [ ] Throughput >1000 req/s
- [ ] Error rate <1%
- [ ] CPU usage <70%
- [ ] Memory usage <80%
- [ ] No memory leaks detected

### 9.3 Security Quality Gates

**Pass Criteria:**
- [ ] Zero critical vulnerabilities
- [ ] Zero high vulnerabilities
- [ ] All secrets encrypted
- [ ] All endpoints authenticated
- [ ] All authorization checks in place
- [ ] All inputs validated
- [ ] All outputs encoded
- [ ] Audit logs complete

### 9.4 Deployment Quality Gates

**Pre-Production:**
- [ ] All tests pass
- [ ] Performance benchmarks met
- [ ] Security scan clean
- [ ] Documentation updated
- [ ] Release notes complete
- [ ] Rollback plan documented

**Production:**
- [ ] Canary deployment successful
- [ ] Smoke tests pass
- [ ] Health checks green
- [ ] Error rate normal
- [ ] Latency normal
- [ ] No alerts triggered

---

## Test Environment Setup

### 10.1 Local Development

**Docker Compose Setup:**

**`docker-compose.test.yml`**
```yaml
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_PASSWORD: test
      POSTGRES_DB: greenlang_test
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest

  localstack:
    image: localstack/localstack:latest
    environment:
      SERVICES: s3,sqs,sns
      AWS_DEFAULT_REGION: us-east-1
    ports:
      - "4566:4566"

volumes:
  postgres_data:
```

**Start Test Environment:**
```bash
docker-compose -f docker-compose.test.yml up -d
```

### 10.2 CI/CD Test Environment

**Kubernetes Test Namespace:**

**`k8s/test-namespace.yaml`**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: greenlang-test
  labels:
    environment: test
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: test-quota
  namespace: greenlang-test
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    pods: "50"
```

### 10.3 Staging Environment

**Infrastructure:**
- Kubernetes cluster (3 nodes, 4 CPU / 16GB RAM each)
- PostgreSQL (RDS or managed)
- Redis (ElastiCache or managed)
- Load balancer
- Monitoring stack (Prometheus + Grafana)

**Deployment:**
```bash
# Deploy to staging
kubectl apply -f k8s/staging/ -n greenlang-staging

# Run smoke tests
pytest tests/smoke/ --host=https://staging.greenlang.example.com

# Run load tests
locust -f tests/performance/locustfile.py \
  --host https://staging.greenlang.example.com \
  --users 1000 \
  --spawn-rate 10 \
  --run-time 10m
```

---

## Success Metrics

### Test Coverage Progress

| Component | Current | Target | Status |
|-----------|---------|--------|--------|
| Core System | 0% | 90% | 🔴 Not Started |
| Agents | 0% | 85% | 🔴 Not Started |
| API | 0% | 90% | 🔴 Not Started |
| Database | 0% | 85% | 🔴 Not Started |
| UI | 0% | 70% | 🔴 Not Started |
| **Overall** | **0%** | **85%** | 🔴 **Not Started** |

### Quality Metrics Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│  GreenLang Agent Factory - Quality Metrics Dashboard       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Test Coverage:              0% / 85% target     🔴         │
│  Unit Tests:                 0 / 3000 tests      🔴         │
│  Integration Tests:          0 / 500 tests       🔴         │
│  E2E Tests:                  0 / 100 tests       🔴         │
│                                                             │
│  Performance:                                               │
│    P95 Latency:              N/A / <500ms        ⚪         │
│    Throughput:               N/A / 1000 req/s    ⚪         │
│                                                             │
│  Security:                                                  │
│    Critical Vulns:           0 / 0 target        🟢         │
│    High Vulns:               0 / 0 target        🟢         │
│                                                             │
│  Availability:               N/A / 99.99%        ⚪         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Weekly Progress Report Template

**Week X Progress Report**

**Tests Implemented:**
- Unit tests: X added (Total: X/3000)
- Integration tests: X added (Total: X/500)
- E2E tests: X added (Total: X/100)

**Coverage:**
- Overall coverage: X% (Target: 85%)
- Core system: X%
- Agents: X%
- API: X%

**Quality Gates:**
- ✅ / ❌ All tests passing
- ✅ / ❌ Coverage ≥85%
- ✅ / ❌ No critical vulnerabilities
- ✅ / ❌ Performance targets met

**Blockers:**
- [List any blockers]

**Next Week Goals:**
- [List goals for next week]

---

## Appendix: Configuration Files

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Markers
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    security: Security tests
    slow: Slow-running tests
    compliance: Compliance tests
    locust: Locust load tests

# Coverage
addopts =
    --strict-markers
    --verbose
    --tb=short
    --cov-branch
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml:coverage.xml
    --maxfail=5
    --durations=10

# Timeout
timeout = 300
timeout_method = thread

# Parallel execution
# addopts = -n auto
```

### requirements-dev.txt

```txt
# Testing
pytest==7.4.0
pytest-asyncio==0.21.0
pytest-cov==4.1.0
pytest-mock==3.11.1
pytest-timeout==2.1.0
pytest-benchmark==4.0.0
pytest-xdist==3.3.1

# Performance Testing
locust==2.15.1

# Security Testing
bandit==1.7.5
safety==2.3.5
gitleaks==8.16.0

# Code Quality
black==23.3.0
flake8==6.0.0
mypy==1.4.0
pylint==2.17.4
isort==5.12.0

# Test Data
faker==18.13.0

# Mocking
responses==0.23.1
freezegun==1.2.2

# Browser Testing
playwright==1.35.0
selenium==4.10.0

# API Testing
httpx==0.24.1
requests-mock==1.11.0

# Monitoring in Tests
psutil==5.9.5

# Pre-commit
pre-commit==3.3.3
```

---

## Next Steps

1. **Week 1-2:** Implement Phase 1 unit tests (LLM integration, agent registry, core system)
2. **Week 3-4:** Implement integration tests (pipeline, database, external APIs)
3. **Week 5-6:** Implement E2E tests (top 20 user journeys)
4. **Week 7-8:** Implement performance tests (load, stress, endurance)
5. **Week 9-10:** Implement security tests (SAST, DAST, penetration)
6. **Week 11:** Chaos engineering experiments
7. **Week 12:** Compliance testing and documentation

**Target Date for 85% Coverage:** 12 weeks from start

**Success Criteria:**
- ✅ 85%+ unit test coverage
- ✅ 100% critical path integration tests
- ✅ Top 20 E2E user journeys covered
- ✅ Performance targets met (<500ms P95)
- ✅ Zero critical/high security vulnerabilities
- ✅ 99.99% uptime in production
- ✅ Zero critical bugs escaped to production

---

**Document Version:** 1.0
**Last Updated:** 2025-01-14
**Owner:** GL-TestEngineer
**Review Cycle:** Monthly
