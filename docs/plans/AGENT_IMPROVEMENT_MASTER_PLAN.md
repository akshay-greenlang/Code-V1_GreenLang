# GL-001 to GL-020 Agent Improvement Master Plan
## Target: Elevate All Agents to 95-97/100

---

## Executive Summary

| Current State | Target State | Gap |
|---------------|--------------|-----|
| Average: 89.21 | Target: 96.0 | +6.79 points |
| Range: 84.90 - 94.70 | Target: 95-97 | Variable |
| Grade A agents: 8 | Target: 20 | +12 agents |

---

## Gap Analysis by Agent (Priority Order)

### TIER 1: Critical Improvement Needed (Gap > 9 points)

| Agent | Current | Target | Gap | Priority |
|-------|---------|--------|-----|----------|
| GL-016 WATERGUARD | 84.90 | 96 | +11.10 | P0 |
| GL-017 CONDENSYNC | 85.80 | 96 | +10.20 | P0 |
| GL-018 FLUEFLOW | 86.50 | 96 | +9.50 | P0 |

### TIER 2: Major Improvement Needed (Gap 7-9 points)

| Agent | Current | Target | Gap | Priority |
|-------|---------|--------|-----|----------|
| GL-015 INSULSCAN | 87.00 | 96 | +9.00 | P1 |
| GL-014 EXCHANGER-PRO | 88.10 | 96 | +7.90 | P1 |
| GL-013 PREDICTMAINT | 89.20 | 96 | +6.80 | P1 |
| GL-012 STEAMQUAL | 87.40 | 96 | +8.60 | P1 |
| GL-007 FURNACEPULSE | 87.80 | 96 | +8.20 | P1 |

### TIER 3: Moderate Improvement Needed (Gap 5-7 points)

| Agent | Current | Target | Gap | Priority |
|-------|---------|--------|-----|----------|
| GL-002 FLAMEGUARD | 88.30 | 96 | +7.70 | P2 |
| GL-008 TRAPCATCHER | 88.90 | 96 | +7.10 | P2 |
| GL-019 HEATSCHEDULER | 88.60 | 96 | +7.40 | P2 |
| GL-004 BURNMASTER | 89.55 | 96 | +6.45 | P2 |
| GL-011 FUELCRAFT | 90.10 | 96 | +5.90 | P2 |
| GL-020 ECONOPULSE | 91.00 | 96 | +5.00 | P2 |

### TIER 4: Minor Improvement Needed (Gap 3-5 points)

| Agent | Current | Target | Gap | Priority |
|-------|---------|--------|-----|----------|
| GL-003 STEAMWISE | 90.30 | 96 | +5.70 | P3 |
| GL-006 HEATRECLAIM | 91.40 | 96 | +4.60 | P3 |
| GL-009 THERMALIQ | 91.80 | 96 | +4.20 | P3 |

### TIER 5: Fine-Tuning Needed (Gap < 3 points)

| Agent | Current | Target | Gap | Priority |
|-------|---------|--------|-----|----------|
| GL-005 COMBUSENSE | 92.65 | 96 | +3.35 | P4 |
| GL-010 EMISSIONWATCH | 93.20 | 96 | +2.80 | P4 |
| GL-001 THERMOSYNC | 94.70 | 96 | +1.30 | P4 |

---

## Improvement Actions by Criterion

### 1. Engineering Quality (25% weight) - Target: 97/100

**Common Issues to Fix:**
- [ ] Remove all `sys.path.append()` patterns - use proper package installation
- [ ] Replace `Dict[str, Any]` returns with typed dataclasses
- [ ] Fix `@lru_cache` with unhashable Dict parameters
- [ ] Add missing `__all__` exports in `__init__.py` files
- [ ] Implement consistent error handling patterns
- [ ] Add type stubs for external dependencies

**Agent-Specific Actions:**
- GL-016/017/018: Add 15+ calculator modules each
- GL-007/012/015: Refactor large methods (>50 lines) into smaller units
- All agents: Ensure consistent code formatting with Black/Ruff

### 2. AI Agent Specs Compliance (20% weight) - Target: 99/100

**Common Issues to Fix:**
- [ ] Remove ALL uses of `random.uniform()` without deterministic seeding
- [ ] Add runtime assertions for temperature=0.0, seed=42
- [ ] Implement `DeterministicClock` consistently
- [ ] Add provenance hash verification on every calculation
- [ ] Create `ProvenanceAuditLog` class for immutable audit trails

**Agent-Specific Actions:**
- GL-003: Fix random usage in scada_connector.py
- GL-002: Add deterministic seeding to all simulated values
- All agents: Add `@deterministic` decorator to all calculation methods

### 3. Intelligence/Complexity (15% weight) - Target: 95/100

**Enhancements Needed:**
- [ ] Add advanced optimization algorithms (genetic, simulated annealing)
- [ ] Implement machine learning integration (with deterministic inference)
- [ ] Add multi-objective optimization with Pareto frontiers
- [ ] Implement uncertainty quantification (Monte Carlo with fixed seeds)

**Agent-Specific Actions:**
- GL-016: Add full water chemistry equilibrium calculations
- GL-017: Implement condenser fouling prediction models
- GL-018: Add complete flue gas speciation (JANAF tables)
- GL-006: Add advanced pinch analysis with multiple utilities
- GL-013: Enhance RUL prediction with Weibull distribution

### 4. Documentation Quality (10% weight) - Target: 98/100

**Enhancements Needed:**
- [ ] Add architecture diagrams (Mermaid/PlantUML)
- [ ] Create formula derivation appendices
- [ ] Add API reference with OpenAPI annotations
- [ ] Create operator runbooks for each agent
- [ ] Add troubleshooting decision trees

**Agent-Specific Actions:**
- GL-016/017/018: Create comprehensive README.md (500+ lines)
- All agents: Add inline formula comments with literature references
- All agents: Create CHANGELOG.md with semantic versioning

### 5. Test Coverage (10% weight) - Target: 95%+

**Enhancements Needed:**
- [ ] Add property-based testing with Hypothesis
- [ ] Create mutation testing suite
- [ ] Add chaos engineering tests
- [ ] Implement contract testing for integrations
- [ ] Add load/stress testing with Locust

**Agent-Specific Actions:**
- GL-016: Add 50+ unit tests (currently minimal)
- GL-017: Add 50+ unit tests
- GL-018: Add 50+ unit tests
- All agents: Add golden test fixtures for reproducibility

### 6. Integration Capabilities (10% weight) - Target: 96/100

**Enhancements Needed:**
- [ ] Add OSIsoft PI connector (historian)
- [ ] Add AspenTech IP.21 connector
- [ ] Implement SAP/Oracle ERP integration
- [ ] Add OPC UA security modes (Sign, SignAndEncrypt)
- [ ] Implement Modbus RTU serial support

**Agent-Specific Actions:**
- GL-016/017/018: Add 5+ integration connectors each
- All agents: Add circuit breaker patterns to all connectors
- All agents: Implement connection pooling

### 7. Security & Compliance (5% weight) - Target: 96/100

**Enhancements Needed:**
- [ ] Remove all hardcoded test credentials
- [ ] Add RBAC with granular permissions
- [ ] Implement OAuth 2.0 with PKCE
- [ ] Add rate limiting to all API endpoints
- [ ] Create security audit reports

**Agent-Specific Actions:**
- All agents: Add `SECURITY_AUDIT_REPORT.md`
- All agents: Implement input sanitization
- All agents: Add API key rotation mechanism

### 8. Observability (5% weight) - Target: 98/100

**Enhancements Needed:**
- [ ] Add 100+ Prometheus metrics per agent
- [ ] Implement distributed tracing (Jaeger/Zipkin)
- [ ] Add log aggregation (ELK/Loki)
- [ ] Create Grafana dashboards (JSON exports)
- [ ] Add PagerDuty/OpsGenie alerting integration

**Agent-Specific Actions:**
- GL-016/017/018: Add comprehensive metrics.py (500+ lines)
- All agents: Add `prometheus_rules.yaml` for alerting
- All agents: Create 5+ Grafana dashboard JSON files

---

## Implementation Phases

### Phase 1: Critical Engineering Fixes (All Agents)
- Duration: Parallel execution
- Teams: Backend Developer, CodeSentinel
- Output: Fixed import patterns, type annotations, error handling

### Phase 2: Test Coverage Enhancement
- Duration: Parallel execution
- Teams: Test Engineer (×4)
- Focus: GL-016, GL-017, GL-018, GL-015
- Output: 95%+ coverage, property-based tests

### Phase 3: Calculator & Integration Enhancement
- Duration: Parallel execution
- Teams: Calculator Engineer, Data Integration Engineer
- Focus: Missing calculators for low-scoring agents
- Output: Advanced algorithms, new connectors

### Phase 4: Documentation Upgrade
- Duration: Parallel execution
- Teams: Tech Writer (×2)
- Output: Gold-standard documentation, runbooks

### Phase 5: Observability Implementation
- Duration: Parallel execution
- Teams: DevOps Engineer
- Output: Metrics, dashboards, alerting

### Phase 6: Security Hardening
- Duration: Parallel execution
- Teams: SecScan Agent
- Output: Security audit, hardening

### Phase 7: Final Verification
- Duration: Sequential
- Teams: CodeSentinel, Task Checker
- Output: Re-scored agents at 95-97

---

## Success Criteria

| Criterion | Current Avg | Target | Improvement |
|-----------|-------------|--------|-------------|
| Engineering Quality | 90.5 | 97 | +6.5 |
| AI Specs Compliance | 94.5 | 99 | +4.5 |
| Intelligence | 90.0 | 95 | +5.0 |
| Documentation | 92.0 | 98 | +6.0 |
| Test Coverage | 88.5 | 95 | +6.5 |
| Integration | 92.0 | 96 | +4.0 |
| Security | 89.0 | 96 | +7.0 |
| Observability | 93.0 | 98 | +5.0 |

**Final Target: All 20 agents scoring 95-97/100**

---

## Resource Allocation

| Team | Agents Assigned | Focus Area |
|------|-----------------|------------|
| Backend Developer Team 1 | GL-016, GL-017, GL-018 | Core calculator modules |
| Backend Developer Team 2 | GL-015, GL-012, GL-007 | Engineering quality |
| Test Engineer Team 1 | GL-016, GL-017, GL-018 | Test coverage |
| Test Engineer Team 2 | GL-014, GL-013, GL-002 | Test coverage |
| Calculator Engineer | All low-complexity agents | Algorithm enhancement |
| Data Integration Engineer | GL-016, GL-017, GL-018 | Integration connectors |
| Tech Writer Team | All agents | Documentation |
| DevOps Engineer | All agents | Observability |
| Security Auditor | All agents | Security hardening |

---

## Timeline Estimate

- Phase 1: Parallel (Engineering fixes)
- Phase 2: Parallel (Test coverage)
- Phase 3: Parallel (Calculators/Integrations)
- Phase 4: Parallel (Documentation)
- Phase 5: Parallel (Observability)
- Phase 6: Parallel (Security)
- Phase 7: Sequential (Verification)

**Total: 7 parallel phases with continuous improvement**
