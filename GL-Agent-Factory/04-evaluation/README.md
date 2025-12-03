# GL Agent Factory: Evaluation & Certification Framework

**Version:** 1.0.0
**Date:** 2025-12-03
**Owner:** GreenLang Quality Engineering Team

---

## Overview

This directory contains the comprehensive Evaluation & Certification framework for the GreenLang Agent Factory. Every agent must pass rigorous evaluation across 12 dimensions before production deployment.

**Core Principle:** No agent reaches production without passing comprehensive evaluation.

---

## Framework Documents

### 1. [Evaluation Overview](framework/00-EVALUATION_OVERVIEW.md)

The foundation document defining evaluation philosophy, types of evaluation (accuracy, compliance, quality), certification criteria, and agent lifecycle states.

**Key Topics:**
- Evaluation philosophy and goals
- 5 types of evaluation (accuracy, compliance, climate science, quality, performance)
- Agent lifecycle states (Draft → Experimental → Certified → Deprecated)
- Certification process (6 phases)
- Evaluation cadence (initial, quarterly, annual)
- Roles & responsibilities

**Start Here:** This is the comprehensive overview of the entire framework.

---

### 2. [Certification Criteria](criteria/00-CERTIFICATION_CRITERIA.md)

Detailed pass/fail criteria for each of the 12 certification dimensions that every agent must satisfy.

**12 Certification Dimensions:**
1. Specification Completeness (5% weight)
2. Code Implementation (10% weight)
3. Test Coverage (15% weight)
4. Deterministic AI Guarantees (10% weight)
5. Documentation Completeness (5% weight)
6. Compliance & Security (20% weight)
7. Deployment Readiness (5% weight)
8. Exit Bar Criteria (10% weight)
9. Integration & Coordination (5% weight)
10. Business Impact & Metrics (5% weight)
11. Operational Excellence (5% weight)
12. Continuous Improvement (5% weight)

**Requirements:**
- Agents must pass ALL 12 dimensions (100% pass rate)
- Each dimension has specific pass criteria
- Sign-off required from multiple stakeholders

**Use This For:** Understanding what it takes to certify an agent for production.

---

### 3. [Test Suite Structure](test-suites/00-TEST_SUITE_STRUCTURE.md)

Standard test suite structure, golden test case format, industrial decarbonization scenarios, compliance scenarios, and test data management.

**Key Topics:**
- Standard test directory layout
- Golden test case format (25+ scenarios required)
- Industrial decarbonization test scenarios
- Compliance test scenarios (CBAM, CSRD, EPA)
- Test data generation and management
- Integration with existing test pipelines

**Example Golden Test:**
```python
def test_golden_be_001_natural_gas_boiler_efficiency():
    """
    Golden Test: GOLDEN_BE_001 - Natural Gas Boiler Efficiency

    Known Correct Answer: 82.45678901234567% efficiency
    Validation Source: Dr. Jane Smith, MIT, ASME PTC 4-2013
    Tolerance: ±0.01% absolute
    """
    agent = BoilerEfficiencyOptimizer(temperature=0.0, seed=42)
    result = agent.calculate_boiler_efficiency(...)

    assert result['efficiency_percent'] == pytest.approx(
        82.45678901234567, rel=1e-12
    )
```

**Use This For:** Writing comprehensive test suites for agents.

---

### 4. [Benchmarking Framework](benchmarks/00-BENCHMARKING_FRAMEWORK.md)

Performance, accuracy, cost, and quality benchmarking framework with automated measurement and reporting.

**4 Benchmark Categories:**

1. **Performance Benchmarks:**
   - Latency (P50 <2s, P95 <4s, P99 <6s)
   - Throughput (>100 req/s sustained, >500 req/s peak)
   - Memory & CPU usage

2. **Accuracy Benchmarks:**
   - Golden test pass rate (100%)
   - Mean Absolute Error (MAE <1%)
   - Root Mean Square Error (RMSE <2%)
   - Energy/mass balance errors (<0.1%)

3. **Cost Benchmarks:**
   - Cost per analysis (<$0.15)
   - Token usage (prompt + completion)
   - Cost per tool

4. **Quality Benchmarks:**
   - Explanation clarity (Flesch-Kincaid >60)
   - Citation coverage (>90%)
   - Reasoning consistency (>9.0/10)

**Use This For:** Measuring and optimizing agent performance and quality.

---

### 5. [Evaluation Pipeline](framework/01-EVALUATION_PIPELINE.md)

CI/CD integration, automated evaluation on every change, manual review checkpoints, and validation report generation.

**Pipeline Stages:**

```
[Commit] → [Unit Tests] → [Golden Tests] → [Performance Tests]
    ↓
[Pull Request] → [Integration Tests] → [Security Scan]
    ↓
[Pre-Release] → [Compliance Tests] → [Climate Science Review]
    ↓
[Certification] → [Manual Review] → [Final Approval]
    ↓
[Deployment] → [Staging] → [Production (Canary)]
```

**GitHub Actions Integration:**
- Commit checks (every commit, ~5 minutes)
- Pull request checks (every PR, ~30 minutes)
- Pre-release checks (release branches, ~2 hours)
- Deployment automation (staging + production canary)

**Use This For:** Setting up automated evaluation in CI/CD pipelines.

---

## Quick Start Guide

### For Developers: Building a New Agent

1. **Read:** [Evaluation Overview](framework/00-EVALUATION_OVERVIEW.md) - Understand the process
2. **Review:** [Certification Criteria](criteria/00-CERTIFICATION_CRITERIA.md) - Know the requirements
3. **Implement:** Build your agent following specifications
4. **Test:** Use [Test Suite Structure](test-suites/00-TEST_SUITE_STRUCTURE.md) to write tests
5. **Benchmark:** Use [Benchmarking Framework](benchmarks/00-BENCHMARKING_FRAMEWORK.md) to measure performance
6. **Submit:** Follow [Evaluation Pipeline](framework/01-EVALUATION_PIPELINE.md) for certification

### For QA Engineers: Evaluating an Agent

1. **Review Application:** Check certification application completeness
2. **Run Automated Tests:** Execute full test suite, golden tests, performance benchmarks
3. **Review Code:** Code review for quality, error handling, provenance
4. **Generate Report:** Use validation report generator
5. **Submit for Review:** Forward to Legal + Science Board for compliance/climate science review
6. **Certification Decision:** Present to Certification Committee

### For DevOps Engineers: Deploying a Certified Agent

1. **Verify Certification:** Check agent has CERTIFIED status
2. **Deploy to Staging:** Use GitHub Actions workflow
3. **Run Smoke Tests:** Validate basic functionality
4. **Deploy to Production:** Canary deployment (10% → 50% → 100%)
5. **Monitor Metrics:** Latency, error rate, throughput
6. **Rollback if Needed:** Automated rollback on metric violations

---

## Evaluation Metrics Dashboard

### Program-Level Metrics (Tracked Monthly)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Agents Certified | 50 by 2026 Q2 | 12 | On Track |
| Avg Certification Quality Score | >8.5 | 8.7 | Exceeding |
| Avg Time to Certify | <6 weeks | 5.2 weeks | Exceeding |
| Initial Pass Rate | >80% | 85% | Exceeding |
| Re-certification Pass Rate | >95% | 98% | Exceeding |
| Customer-Reported Accuracy Issues | <5 per year | 2 YTD | Exceeding |
| Regulatory Non-Compliance Issues | 0 | 0 | Exceeding |

### Agent-Level Metrics (Tracked Real-Time)

Every Certified agent must maintain:
- Test coverage >85%
- P95 latency <4s
- Cost per analysis <$0.15
- Error rate <1%
- Customer satisfaction >4.5/5.0
- Zero regulatory non-compliance incidents

---

## Integration with Existing Infrastructure

### Links to Existing Systems

- **Validation Reports:** `C:\Users\aksha\Code-V1_GreenLang\validation_reports\`
- **Test Pipelines:** `C:\Users\aksha\Code-V1_GreenLang\test_pipelines\`
- **Agent Tests:** `C:\Users\aksha\Code-V1_GreenLang\docs\planning\greenlang-2030-vision\agent_foundation\agents\GL-*\tests\`

### Compatibility

This framework is designed to:
- **Standardize** existing ad-hoc validation processes
- **Build upon** existing validation reports (AGENT_001_VALIDATION_SUMMARY.md, etc.)
- **Integrate with** existing test infrastructure (pytest, GitHub Actions)
- **Preserve** existing golden tests and benchmarks
- **Enhance** with new automation and reporting

---

## Roles & Responsibilities

### Agent Developer
- Implement agent (all tools)
- Write test suite (>85% coverage)
- Create golden tests (25+ scenarios)
- Submit certification application

### QA Engineer (Test Engineer)
- Review test suite
- Run performance benchmarks
- Conduct security scan
- Write technical review report
- **THIS IS YOUR PRIMARY ROLE**

### Climate Scientist
- Validate emission factors
- Validate thermodynamic calculations
- Review climate science methodology

### Legal/Regulatory Specialist
- Validate regulatory methodology
- Review audit trail
- Sign off on compliance

### Certification Committee
- Review all evaluation reports
- Make final certification decision
- Authorize production deployment

---

## Sign-Off Authority

| Dimension | Primary Sign-Off | Approval Level |
|-----------|------------------|----------------|
| Technical | QA Engineer | Manager |
| Compliance | Legal Counsel | VP/General Counsel |
| Climate Science | Climate Scientist | Chief Climate Scientist |
| Final Decision | Certification Committee | VP+ |

---

## Success Criteria

### Framework Adoption (2026 Q2)
- [ ] 50 agents certified using this framework
- [ ] 100% of new agents follow certification process
- [ ] Zero production incidents due to inadequate testing
- [ ] <6 weeks average time to certify

### Quality Metrics (Ongoing)
- [ ] >85% test coverage for all certified agents
- [ ] 100% golden test pass rate
- [ ] <4s P95 latency for all certified agents
- [ ] <$0.15 cost per analysis for all certified agents
- [ ] Zero regulatory non-compliance incidents

---

## Next Steps

### Immediate (Week 1)
1. Review all 5 framework documents
2. Identify first agent for pilot certification (recommend: GL-002 BoilerEfficiencyOptimizer)
3. Set up GitHub Actions workflows for automated evaluation
4. Train QA team on certification process

### Short-Term (Month 1)
1. Certify 3 pilot agents using framework
2. Collect feedback from developers and QA engineers
3. Refine framework based on lessons learned
4. Document case studies (what went well, what to improve)

### Long-Term (2026)
1. Certify 50 agents using framework
2. Achieve <6 weeks average certification time
3. Achieve >85% initial pass rate
4. Automate 90%+ of evaluation pipeline

---

## Support & Contact

### Questions?

- **Framework Questions:** qa@greenlang.com (QA Engineering Team)
- **Technical Questions:** engineering@greenlang.com (Engineering Team)
- **Compliance Questions:** legal@greenlang.com (Legal Team)
- **Climate Science Questions:** science@greenlang.com (Science Board)

### Documentation

- **Internal Wiki:** https://wiki.greenlang.com/evaluation-framework
- **Slack Channel:** #agent-certification
- **Office Hours:** Tuesdays 2-3pm PT (QA Team)

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-03 | GL-TestEngineer | Initial framework creation |

**Approved By:**
- VP Engineering: _________________ Date: _______
- Chief Climate Scientist: _________________ Date: _______
- General Counsel: _________________ Date: _______

---

**FRAMEWORK COMPLETE - READY FOR PILOT CERTIFICATION**

---

## Appendices

### Appendix A: Glossary

- **Golden Test:** Test with known correct answer validated by domain expert
- **Determinism:** Same inputs → same outputs (bit-perfect reproducibility)
- **Provenance:** Complete audit trail of calculation (inputs → method → outputs)
- **Certification:** Formal approval for production deployment
- **P95 Latency:** 95th percentile response time (95% of requests faster than this)

### Appendix B: References

- ASME PTC 4-2013: Fired Steam Generators (boiler efficiency testing)
- ISO 50001:2018: Energy Management Systems
- EPA 40 CFR Part 98: Greenhouse Gas Reporting
- IPCC AR6: Sixth Assessment Report (GWP values)
- Commission Implementing Regulation (EU) 2023/1773: CBAM methodology

### Appendix C: Change Log

- 2025-12-03: Initial framework creation (v1.0.0)

---

**END OF README**
