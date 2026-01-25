# Agent Factory: Team KPIs

**Version:** 1.0
**Date:** 2025-12-03
**Program:** Agent Factory

---

## Overview

This document defines Key Performance Indicators (KPIs) for each team, aligned with program objectives. KPIs are measured weekly/monthly and reviewed in sprint retrospectives and all-hands meetings.

**Measurement Philosophy:**
- **Outcome-focused:** Measure results, not activity
- **Actionable:** Teams can influence their KPIs
- **Balanced:** Mix of quality, velocity, and reliability
- **Transparent:** All KPIs visible to entire program

---

## Program-Level North Star Metrics

### Overall Program Success

| Metric | Phase 1 Target | Phase 2 Target | Phase 3 Target | Owner |
|--------|---------------|---------------|---------------|-------|
| **Agents Generated** | 1 | 10 | 100 | All Teams |
| **Certification Pass Rate (1st Attempt)** | >90% | >95% | >98% | AI/Agent + Climate Science |
| **Agent Generation Time** | <2 hours | <1 hour | <30 min | AI/Agent + ML Platform |
| **Zero-Hallucination Rate** | 100% | 100% | 100% | ML Platform + Climate Science |
| **Platform Uptime** | 99.9% | 99.95% | 99.99% | DevOps |
| **Cost per Agent** | <$100 | <$50 | <$20 | All Teams |

**Reporting:**
- **Frequency:** Weekly (dashboard), Monthly (all-hands presentation)
- **Owner:** Product Manager
- **Dashboard:** Grafana `Agent Factory - Program Overview`

---

## ML Platform Team KPIs

### 1. Model Infrastructure Quality

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Model API Uptime** | 99.95% | `uptime{service="model-api"} / total_time` | Real-time | Grafana: Model Health |
| **Model API Latency (p95)** | <3 sec | `histogram_quantile(0.95, model_request_duration_seconds)` | Real-time | Grafana: Model Health |
| **Zero-Hallucination Rate** | 100% | `deterministic_outputs / total_outputs` | Daily | Custom: Determinism Report |
| **Model Correctness** | >95% | `golden_tests_passed / golden_tests_total` | Per test run | Custom: Golden Test Results |
| **Cost per Agent (Model Tokens)** | <$50 | `total_token_cost / agents_generated` | Monthly | Grafana: Cost Dashboard |

**Weekly Review:**
- Review failed golden tests
- Analyze latency outliers (>5 sec)
- Token usage optimization opportunities

**Monthly Review:**
- Model performance trends
- Cost optimization initiatives
- Golden test suite expansion

---

### 2. Evaluation Harness Quality

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Golden Test Suite Size** | 500 (P1), 1,000 (P2), 2,000 (P3) | `COUNT(golden_tests)` | Monthly | Custom: Test Coverage |
| **Test Execution Time** | <5 min for 100 tests | `test_suite_duration_seconds / 100` | Per test run | Custom: Test Performance |
| **Test Pass Rate (Certified Agents)** | >95% | `tests_passed / tests_total` | Per certification | Custom: Certification Report |
| **Regression Test Coverage** | 100% | `regression_tests / total_features` | Monthly | Custom: Test Coverage |

**Weekly Review:**
- New golden tests added
- Flaky test identification and fixes
- Test execution performance

**Monthly Review:**
- Test suite quality (coverage, edge cases)
- Regression test effectiveness
- Human evaluation requirements

---

### 3. Observability

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Model Telemetry Coverage** | 100% | `instrumented_endpoints / total_endpoints` | Weekly | Custom: Observability Coverage |
| **Alert Accuracy** | >90% | `actionable_alerts / total_alerts` | Monthly | Grafana: Alert Dashboard |
| **Dashboard Uptime** | 99.9% | `dashboard_uptime / total_time` | Real-time | Grafana: System Health |

**Weekly Review:**
- False positive alerts
- Missing telemetry gaps
- Dashboard improvements

---

## AI/Agent Team KPIs

### 1. Agent Generation Quality

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Agent Generation Success Rate** | >95% | `successful_generations / total_attempts` | Real-time | Grafana: Agent Factory Health |
| **Agent Generation Time (p95)** | <2 hrs (P1), <1 hr (P2), <30 min (P3) | `histogram_quantile(0.95, generation_duration_seconds)` | Real-time | Grafana: Agent Factory Health |
| **Code Quality Score** | >90/100 | `avg(linting_score)` | Per generation | Custom: Quality Report |
| **Test Coverage (Generated Agents)** | >85% | `avg(test_coverage_percent)` | Per generation | Custom: Quality Report |
| **Certification Pass Rate (1st Attempt)** | >90% (P1), >95% (P2), >98% (P3) | `certifications_passed_first / certifications_total` | Monthly | Custom: Certification Report |

**Weekly Review:**
- Failed agent generations (root cause)
- Code quality issues
- Certification failures

**Monthly Review:**
- Generation time trends
- Quality score improvements
- Certification success factors

---

### 2. Agent SDK Quality

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **SDK Adoption** | 100% | `agents_using_sdk / total_agents` | Monthly | Custom: SDK Adoption |
| **SDK Test Coverage** | >90% | `sdk_test_coverage_percent` | Per release | CodeCov |
| **SDK Performance Overhead** | <10ms | `avg(sdk_overhead_ms)` | Per agent call | Grafana: SDK Performance |
| **SDK Documentation Coverage** | 100% | `documented_apis / total_apis` | Per release | Custom: Documentation Report |
| **SDK Bug Rate** | <5 bugs/release | `bugs_reported / releases` | Per release | GitHub Issues |

**Weekly Review:**
- SDK bug reports and fixes
- Performance bottlenecks
- Documentation gaps

**Monthly Review:**
- SDK adoption trends
- Feature requests prioritization
- Breaking changes planning

---

### 3. AgentSpec Evolution

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **AgentSpec Validation Accuracy** | 100% | `correct_validations / total_validations` | Per validation | Custom: Validation Report |
| **AgentSpec Coverage** | 100% | `regulations_covered / total_regulations` | Monthly | Custom: Coverage Report |
| **Backward Compatibility** | 100% | `old_agentspecs_working / total_old_agentspecs` | Per schema change | Custom: Compatibility Test |

**Weekly Review:**
- AgentSpec validation failures
- Schema change requests
- Coverage gaps

---

## Climate Science Team KPIs

### 1. Validation Accuracy

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Validation Hook Accuracy** | 100% | `correct_validations / total_validations` | Per validation | Custom: Validation Report |
| **Validation Hook Latency** | <1 sec | `avg(validation_duration_seconds)` | Real-time | Grafana: Validation Performance |
| **Regulatory Compliance** | 100% | `compliant_agents / certified_agents` | Per certification | Custom: Compliance Report |
| **Audit Readiness** | 100% | `audit_ready_agents / certified_agents` | Monthly | Custom: Audit Report |

**Weekly Review:**
- Validation failures (false positives/negatives)
- Performance bottlenecks
- Compliance gaps

**Monthly Review:**
- Regulatory changes impact
- Validation rule updates
- Audit preparation

---

### 2. Certification Quality

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Certification Throughput** | <2 days | `avg(certification_duration_days)` | Per certification | Custom: Certification Report |
| **Certification Pass Rate (1st Attempt)** | >90% (P1), >95% (P2), >98% (P3) | `passed_first_attempt / total_certifications` | Monthly | Custom: Certification Report |
| **Re-Certification Success** | 100% | `re_certifications_passed / re_certifications_total` | Quarterly | Custom: Re-Cert Report |

**Weekly Review:**
- Certification queue status
- Failed certifications (root cause)
- Manual review requirements

**Monthly Review:**
- Certification process improvements
- Automated vs. manual review ratio
- Certification criteria updates

---

### 3. Golden Test Quality

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Golden Test Suite Size** | 100 (P1), 2,000 (P2), 5,000 (P3) | `COUNT(golden_tests)` | Monthly | Custom: Test Coverage |
| **Golden Test Coverage** | 100% | `regulations_tested / total_regulations` | Monthly | Custom: Test Coverage |
| **Golden Test Quality** | >95% | `tests_with_high_confidence / total_tests` | Monthly | Custom: Test Quality Report |
| **Test Contribution Rate** | 50 tests/month | `new_tests_added / month` | Monthly | Custom: Test Contribution |

**Weekly Review:**
- New golden tests contributed
- Test quality issues
- Coverage gaps

**Monthly Review:**
- Test suite expansion planning
- Test maintenance (updates for regulation changes)
- Test quality improvements

---

### 4. Regulatory Intelligence

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Regulation Update Detection Time** | <7 days | `avg(detection_time_days)` | Per update | Custom: Regulation Tracker |
| **Impact Assessment Time** | <14 days | `avg(assessment_time_days)` | Per update | Custom: Regulation Tracker |
| **Agent Update Time** | <30 days | `avg(update_deployment_days)` | Per update | Custom: Regulation Tracker |

**Monthly Review:**
- Regulation changes detected
- Impact assessments completed
- Agent updates deployed

---

## Platform Team KPIs

### 1. Agent Registry Quality

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Registry Uptime** | 99.95% (P1), 99.99% (P3) | `uptime{service="agent-registry"} / total_time` | Real-time | Grafana: Registry Health |
| **Registry API Latency (p95)** | <100ms (P1), <50ms (P3) | `histogram_quantile(0.95, registry_api_duration_seconds)` | Real-time | Grafana: Registry Health |
| **Agents Registered** | 100 (P1), 500 (P2), 2,000 (P3) | `COUNT(agents)` | Monthly | Custom: Registry Stats |
| **Search Performance** | <500ms | `avg(search_duration_ms)` | Real-time | Grafana: Search Performance |

**Weekly Review:**
- Registry performance issues
- Search query optimization
- Storage cost trends

**Monthly Review:**
- Registry growth trends
- Feature requests (search, filters)
- Cost optimization

---

### 2. SDK Core Quality

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **SDK Adoption** | 100% | `agents_using_sdk_core / total_agents` | Monthly | Custom: SDK Adoption |
| **Authentication Success Rate** | >99.9% | `successful_auths / total_auths` | Real-time | Grafana: Auth Health |
| **SDK Performance Overhead** | <10ms | `avg(sdk_overhead_ms)` | Per call | Grafana: SDK Performance |
| **SDK Test Coverage** | >90% | `sdk_test_coverage_percent` | Per release | CodeCov |

**Weekly Review:**
- Authentication failures
- Performance bottlenecks
- Bug reports

**Monthly Review:**
- SDK usage trends
- Feature requests
- Breaking changes planning

---

### 3. CLI Tools Quality

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **CLI Adoption** | >80% of developers | `cli_users / total_developers` | Monthly | Custom: CLI Usage |
| **CLI Command Success Rate** | >95% | `successful_commands / total_commands` | Real-time | Custom: CLI Telemetry |
| **CLI Documentation Coverage** | 100% | `documented_commands / total_commands` | Per release | Custom: Documentation Report |
| **CLI User Satisfaction (NPS)** | >60 | `NPS_score` | Quarterly | Survey |

**Weekly Review:**
- CLI failures (root cause)
- Feature requests
- Documentation gaps

**Monthly Review:**
- CLI usage trends
- UX improvements
- New commands planning

---

### 4. API Gateway Quality

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **API Gateway Uptime** | 99.95% | `uptime{service="api-gateway"} / total_time` | Real-time | Grafana: Gateway Health |
| **API Gateway Latency (p95)** | <200ms (P1), <100ms (P2), <50ms (P3) | `histogram_quantile(0.95, gateway_latency_seconds)` | Real-time | Grafana: Gateway Health |
| **Rate Limit Accuracy** | 100% | `correct_rate_limits / total_rate_limits` | Real-time | Custom: Rate Limit Report |
| **API Documentation Coverage** | 100% | `documented_endpoints / total_endpoints` | Per release | Custom: Documentation Report |

**Weekly Review:**
- API performance issues
- Rate limit violations
- Documentation updates

---

## Data Engineering Team KPIs

### 1. Data Contract Quality

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Data Contract Coverage** | 100% | `contracts_defined / data_flows` | Monthly | Custom: Contract Coverage |
| **Contract Validation Accuracy** | 100% | `correct_validations / total_validations` | Per validation | Custom: Validation Report |
| **Contract Stability** | 100% (no breaking changes) | `breaking_changes / contract_updates` | Per release | Custom: Contract Changelog |
| **Contract Documentation Coverage** | 100% | `documented_contracts / total_contracts` | Per release | Custom: Documentation Report |

**Weekly Review:**
- Contract validation failures
- Schema change requests
- Documentation gaps

**Monthly Review:**
- Contract usage trends
- Backward compatibility issues
- Schema evolution planning

---

### 2. Data Pipeline Quality

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Pipeline Uptime** | 99.9% | `successful_runs / total_runs` | Real-time | Grafana: Pipeline Health |
| **Data Latency** | <1 hour (P1), <10 min (P2), <1 min (P3) | `avg(data_latency_seconds)` | Real-time | Grafana: Pipeline Health |
| **Pipeline Success Rate** | >99% | `successful_runs / total_runs` | Real-time | Grafana: Pipeline Health |
| **Data Quality Score** | >99.9% | `valid_records / total_records` | Per pipeline run | Custom: Data Quality Report |

**Daily Review:**
- Failed pipeline runs (root cause)
- Data quality issues
- Latency spikes

**Weekly Review:**
- Pipeline performance optimization
- Data quality trends
- Capacity planning

---

### 3. Data Quality

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Data Quality Score** | >99.9% | `valid_records / total_records` | Per pipeline run | Custom: Data Quality Report |
| **Data Completeness** | 100% | `non_null_required_fields / total_required_fields` | Per pipeline run | Custom: Data Quality Report |
| **Data Accuracy** | >99.9% | `accurate_records / total_records` | Per validation | Custom: Data Quality Report |
| **Anomaly Detection Accuracy** | >95% | `true_anomalies / detected_anomalies` | Monthly | Custom: Anomaly Report |

**Daily Review:**
- Data quality issues
- Anomalies detected
- Validation failures

**Weekly Review:**
- Data quality trends
- Validation rule updates
- Anomaly detection tuning

---

### 4. Data Provenance

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Provenance Coverage** | 100% | `tracked_transformations / total_transformations` | Per pipeline run | Custom: Provenance Report |
| **Lineage Completeness** | 100% | `complete_lineages / total_data_flows` | Monthly | Custom: Lineage Report |
| **Audit Trail Integrity** | 100% | `verified_hashes / total_hashes` | Monthly | Custom: Audit Report |

**Monthly Review:**
- Provenance coverage gaps
- Lineage visualization improvements
- Audit trail compliance

---

## DevOps/SRE/Security Team KPIs

### 1. Platform Reliability (SRE)

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Platform Uptime (Agent Factory)** | 99.9% (P1), 99.95% (P2), 99.99% (P3) | `uptime / total_time` | Real-time | Grafana: Platform Health |
| **Mean Time to Detect (MTTD)** | <5 min | `avg(detection_time_minutes)` | Per incident | Custom: Incident Report |
| **Mean Time to Resolve (MTTR)** | <1 hour (P1), <30 min (P2) | `avg(resolution_time_minutes)` | Per incident | Custom: Incident Report |
| **Incident Postmortem Completion** | 100% | `postmortems_completed / incidents` | Monthly | Custom: Incident Report |
| **SLO Compliance** | >99.9% | `time_within_slo / total_time` | Monthly | Grafana: SLO Dashboard |

**Daily Review:**
- Incidents (active and resolved)
- MTTD/MTTR trends
- SLO violations

**Weekly Review:**
- Incident root causes
- Postmortem action items
- On-call rotation effectiveness

**Monthly Review:**
- Reliability trends
- SLO/SLA updates
- Capacity planning

---

### 2. Deployment Velocity

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Deployment Frequency** | 5/day (P1), 10/day (P2), 20/day (P3) | `COUNT(deployments) / days` | Daily | Grafana: Deployment Dashboard |
| **Deployment Success Rate** | >95% | `successful_deployments / total_deployments` | Real-time | Grafana: Deployment Dashboard |
| **Deployment Time** | <10 min (CI) + <5 min (CD) | `avg(deployment_duration_minutes)` | Per deployment | Grafana: Deployment Dashboard |
| **Rollback Time** | <5 min | `avg(rollback_duration_minutes)` | Per rollback | Custom: Rollback Report |
| **Change Failure Rate** | <5% | `failed_deployments / total_deployments` | Monthly | Grafana: Deployment Dashboard |

**Daily Review:**
- Failed deployments (root cause)
- Deployment time trends
- CI/CD pipeline health

**Weekly Review:**
- Deployment process improvements
- Rollback frequency
- Change failure analysis

---

### 3. Security

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Zero Critical Vulnerabilities** | 100% of time | `days_with_zero_critical_cves / total_days` | Real-time | Grafana: Security Dashboard |
| **Mean Time to Patch (MTTP)** | <24 hours | `avg(patch_time_hours)` | Per CVE | Custom: Security Report |
| **Security Scan Coverage** | 100% | `scanned_code / total_code` | Per commit | Custom: Security Report |
| **Secret Rotation Compliance** | 100% | `rotated_secrets / total_secrets` | Quarterly | Custom: Secret Rotation Report |
| **Audit Log Completeness** | 100% | `logged_actions / total_actions` | Real-time | Custom: Audit Report |

**Daily Review:**
- New vulnerabilities detected
- Security scan failures
- Patch status

**Weekly Review:**
- Vulnerability trends
- Security scanning improvements
- Access control review

**Monthly Review:**
- Security posture assessment
- Compliance audit preparation
- Penetration test findings

---

### 4. Infrastructure Efficiency

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Infrastructure Cost** | <$10K/month (P1), <$20K/month (P2), <$50K/month (P3) | `total_cloud_cost / month` | Monthly | Grafana: Cost Dashboard |
| **Cost per Agent** | <$100 (P1), <$50 (P2), <$20 (P3) | `total_cost / agents_generated` | Monthly | Grafana: Cost Dashboard |
| **Resource Utilization (CPU)** | 60-80% | `avg(cpu_utilization_percent)` | Real-time | Grafana: Resource Dashboard |
| **Resource Utilization (Memory)** | 60-80% | `avg(memory_utilization_percent)` | Real-time | Grafana: Resource Dashboard |
| **Autoscaling Effectiveness** | >90% | `successful_scale_events / total_scale_events` | Real-time | Grafana: Autoscaling Dashboard |

**Daily Review:**
- Cost anomalies
- Resource utilization trends
- Autoscaling events

**Weekly Review:**
- Cost optimization opportunities
- Resource rightsizing
- Capacity planning

**Monthly Review:**
- Budget vs. actual
- Cost reduction initiatives
- Infrastructure optimization

---

### 5. Observability

| KPI | Target | Measurement | Frequency | Dashboard |
|-----|--------|-------------|-----------|-----------|
| **Monitoring Coverage** | 100% | `monitored_services / total_services` | Weekly | Custom: Observability Report |
| **Alert Accuracy** | >90% | `actionable_alerts / total_alerts` | Monthly | Grafana: Alert Dashboard |
| **Dashboard Uptime** | 99.9% | `dashboard_uptime / total_time` | Real-time | Grafana: System Health |
| **Log Retention Compliance** | 100% | `logs_retained / logs_required` | Daily | Custom: Logging Report |
| **Trace Coverage** | >80% | `traced_requests / total_requests` | Real-time | Grafana: Tracing Dashboard |

**Weekly Review:**
- Monitoring gaps
- False positive alerts
- Dashboard improvements

**Monthly Review:**
- Observability maturity
- Logging/tracing optimization
- Alert rule tuning

---

## KPI Reporting Structure

### Weekly Team Review (Fridays 3:00 PM)

**Format:** 30-minute team meeting

**Agenda:**
1. Review weekly KPIs (10 min)
2. Discuss misses and root causes (10 min)
3. Action items for next week (10 min)

**Output:** Slack summary in team channel

---

### Monthly All-Hands (First Friday 3:00 PM)

**Format:** 60-minute program-wide meeting

**Agenda:**
1. Program-level KPI review (15 min)
2. Team highlights and demos (30 min)
3. Risks and blockers (10 min)
4. Next month priorities (5 min)

**Output:** Confluence page with KPI dashboard screenshots and narrative

---

### Quarterly Business Review (QBR)

**Format:** 90-minute executive presentation

**Attendees:** Product Manager, Engineering Lead, All Tech Leads, Stakeholders

**Agenda:**
1. Program progress vs. roadmap (20 min)
2. KPI trends and analysis (30 min)
3. Wins and challenges (20 min)
4. Next quarter priorities (20 min)

**Output:** Executive summary deck (10-15 slides)

---

## KPI Dashboard Access

### Grafana Dashboards (Public)

- **Agent Factory - Program Overview:** https://grafana.greenlang.com/d/program-overview
- **Model Health:** https://grafana.greenlang.com/d/model-health
- **Agent Factory Health:** https://grafana.greenlang.com/d/agent-factory-health
- **Platform Health:** https://grafana.greenlang.com/d/platform-health
- **Pipeline Health:** https://grafana.greenlang.com/d/pipeline-health
- **Deployment Dashboard:** https://grafana.greenlang.com/d/deployment
- **Cost Dashboard:** https://grafana.greenlang.com/d/cost
- **Security Dashboard:** https://grafana.greenlang.com/d/security

### Custom Dashboards (Notion/Confluence)

- **Certification Report:** Updated weekly
- **Golden Test Coverage:** Updated daily
- **Data Quality Report:** Updated per pipeline run
- **Incident Report:** Updated per incident
- **SDK Adoption:** Updated monthly

---

## KPI Target Evolution

### Phase 1 → Phase 2 Transitions (Week 16)

**Tightened Targets:**
- Platform uptime: 99.9% → 99.95%
- Agent generation time: <2 hours → <1 hour
- API latency: <200ms → <100ms
- Deployment frequency: 5/day → 10/day

**New KPIs Added:**
- Multi-region availability
- Streaming pipeline latency
- Advanced security metrics

---

### Phase 2 → Phase 3 Transitions (Week 28)

**Tightened Targets:**
- Platform uptime: 99.95% → 99.99%
- Agent generation time: <1 hour → <30 min
- Cost per agent: <$50 → <$20
- Certification pass rate: >95% → >98%

**New KPIs Added:**
- Enterprise SLAs
- Compliance certifications (SOC 2, ISO 27001)
- Multi-tenant metrics

---

## KPI Accountability

| KPI Category | Responsible Team | Accountable (Escalation) |
|-------------|------------------|-------------------------|
| **Model Infrastructure** | ML Platform | ML Platform Tech Lead |
| **Agent Generation** | AI/Agent | AI/Agent Tech Lead |
| **Validation & Certification** | Climate Science | Climate Science Tech Lead |
| **Registry & SDK** | Platform | Platform Tech Lead |
| **Data Pipelines & Quality** | Data Engineering | Data Engineering Tech Lead |
| **Infrastructure & Security** | DevOps | DevOps Tech Lead |
| **Program-Level** | Product Manager | Engineering Lead |

---

## Appendix: KPI Formulas

### Uptime Calculation
```
Uptime % = (Total Time - Downtime) / Total Time × 100

Where:
- Total Time = 30 days × 24 hours × 60 minutes = 43,200 minutes
- Downtime = Sum of all outage durations (in minutes)
- Target: 99.95% = Max 21.6 minutes downtime per month
```

### Latency (p95) Calculation
```
p95 Latency = 95th percentile of all request latencies

Prometheus query:
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### Cost per Agent Calculation
```
Cost per Agent = Total Monthly Cost / Agents Generated

Where:
- Total Monthly Cost = Infrastructure + Model Tokens + Labor
- Agents Generated = COUNT(agents) WHERE status = "certified"
```

### Certification Pass Rate Calculation
```
Pass Rate % = (Certifications Passed on 1st Attempt / Total Certification Requests) × 100

Example:
- 90 agents passed certification on 1st attempt
- 5 agents required changes and re-certification
- Total: 95 certification requests
- Pass Rate = (90 / 95) × 100 = 94.7%
```

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-03 | GL Product Manager | Initial team KPIs |

---

**Approvals:**

- Product Manager: ___________________
- Engineering Lead: ___________________
- All Tech Leads: ___________________
