# ADR-XXX: [Title - Short Description]

**Date:** YYYY-MM-DD
**Status:** [Proposed | Accepted | Rejected | Superseded | Deprecated]
**Deciders:** [List of people involved in decision]
**Consulted:** [List of people consulted]

---

## Context

### Problem Statement
What is the issue we're facing that requires custom implementation instead of GreenLang infrastructure?

### Current Situation
- What infrastructure exists in GreenLang?
- Why can't we use it?
- What constraints are we facing?

### Business Impact
- What business value does this deliver?
- What's the urgency?
- What happens if we don't implement this?

---

## Decision

### What We're Implementing
Clear description of the custom implementation.

### Technology Stack
- Languages/frameworks used
- External dependencies
- Infrastructure requirements

### Code Location
Where will this custom code live in the codebase?

---

## Rationale

### Why GreenLang Infrastructure Can't Support This

**Specific Limitations:**
1. [Limitation 1]
2. [Limitation 2]
3. [Limitation 3]

**What Would Need to Change in GreenLang:**
- Infrastructure changes required
- Estimated effort to add to core
- Timeline considerations

---

## Alternatives Considered

### Alternative 1: [Description]
**Pros:**
- [Pro 1]
- [Pro 2]

**Cons:**
- [Con 1]
- [Con 2]

**Why Rejected:** [Reason]

### Alternative 2: [Description]
**Pros:**
- [Pro 1]
- [Pro 2]

**Cons:**
- [Con 1]
- [Con 2]

**Why Rejected:** [Reason]

### Alternative 3: Wait for GreenLang Support
**Pros:**
- Would be fully compliant
- Would benefit from infrastructure updates

**Cons:**
- Timeline not acceptable
- Business requirements can't wait

**Why Rejected:** [Reason]

---

## Consequences

### Positive
- [Benefit 1]
- [Benefit 2]
- [Benefit 3]

### Negative
- **Technical Debt:** Custom code to maintain
- **No Auto-Updates:** Won't benefit from GreenLang infrastructure improvements
- **Custom Monitoring:** Need to implement own logging/metrics
- **Security:** Need to maintain security separately
- [Other negative consequence]

### Neutral
- [Neutral consequence 1]
- [Neutral consequence 2]

---

## Implementation Plan

### Phase 1: Development
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Phase 2: Testing
- Unit tests
- Integration tests
- Security review
- Performance benchmarks

### Phase 3: Deployment
- Rollout strategy
- Monitoring setup
- Documentation

### Phase 4: Maintenance
- Who owns this code?
- How will it be maintained?
- Update schedule

---

## Compliance & Security

### Security Considerations
- Authentication/Authorization
- Data encryption
- Secrets management
- Audit logging
- Compliance requirements (SOC2, GDPR, etc.)

### Monitoring & Observability
- Metrics to track
- Logs to collect
- Alerts to configure
- Dashboard to create

### Testing Strategy
- Unit test coverage target: [X%]
- Integration tests
- Performance tests
- Security tests

---

## Migration Plan

### Short-term (0-6 months)
What custom implementation will we use?

### Medium-term (6-12 months)
Can we contribute this to GreenLang infrastructure?

### Long-term (12+ months)
Plan to migrate back to GreenLang infrastructure when available.

**Migration Trigger:**
When [condition is met], we will migrate to GreenLang infrastructure.

**Estimated Migration Effort:**
[X person-days]

---

## Documentation

### User Documentation
- [ ] Usage guide written
- [ ] Examples provided
- [ ] API documentation
- [ ] Troubleshooting guide

### Developer Documentation
- [ ] Architecture documented
- [ ] Code comments added
- [ ] Deployment guide
- [ ] Runbook created

### Team Communication
- [ ] Team notified
- [ ] Knowledge sharing session scheduled
- [ ] Wiki page created

---

## Review & Approval

### Technical Review
- [ ] Security team approval
- [ ] Architecture team approval
- [ ] DevOps team approval

### Business Review
- [ ] Product owner approval
- [ ] Stakeholder sign-off

### Approvals
- **Engineering Lead:** [Name] - [Date]
- **Security Lead:** [Name] - [Date]
- **Architecture Lead:** [Name] - [Date]

---

## Links & References

- Related GitHub Issues: #XXX
- Related PRs: #XXX
- External Documentation: [URL]
- Slack Discussion: [Link]

---

## Updates

### [Date] - Status Change
[Description of change]

### [Date] - Implementation Complete
[Notes]

### [Date] - Superseded by ADR-YYY
[Reason for superseding]

---

**Template Version:** 1.0
**Last Updated:** 2024-11-09
