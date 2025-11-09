# ADR-XXX: [Short Title of Decision]

**Status:** [Proposed | Accepted | Rejected | Superseded]
**Date:** YYYY-MM-DD
**Author:** [Your Name]
**Deciders:** [List of people involved in decision]

---

## Summary

Brief 2-3 sentence summary of the decision and its impact.

---

## Context

### Problem Statement

What problem are we trying to solve? Why is a decision needed?

### Background

Provide context about:
- Current situation
- Constraints
- Requirements
- Stakeholders affected

---

## Infrastructure Evaluation

**This section is REQUIRED for all custom code requests.**

### Infrastructure Components Evaluated

| Component | Location | Why Not Used |
|-----------|----------|--------------|
| [Component Name] | `greenlang.x.y.Component` | [Specific reason] |
| [Component Name] | `greenlang.x.y.Component` | [Specific reason] |

### Detailed Evaluation

For each infrastructure component:

#### Component: [Name]

**Evaluated:** Yes/No
**Location:** `greenlang.x.y.Component`
**Why it doesn't meet our needs:**
- Reason 1
- Reason 2
- Reason 3

**Could it be enhanced instead?**
- Yes/No
- If yes, why enhancement isn't viable (timeline, scope, etc.)

---

## Decision

### What We Decided

Clear statement of the decision made.

### Rationale

Why did we choose this approach?
- Reason 1
- Reason 2
- Reason 3

---

## Alternatives Considered

### Alternative 1: [Name]

**Description:** Brief description

**Pros:**
- Pro 1
- Pro 2

**Cons:**
- Con 1
- Con 2

**Why not chosen:** Explanation

### Alternative 2: [Name]

**Description:** Brief description

**Pros:**
- Pro 1
- Pro 2

**Cons:**
- Con 1
- Con 2

**Why not chosen:** Explanation

### Alternative 3: Use Infrastructure As-Is

**Description:** Use existing GreenLang infrastructure without modifications

**Pros:**
- Zero custom code
- Maintained by infrastructure team
- Battle-tested

**Cons:**
- [List specific limitations]

**Why not chosen:** [Explanation]

---

## Consequences

### Positive Consequences

- Consequence 1
- Consequence 2
- Consequence 3

### Negative Consequences

- Consequence 1
- Consequence 2
- Consequence 3

### Trade-offs

What are we giving up? What are we gaining?

---

## Implementation

### Code Location

Where will custom code live?
- Directory: `path/to/custom/code`
- Files: List of files to be created/modified

### Estimated Effort

- Development: X days
- Testing: X days
- Documentation: X days
- **Total: X days**

### Maintenance Plan

Who will maintain this custom code?
- Owner: [Team/Person]
- Review cadence: [Monthly/Quarterly]
- Sunset plan: [When/how will this be replaced]

---

## Documentation Requirements

**All custom code MUST be documented:**

- [ ] Code comments explaining why custom (reference this ADR)
- [ ] README section documenting custom components
- [ ] API documentation (if applicable)
- [ ] Unit tests (90%+ coverage)
- [ ] Integration tests
- [ ] Migration guide (if replacing existing code)

---

## Approval Checklist

**Required approvals before implementation:**

- [ ] Infrastructure team reviewed alternatives
- [ ] Architecture team approved design
- [ ] Security team reviewed (if touching auth, secrets, PII)
- [ ] Product team approved (if affecting user experience)
- [ ] CTO/Tech Lead final approval

**Approval signatures:**

- Infrastructure: __________________ Date: __________
- Architecture: __________________ Date: __________
- Security: __________________ Date: __________ (if applicable)
- Product: __________________ Date: __________ (if applicable)
- Tech Lead: __________________ Date: __________

---

## Metrics & Success Criteria

### How will we measure success?

- Metric 1: [e.g., Performance < X ms]
- Metric 2: [e.g., Cost < $X/month]
- Metric 3: [e.g., Development time < X days]

### Review Timeline

- **First review:** [Date] (1 month after implementation)
- **Second review:** [Date] (3 months after implementation)
- **Annual review:** [Date]

---

## References

### Related ADRs

- ADR-XXX: [Title]
- ADR-YYY: [Title]

### External References

- [Link to infrastructure catalog section]
- [Link to relevant documentation]
- [Link to GitHub issues]

### Discussion

- Discord: #infrastructure [Date]
- GitHub: Issue #XXX
- RFC: [Link]

---

## Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| YYYY-MM-DD | 1.0 | [Name] | Initial version |
| YYYY-MM-DD | 1.1 | [Name] | Updated based on feedback |

---

## Notes

Any additional notes, caveats, or context that doesn't fit above.

---

**Template Version:** 1.0.0
**Last Updated:** November 9, 2025
