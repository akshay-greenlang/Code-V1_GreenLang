# POL-007: Change Management Policy

## Document Control

| Field | Value |
|-------|-------|
| Policy ID | POL-007 |
| Version | 1.0 |
| Effective Date | 2026-02-06 |
| Last Review | 2026-02-06 |
| Next Review | 2027-02-06 |
| Owner | Chief Technology Officer (CTO) |
| Approver | Chief Information Security Officer (CISO) |
| Classification | Internal |

---

## 1. Purpose

This Change Management Policy establishes a structured approach for requesting, evaluating, approving, implementing, and reviewing changes to GreenLang's information technology systems, infrastructure, and applications. The policy ensures that changes are managed in a controlled manner to minimize service disruption, reduce security risks, and maintain system stability.

Effective change management:
- Minimizes unplanned outages and service disruptions
- Ensures changes are properly tested before implementation
- Maintains audit trails for compliance and forensics
- Reduces security vulnerabilities from improper configurations
- Supports rapid recovery through documented rollback procedures
- Enables continuous improvement through post-implementation reviews

---

## 2. Scope

### 2.1 Applicability

This policy applies to:
- All employees, contractors, and third parties making changes to GreenLang systems
- All IT systems, applications, infrastructure, and services
- Changes in production, staging, and development environments
- Both automated (CI/CD) and manual changes

### 2.2 Covered Changes

This policy covers changes to:
- Production applications and services
- Database schemas, configurations, and data migrations
- Infrastructure (servers, networks, storage, cloud resources)
- Security configurations (firewalls, IAM, certificates)
- Monitoring and alerting systems
- Third-party integrations and APIs
- Configuration management (Terraform, Kubernetes manifests)
- Documentation affecting operational procedures

### 2.3 Exclusions

The following are generally excluded from formal change management:
- Development branch work prior to merge request
- Personal development environment configurations
- Minor documentation updates (typos, formatting)
- Test data in non-production environments

---

## 3. Policy Statement

### 3.1 Change Management Principles

All changes to GreenLang systems shall:
- Be formally requested, documented, and tracked
- Undergo risk assessment appropriate to their category
- Receive proper approval before implementation
- Be tested in non-production environments when applicable
- Have documented rollback procedures
- Be implemented within approved change windows
- Be reviewed post-implementation for effectiveness

### 3.2 Change Categories

#### 3.2.1 Standard Changes (Pre-Approved)

**Definition:** Low-risk, routine changes that follow a documented, pre-approved procedure.

**Characteristics:**
- Performed frequently with predictable outcomes
- Low risk of negative impact
- Well-documented procedure exists
- Requires no individual approval

**Examples:**
- Password resets and account unlocks
- SSL certificate renewals (automated)
- Routine patching within approved patch windows
- Minor configuration updates to non-production
- Pre-approved dependency updates (minor versions)
- Log rotation and cleanup tasks

**Process:**
1. Verify change matches Standard Change catalog
2. Follow documented procedure
3. Document completion in change log
4. No CAB approval required

#### 3.2.2 Normal Changes (CAB Review Required)

**Definition:** Changes that require review and approval by the Change Advisory Board (CAB) due to moderate risk or impact.

**Characteristics:**
- Moderate risk or impact on systems/users
- Requires planning and coordination
- May affect multiple teams or services
- Needs formal testing and validation

**Examples:**
- New feature deployments to production
- Database schema migrations
- Infrastructure scaling or modifications
- New service deployments
- Security configuration changes
- Integration with third-party services
- Major dependency upgrades (major versions)

**Process:**
1. Submit Change Request Form (RFC)
2. Complete risk assessment
3. Develop test and rollback plans
4. Present to CAB for review
5. Implement during approved window
6. Complete post-implementation review

#### 3.2.3 Emergency Changes (Expedited Approval)

**Definition:** Changes required to resolve incidents or prevent imminent failures that cannot wait for normal approval cycles.

**Characteristics:**
- Required to restore service or prevent failure
- Cannot wait for scheduled CAB meeting
- Higher risk due to compressed timeline
- Requires immediate post-approval review

**Examples:**
- Security patches for actively exploited vulnerabilities
- Hotfixes for production outages
- Emergency rollbacks after failed deployments
- Critical infrastructure repairs
- Data recovery operations

**Process:**
1. Obtain verbal approval from CISO and CTO (or delegates)
2. Document change details as thoroughly as time permits
3. Implement change with available safeguards
4. Submit formal RFC within 24 hours
5. Present to CAB at next meeting for post-approval
6. Complete incident post-mortem if applicable

### 3.3 Change Request Procedures

#### 3.3.1 Request Form Requirements

All Normal and Emergency change requests must include:

| Field | Description | Required For |
|-------|-------------|--------------|
| Change Title | Brief description of the change | All |
| Requestor | Person initiating the request | All |
| Change Owner | Person responsible for implementation | All |
| Change Category | Standard, Normal, or Emergency | All |
| Description | Detailed explanation of what will change | All |
| Business Justification | Why this change is needed | Normal, Emergency |
| Technical Details | Systems, configurations, code affected | Normal |
| Affected Services | Services and dependencies impacted | All |
| Stakeholders | Teams and individuals affected | Normal |
| Implementation Plan | Step-by-step implementation procedure | Normal |
| Test Plan | How the change will be validated | Normal |
| Rollback Plan | Steps to reverse the change | All |
| Implementation Window | Requested date/time for change | Normal |
| Estimated Duration | Expected time to complete | All |
| Risk Assessment | Impact and likelihood evaluation | Normal |
| Approval Status | Current approval state | All |

#### 3.3.2 Business Justification

Business justification must address:
- Problem or opportunity being addressed
- Expected benefits (quantified where possible)
- Cost of not implementing the change
- Alignment with business objectives
- Customer impact (positive or negative)

#### 3.3.3 Technical Details

Technical documentation must include:
- Systems and components affected
- Configuration changes (before and after)
- Code changes (PRs, commits, deployments)
- Infrastructure modifications
- Database changes (migrations, scripts)
- Dependencies and prerequisites
- Integration points

### 3.4 Risk Assessment Requirements

#### 3.4.1 Impact Analysis

Assess impact across dimensions:

| Dimension | Low (1) | Medium (2) | High (3) |
|-----------|---------|------------|----------|
| **Users Affected** | <100 | 100-1000 | >1000 |
| **Services Affected** | 1 | 2-5 | >5 |
| **Data at Risk** | Public | Internal | Confidential/Restricted |
| **Revenue Impact** | None | Indirect | Direct |
| **Recovery Time** | <1 hour | 1-4 hours | >4 hours |

**Impact Score:** Sum of all dimensions (5-15)
- Low: 5-7
- Medium: 8-11
- High: 12-15

#### 3.4.2 Likelihood Assessment

Evaluate likelihood of negative outcome:

| Factor | Low (1) | Medium (2) | High (3) |
|--------|---------|------------|----------|
| **Change Complexity** | Simple | Moderate | Complex |
| **Testing Coverage** | Full | Partial | None |
| **Team Experience** | High | Medium | Low |
| **Rollback Feasibility** | Easy | Moderate | Difficult |
| **Change Frequency** | Routine | Occasional | First time |

**Likelihood Score:** Sum of all factors (5-15)
- Low: 5-7
- Medium: 8-11
- High: 12-15

#### 3.4.3 Risk Matrix

| | Likelihood Low | Likelihood Medium | Likelihood High |
|---------|----------------|-------------------|-----------------|
| **Impact High** | Medium Risk | High Risk | Critical Risk |
| **Impact Medium** | Low Risk | Medium Risk | High Risk |
| **Impact Low** | Low Risk | Low Risk | Medium Risk |

#### 3.4.4 Rollback Plan Requirements

All changes must have documented rollback procedures:

1. **Rollback Triggers:** Conditions that initiate rollback
   - Service degradation beyond threshold
   - Failed health checks post-deployment
   - Customer-impacting errors
   - Security vulnerabilities discovered

2. **Rollback Steps:** Specific actions to revert
   - Code/configuration reversion steps
   - Database rollback scripts (if applicable)
   - Infrastructure restoration commands
   - Communication procedures

3. **Rollback Verification:** How to confirm successful rollback
   - Health check endpoints
   - Functional test execution
   - Monitoring dashboard confirmation
   - User validation (if applicable)

4. **Rollback Owner:** Person responsible for rollback decision and execution

#### 3.4.5 Testing Requirements

Testing requirements based on risk level:

| Risk Level | Testing Required |
|------------|------------------|
| **Low** | Unit tests passing, smoke test in staging |
| **Medium** | Integration tests, staging validation, peer review |
| **High** | Full regression, load testing, security review, staging soak |
| **Critical** | All above + external review, phased rollout |

### 3.5 Approval Workflows

#### 3.5.1 Standard Changes

- **Approval:** Pre-approved via Standard Change catalog
- **Approvers:** None required (self-service)
- **Documentation:** Change log entry required
- **Timeline:** Immediate execution allowed

#### 3.5.2 Normal Changes

- **Approval:** Change Advisory Board (CAB)
- **Approvers:** CAB Chair + Technical Lead + impacted team leads
- **Documentation:** Full RFC required
- **Timeline:** Submit RFC 3 business days before CAB meeting
- **CAB Meeting:** Weekly (Tuesdays 2:00 PM UTC)

CAB review criteria:
- Risk assessment completeness and accuracy
- Test plan adequacy
- Rollback plan feasibility
- Implementation window appropriateness
- Resource availability confirmed
- Stakeholder notification complete

#### 3.5.3 Emergency Changes

- **Approval:** CISO and CTO (or designated delegates)
- **Approvers:** Both must approve verbally or via Slack/email
- **Documentation:** Abbreviated RFC (full RFC within 24 hours)
- **Timeline:** Immediate execution upon dual approval
- **Post-Approval:** Present at next CAB meeting

Emergency approval delegates:
- CISO delegates: Security Director, Security Operations Manager
- CTO delegates: VP Engineering, Platform Lead

### 3.6 Change Windows

#### 3.6.1 Standard Maintenance Windows

| Window | Time (UTC) | Duration | Suitable For |
|--------|------------|----------|--------------|
| **Primary** | Sunday 02:00-06:00 | 4 hours | High-risk infrastructure changes |
| **Secondary** | Wednesday 02:00-04:00 | 2 hours | Medium-risk changes |
| **Daily** | Any day 02:00-03:00 | 1 hour | Low-risk, quick changes |

#### 3.6.2 Change Freeze Periods

No changes allowed (except emergencies) during:
- Major customer events (announced 2 weeks in advance)
- End of quarter reporting (last 3 business days)
- Public holidays (US, EU major holidays)
- Annual change freeze: December 15 - January 5

#### 3.6.3 Off-Window Changes

Changes outside standard windows require:
- Additional justification for timing
- CISO or CTO approval
- Enhanced monitoring during implementation
- On-call escalation path confirmed

### 3.7 Testing and Validation Requirements

#### 3.7.1 Pre-Implementation Testing

| Environment | Purpose | Requirements |
|-------------|---------|--------------|
| **Development** | Initial validation | Unit tests, local integration |
| **Staging** | Pre-production testing | Full integration, performance |
| **Pre-production** | Final validation | Production-like data, load testing |

#### 3.7.2 Validation Criteria

Changes must pass:
- All automated tests (unit, integration, e2e)
- Security scanning (SAST, SCA, secrets detection)
- Performance benchmarks (no degradation >10%)
- Health check endpoints responding
- Monitoring and alerting functional

#### 3.7.3 Sign-off Requirements

| Change Risk | Sign-off Required |
|-------------|-------------------|
| **Low** | Change Owner |
| **Medium** | Change Owner + Tech Lead |
| **High** | Change Owner + Tech Lead + Security |
| **Critical** | All above + CTO |

### 3.8 Rollback Procedures and Triggers

#### 3.8.1 Automatic Rollback Triggers

Automated rollback initiated when:
- Error rate exceeds 5% for 5 minutes
- Latency P99 exceeds SLO by 50%
- Health checks fail for 3 consecutive attempts
- Critical security alert triggered
- Data integrity check fails

#### 3.8.2 Manual Rollback Triggers

Manual rollback decision when:
- Customer-reported critical issues
- Security vulnerability discovered post-deployment
- Unexpected resource consumption
- Integration failures with third parties
- Business stakeholder escalation

#### 3.8.3 Rollback Authority

| Timeframe | Rollback Authority |
|-----------|-------------------|
| **During change window** | Change Owner |
| **Within 4 hours of completion** | Change Owner or On-Call Engineer |
| **After 4 hours** | Incident Commander or Technical Lead |

### 3.9 Change Documentation Requirements

#### 3.9.1 Required Documentation

All changes must maintain:
- RFC record in change management system
- Approval chain with timestamps
- Implementation notes and actual steps taken
- Test results and validation evidence
- Rollback execution (if performed)
- Post-implementation review notes

#### 3.9.2 Audit Trail

Change records must be:
- Immutable once approved
- Retained for 7 years (compliance requirement)
- Searchable by date, system, owner, status
- Available for audit upon request

### 3.10 Post-Implementation Review

#### 3.10.1 Review Timeline

- **Successful changes:** Review within 5 business days
- **Failed changes:** Review within 2 business days
- **Emergency changes:** Review at next CAB meeting

#### 3.10.2 Review Contents

Post-implementation reviews assess:
- Were objectives achieved?
- Did implementation follow the plan?
- Were there unexpected issues?
- Was rollback necessary? Why?
- What lessons were learned?
- Should this become a Standard Change?
- Process improvement recommendations

#### 3.10.3 Review Documentation

Document in the RFC record:
- Actual vs. planned implementation time
- Issues encountered and resolution
- Stakeholder feedback
- Metrics before and after change
- Recommendations for future changes

### 3.11 Emergency Change Post-Approval

#### 3.11.1 Post-Hoc Documentation

Within 24 hours of emergency change:
1. Complete full RFC with all required fields
2. Document verbal approval chain
3. Provide detailed implementation steps taken
4. Document any deviations from standard procedure
5. Complete preliminary risk assessment

#### 3.11.2 CAB Review

At next CAB meeting:
1. Present emergency change details
2. Review appropriateness of emergency designation
3. Assess if change could have waited
4. Identify process improvements
5. Formally approve or flag for follow-up

#### 3.11.3 Pattern Analysis

Emergency changes are analyzed quarterly:
- Identify recurring emergency patterns
- Convert appropriate emergencies to Standard Changes
- Address root causes requiring frequent emergencies
- Update procedures to reduce future emergencies

---

## 4. Roles and Responsibilities

### 4.1 Change Requestor

- Submit complete and accurate RFCs
- Provide business justification
- Coordinate with stakeholders
- Respond to CAB questions
- Withdraw or modify requests as needed

### 4.2 Change Owner

- Overall responsibility for change success
- Develop implementation and rollback plans
- Coordinate resources and scheduling
- Execute or oversee implementation
- Make rollback decisions
- Complete post-implementation review

### 4.3 Change Advisory Board (CAB)

- Review Normal change requests
- Assess risk and approve/reject changes
- Ensure adequate testing and rollback plans
- Monitor change success metrics
- Review emergency changes post-hoc
- Recommend process improvements

**CAB Composition:**
- CAB Chair (rotating): Senior Engineering Manager
- Security Representative: Security Operations
- Infrastructure Representative: Platform Engineering
- Application Representative: Development Lead
- Operations Representative: SRE Team Lead

### 4.4 CAB Chair

- Schedule and facilitate CAB meetings
- Ensure agenda is distributed in advance
- Document decisions and action items
- Escalate conflicts to CTO
- Report metrics to leadership

### 4.5 Technical Reviewers

- Assess technical feasibility and risk
- Review implementation and rollback plans
- Validate testing adequacy
- Provide recommendations to CAB
- Support troubleshooting if needed

### 4.6 Security Team

- Review security implications of changes
- Approve changes affecting security controls
- Validate security testing completion
- Provide emergency approval for security incidents
- Monitor for security-related change patterns

### 4.7 Operations/SRE Team

- Validate operational readiness
- Ensure monitoring and alerting coverage
- Provide implementation support
- Execute rollbacks as needed
- Maintain change management tooling

---

## 5. Procedures

### 5.1 Submitting a Change Request

1. Log into change management system (Jira/ServiceNow)
2. Create new Change Request using appropriate template
3. Complete all required fields
4. Attach supporting documentation
5. Submit for review
6. Address feedback and questions
7. Attend CAB meeting if required

### 5.2 CAB Meeting Process

1. Agenda published 24 hours in advance
2. Requestors present changes (5 minutes each)
3. CAB members ask clarifying questions
4. Security review findings presented
5. Vote on approval (majority required)
6. Document decisions and conditions
7. Publish meeting minutes within 24 hours

### 5.3 Implementing Changes

1. Verify approval status in system
2. Notify stakeholders of implementation start
3. Follow implementation plan step-by-step
4. Document actual steps and any deviations
5. Execute validation checks
6. Notify stakeholders of completion
7. Monitor for issues during stabilization period

### 5.4 Executing Rollback

1. Identify rollback trigger condition
2. Notify incident channel of rollback decision
3. Follow documented rollback steps
4. Validate system restored to previous state
5. Document rollback execution
6. Initiate incident review process
7. Update RFC with rollback details

---

## 6. Exceptions

### 6.1 Exception Criteria

Exceptions to standard change procedures may be granted for:
- Regulatory deadline requirements
- Customer contractual obligations
- Critical security vulnerabilities
- Business continuity requirements

### 6.2 Exception Process

1. Submit exception request to CAB Chair
2. Provide justification and risk assessment
3. Identify compensating controls
4. Obtain CTO approval for Normal change exceptions
5. Obtain CEO approval for change freeze exceptions
6. Document exception and rationale

### 6.3 Non-Negotiable Requirements

These elements have no exceptions:
- Documentation of all changes (may be post-hoc for emergencies)
- Security review for security-impacting changes
- Rollback capability for production changes
- Post-implementation review for failed changes

---

## 7. Enforcement

### 7.1 Compliance Monitoring

Change management compliance is monitored through:
- Automated deployment pipeline enforcement
- CAB review of unauthorized changes
- Audit log analysis
- Metrics dashboards and reporting

### 7.2 Non-Compliance Consequences

Unauthorized changes may result in:
- Change reverted immediately
- Incident report filed
- Verbal warning for first offense
- Written warning for repeat offenses
- Termination for willful violations
- Access revocation during investigation

### 7.3 Metrics and Reporting

Track and report monthly:
- Change volume by category
- Change success rate
- Emergency change frequency
- Mean time to implement
- Rollback frequency
- Post-implementation review completion

---

## 8. Related Documents

| Document | Description |
|----------|-------------|
| POL-001: Information Security Policy | Master security policy |
| POL-010: SDLC Security Policy | Development and deployment security |
| PRD-INFRA-007: CI/CD Pipelines | Automated deployment infrastructure |
| Incident Response Playbook | Handling change-related incidents |
| Disaster Recovery Runbook | Major outage recovery procedures |
| Release Management Guide | Software release procedures |

---

## 9. Definitions

| Term | Definition |
|------|------------|
| **CAB** | Change Advisory Board - group that reviews and approves changes |
| **RFC** | Request for Change - formal change proposal document |
| **Standard Change** | Pre-approved, low-risk, routine change |
| **Normal Change** | Change requiring CAB review and approval |
| **Emergency Change** | Urgent change bypassing normal approval |
| **Change Window** | Scheduled time period for implementing changes |
| **Change Freeze** | Period when non-emergency changes are prohibited |
| **Rollback** | Reverting a change to the previous state |
| **Post-Implementation Review** | Assessment of change success after completion |
| **Change Owner** | Person responsible for change implementation |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | CTO | Initial policy release |

---

## Appendix A: Standard Change Catalog

| ID | Standard Change | Procedure | Frequency |
|----|----------------|-----------|-----------|
| SC-001 | Password reset | IT-PROC-001 | As needed |
| SC-002 | SSL certificate renewal | AUTO-CERT-001 | Automated |
| SC-003 | Minor dependency update | CI-DEP-001 | Weekly |
| SC-004 | Log rotation | OPS-LOG-001 | Daily |
| SC-005 | Test data refresh | QA-DATA-001 | Weekly |
| SC-006 | Monitoring threshold adjustment | OPS-MON-001 | As needed |
| SC-007 | Documentation update | DOC-UPD-001 | As needed |
| SC-008 | Feature flag toggle | FF-TOG-001 | As needed |

---

## Appendix B: Change Request Form Template

```
=== CHANGE REQUEST FORM ===

CHANGE ID: [Auto-generated]
DATE SUBMITTED: [YYYY-MM-DD]

REQUESTOR INFORMATION
- Name:
- Team:
- Email:

CHANGE DETAILS
- Title:
- Category: [ ] Standard [ ] Normal [ ] Emergency
- Description:
- Business Justification:
- Technical Details:

AFFECTED SYSTEMS
- Services:
- Dependencies:
- Teams:

SCHEDULING
- Requested Window:
- Estimated Duration:
- Downtime Required: [ ] Yes [ ] No

RISK ASSESSMENT
- Impact Score (5-15):
- Likelihood Score (5-15):
- Overall Risk Level:

PLANS
- Implementation Plan: [Attach document]
- Test Plan: [Attach document]
- Rollback Plan: [Attach document]

APPROVALS
- Technical Lead: [ ] Approved [ ] Rejected
- Security: [ ] Approved [ ] Rejected
- CAB: [ ] Approved [ ] Rejected

IMPLEMENTATION
- Start Time:
- End Time:
- Status: [ ] Success [ ] Failed [ ] Rolled Back

POST-IMPLEMENTATION
- Review Completed: [ ] Yes [ ] No
- Lessons Learned:
```

---

**Document Classification: Internal**
**Policy Owner: Chief Technology Officer**
**Copyright 2026 GreenLang Climate OS. All Rights Reserved.**
