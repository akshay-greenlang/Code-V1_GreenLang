# POL-018: Incident Communication Policy

**Document Control**

| Attribute | Value |
|-----------|-------|
| Document ID | POL-018 |
| Version | 1.0 |
| Classification | Internal |
| Owner | Chief Information Security Officer (CISO) |
| Approved By | Chief Executive Officer (CEO) |
| Effective Date | 2026-02-06 |
| Last Updated | 2026-02-06 |
| Next Review | 2026-08-06 |
| Policy Tier | Tier 4 - Operational |

---

## 1. Purpose

This policy establishes the requirements and procedures for communicating during security incidents, service disruptions, and data breaches at GreenLang. Effective communication during incidents is critical for maintaining stakeholder trust, meeting regulatory obligations, and ensuring coordinated response efforts.

This policy ensures that internal teams, customers, regulators, and the public receive timely, accurate, and appropriate information during incidents while protecting sensitive investigation details and avoiding premature or harmful disclosures.

---

## 2. Scope

This policy applies to communications related to:

- **Security Incidents:** Unauthorized access, data breaches, malware, DDoS attacks
- **Service Disruptions:** Platform outages, degraded performance, feature unavailability
- **Data Breaches:** Unauthorized access to or disclosure of personal or confidential data
- **Compliance Events:** Regulatory violations, audit findings, mandatory disclosures
- **Physical Security Events:** Facility incidents affecting operations

This policy covers communications with:
- Internal stakeholders (employees, executives, board)
- External stakeholders (customers, partners, vendors)
- Regulators and supervisory authorities
- Media and the general public

---

## 3. Policy Statement

GreenLang is committed to transparent, timely, and accurate communication during incidents while protecting ongoing investigations and avoiding speculation. All incident communications must be coordinated through designated channels and spokespersons to ensure consistency and appropriateness.

---

## 4. Communication Triggers by Severity

### 4.1 Priority Classification

| Priority | Description | Communication Timeline |
|----------|-------------|----------------------|
| **P0 - Critical** | Complete service outage, active breach affecting customer data, ransomware | Immediate (within 1 hour) |
| **P1 - High** | Significant service degradation, suspected breach, vulnerability exploitation | Same business day (within 8 hours) |
| **P2 - Moderate** | Limited service impact, contained incident, non-sensitive data exposure | Within 24 hours |
| **P3 - Low** | Minor issues, resolved incidents, security improvements | Weekly summary or as appropriate |

### 4.2 P0 Critical Incident Triggers

Immediate communication (within 1 hour) is required when:
- Complete platform unavailability affecting all customers
- Confirmed unauthorized access to customer data
- Active ransomware or destructive malware
- Regulatory breach requiring immediate notification
- Physical security event threatening personnel safety
- Third-party compromise affecting GreenLang data

### 4.3 P1 High Incident Triggers

Same-day communication is required when:
- Significant service degradation affecting multiple customers
- Suspected but unconfirmed data breach
- Active vulnerability exploitation (non-critical data)
- Major third-party service disruption
- Security control failure

### 4.4 P2 Moderate Incident Triggers

Communication within 24 hours for:
- Limited service impact (single feature or region)
- Contained incident with no data exposure
- Successfully mitigated attack
- Non-sensitive internal data exposure

### 4.5 P3 Low Incident Triggers

Weekly summary or discretionary communication for:
- Minor service hiccups quickly resolved
- Blocked attack attempts
- Security improvements and patches
- Training and awareness activities

---

## 5. Internal Communication

### 5.1 Incident Channel Creation

Upon incident declaration:

| Action | Responsibility | Timeline |
|--------|----------------|----------|
| Create dedicated Slack channel | Incident Commander | Within 15 minutes |
| Channel naming convention | #incident-YYYY-MM-DD-brief-description | Standardized |
| Add core responders | Incident Commander | Immediate |
| Pin incident details | Incident Commander | Within 30 minutes |
| Set channel purpose | Incident Commander | Immediate |

**Channel Rules:**
- Discussions limited to incident response activities
- No speculation or unverified information
- All key decisions documented in channel
- Status updates posted at regular intervals
- Channel archived after post-incident review

### 5.2 Stakeholder Notification Matrix

| Stakeholder | P0 Critical | P1 High | P2 Moderate | P3 Low |
|-------------|-------------|---------|-------------|--------|
| CISO | Immediate | Immediate | Within 4 hours | Weekly |
| CEO | Within 1 hour | Within 4 hours | Daily summary | Weekly |
| CTO | Immediate | Immediate | Within 4 hours | Weekly |
| Legal Counsel | Within 1 hour | Within 4 hours | Within 24 hours | As needed |
| CFO | Within 2 hours | Within 8 hours | Daily summary | Weekly |
| Board of Directors | Within 4 hours | Within 24 hours | N/A | Quarterly |
| Department Heads | Within 2 hours | Within 8 hours | Within 24 hours | Weekly |
| All Employees | As appropriate | As appropriate | N/A | Newsletter |

### 5.3 Executive Briefing Triggers

Executive briefing calls are convened when:
- P0 incident declared
- P1 incident lasting more than 4 hours
- Customer data breach confirmed
- Regulatory notification required
- Media inquiry received
- Material business impact anticipated
- Board notification required

**Briefing Format:**
1. Incident summary (2 minutes)
2. Current status and impact (3 minutes)
3. Response actions taken (3 minutes)
4. Customer/regulatory communication status (2 minutes)
5. Resource needs (2 minutes)
6. Questions and decisions (10 minutes)
7. Next update time (1 minute)

### 5.4 Internal Status Updates

| Priority | Update Frequency | Method |
|----------|------------------|--------|
| P0 | Every 30 minutes | Incident channel + email |
| P1 | Every 2 hours | Incident channel + email |
| P2 | Every 4 hours | Incident channel |
| P3 | Daily | Incident channel |

---

## 6. Customer Notification

### 6.1 Notification Triggers

| Event Type | Notification Requirement | Timeline |
|------------|-------------------------|----------|
| Data breach affecting customer data | Mandatory | Within 72 hours |
| Service outage (platform-wide) | Required | Within 1 hour |
| Service degradation (significant) | Required | Within 4 hours |
| Planned maintenance | Required | 72 hours advance |
| Security vulnerability (customer action required) | Required | Within 24 hours |
| Security vulnerability (no action required) | Discretionary | After remediation |

### 6.2 Data Breach Customer Notification

When customer personal data is breached:

**Timeline:** Without undue delay, and no later than 72 hours after confirmation

**Content Requirements:**
- Clear description of the breach
- Types of data involved
- Approximate date of the breach
- Actions GreenLang has taken
- Recommendations for customers
- Contact information for questions
- Reference to detailed information (if applicable)

**Notification Methods:**
- Direct email to account administrators
- In-app notification banner
- Status page update
- Customer success manager outreach (enterprise)

### 6.3 Service Impact Notifications

**Proactive Communication:**
- Status page updated within 15 minutes of detection
- Email notification for significant outages
- In-app banner for platform-wide issues
- Social media update for widespread awareness

**Resolution Communication:**
- Status page updated upon resolution
- Root cause summary (non-technical)
- Preventive measures taken
- Apology and acknowledgment (if appropriate)
- SLA credit information (if applicable)

### 6.4 Status Page Management

| Status | Criteria | Visibility |
|--------|----------|------------|
| **Operational** | All systems functioning normally | Public |
| **Degraded Performance** | Slow response times, partial functionality | Public |
| **Partial Outage** | Some features or regions unavailable | Public |
| **Major Outage** | Platform-wide unavailability | Public |
| **Under Maintenance** | Planned maintenance window | Public |

**Status Page URL:** status.greenlang.io

**Update Requirements:**
- Initial update within 15 minutes of incident
- Progress updates every 30 minutes (P0) or hourly (P1/P2)
- Resolution update with summary
- Post-incident report link (for significant incidents)

---

## 7. Regulatory Notification

### 7.1 GDPR Data Protection Authority Notification

| Requirement | Detail |
|-------------|--------|
| Trigger | Personal data breach likely to result in risk to individuals |
| Timeline | Within 72 hours of becoming aware |
| Authority | Lead Supervisory Authority (based on main establishment) |
| Method | DPA online portal or designated form |

**Notification Content (Article 33):**
- Nature of the breach
- Categories and approximate number of data subjects
- Categories and approximate number of records
- Name and contact details of DPO
- Likely consequences
- Measures taken or proposed to address the breach

**Delayed Notification:**
If notification cannot be provided within 72 hours:
- Provide available information without delay
- Submit reasons for delay
- Provide additional information in phases

### 7.2 State Privacy Law Notifications

| Jurisdiction | Threshold | Timeline | Authority |
|--------------|-----------|----------|-----------|
| California (CCPA/CPRA) | CA residents affected | "Without unreasonable delay" | CA Attorney General |
| New York (SHIELD) | NY residents affected | "Without unreasonable delay" | NY Attorney General |
| Colorado (CPA) | CO residents affected | 30 days | CO Attorney General |
| Virginia (CDPA) | VA residents affected | "Without unreasonable delay" | VA Attorney General |

### 7.3 SEC Material Incident Notification

For publicly traded companies or when GreenLang reaches public company status:

| Requirement | Detail |
|-------------|--------|
| Trigger | Material cybersecurity incident |
| Timeline | Within 4 business days of materiality determination |
| Form | Form 8-K Item 1.05 |
| Content | Nature, scope, timing, material impact |

### 7.4 Regulatory Notification Workflow

1. **Assessment:** Legal and Security teams assess notification requirements
2. **Preparation:** Draft notification content using approved templates
3. **Review:** Legal review and approval
4. **Executive Sign-off:** CEO or CISO approval
5. **Submission:** DPO or Legal submits notification
6. **Documentation:** Record submission details and confirmations
7. **Follow-up:** Respond to regulatory inquiries

---

## 8. Media Communication

### 8.1 Media Response Authority

| Role | Authority |
|------|-----------|
| **CEO** | Primary spokesperson for material incidents |
| **CISO** | Technical spokesperson (with CEO approval) |
| **VP Communications** | Media relations coordination |
| **Legal Counsel** | Statement review and approval |

**All other employees are not authorized to speak to media regarding incidents.**

### 8.2 Media Inquiry Handling

| Step | Action | Responsibility |
|------|--------|----------------|
| 1 | Receive inquiry | Any employee |
| 2 | Record details (outlet, reporter, deadline, questions) | Recipient |
| 3 | Forward to VP Communications immediately | Recipient |
| 4 | Acknowledge receipt to reporter | VP Communications |
| 5 | Coordinate response with Legal, Security, Executive | VP Communications |
| 6 | Draft response using approved messaging | VP Communications |
| 7 | Legal and executive approval | Legal, CEO/CISO |
| 8 | Deliver response | Authorized spokesperson |
| 9 | Document exchange | VP Communications |

### 8.3 Talking Points Preparation

For significant incidents, prepare:
- Key messages (3-5 bullets)
- Background information
- Q&A document (anticipated questions)
- Statements for different scenarios
- "No comment" boundaries
- Redirect phrases

### 8.4 Media Response Principles

**Do:**
- Acknowledge the situation factually
- Express concern for affected parties
- Describe response actions taken
- Commit to updates as appropriate
- Direct to official channels for information

**Do Not:**
- Speculate on causes or attribution
- Provide technical details that could aid attackers
- Blame third parties without verification
- Make promises that cannot be kept
- Comment on ongoing investigations
- Discuss financial impact prematurely

### 8.5 Approval Workflow for Media Statements

| Incident Severity | Approval Required |
|-------------------|-------------------|
| P0 Critical | CEO + Legal |
| P1 High | CISO + Legal |
| P2 Moderate | VP Communications + Legal |
| P3 Low | VP Communications |

---

## 9. Social Media Handling

### 9.1 Monitoring

During incidents, monitor:
- @greenlang mentions on Twitter/X
- LinkedIn company page comments
- Reddit (r/sustainability, r/climate, r/technology)
- Hacker News
- Customer community forums
- Industry news sites

### 9.2 Response Protocol

| Situation | Response |
|-----------|----------|
| Customer reporting issue | Acknowledge, direct to status page/support |
| Speculation about breach | Do not engage; prepare statement if needed |
| Inaccurate information | Correct factually, link to official statement |
| Media inquiry via social | Direct to media@greenlang.io |
| Hostile/inflammatory posts | Do not engage; document for legal |

### 9.3 Social Media Response Guidelines

**Respond to:**
- Direct customer inquiries about service status
- Requests for information (with approved messaging)
- Opportunities to provide official statements

**Do Not Respond to:**
- Speculation about causes or attackers
- Inflammatory or hostile posts
- Questions requiring non-public information
- Competitive commentary

### 9.4 Coordinated Response

- All social media responses during incidents require VP Communications approval
- Use pre-approved messaging templates
- Consistent messaging across all platforms
- Log all public interactions

---

## 10. Communication Templates

### 10.1 Internal Incident Notification

```
Subject: [PRIORITY] Incident Declared - [Brief Description]

Incident ID: INC-YYYY-NNNN
Priority: P0/P1/P2/P3
Incident Commander: [Name]
Status: Active/Investigating/Contained/Resolved

SUMMARY:
[2-3 sentence description of the incident]

IMPACT:
- Customer Impact: [Description]
- Systems Affected: [List]
- Data Affected: [Yes/No/Under Investigation]

CURRENT STATUS:
[Current response activities and status]

NEXT UPDATE:
[Time of next scheduled update]

INCIDENT CHANNEL:
#incident-YYYY-MM-DD-description

---
Do not forward this message externally.
```

### 10.2 Customer Service Disruption Notification

```
Subject: Service Alert - [Service Name] [Status]

Dear [Customer Name],

We are currently experiencing [brief description of issue] affecting
[service/feature]. Our team is actively working to resolve this.

IMPACT:
[Description of what customers may experience]

WHAT WE'RE DOING:
[Brief description of response actions]

ESTIMATED RESOLUTION:
[Time estimate if known, or "We will provide updates as available"]

UPDATES:
Please monitor status.greenlang.io for real-time updates.

We apologize for any inconvenience and appreciate your patience.

GreenLang Support Team
support@greenlang.io
```

### 10.3 Data Breach Notification to Customers

```
Subject: Important Security Notice from GreenLang

Dear [Customer Name],

We are writing to inform you of a security incident that may have
affected your data.

WHAT HAPPENED:
On [date], we discovered [brief, factual description of the incident].

WHAT INFORMATION WAS INVOLVED:
[List of data types affected]

WHAT WE ARE DOING:
- [Action 1]
- [Action 2]
- [Action 3]

WHAT YOU CAN DO:
- [Recommendation 1]
- [Recommendation 2]
- [Recommendation 3]

FOR MORE INFORMATION:
If you have questions, please contact our dedicated response team:
- Email: security-incident@greenlang.io
- Phone: [Dedicated hotline]

We sincerely apologize for this incident and are committed to
protecting your information.

[Name]
[Title]
GreenLang
```

### 10.4 Regulatory Notification Template

```
DATA BREACH NOTIFICATION TO SUPERVISORY AUTHORITY
Pursuant to GDPR Article 33

1. CONTROLLER DETAILS
Organization: GreenLang Inc.
Address: [Address]
DPO Contact: dpo@greenlang.io

2. NATURE OF THE BREACH
[Description of the breach, including categories of data]

3. DATA SUBJECTS AFFECTED
Approximate number: [Number]
Categories: [e.g., customers, employees, partners]

4. PERSONAL DATA RECORDS AFFECTED
Approximate number: [Number]
Categories: [e.g., contact information, account data]

5. LIKELY CONSEQUENCES
[Assessment of potential impact on data subjects]

6. MEASURES TAKEN
[Actions taken to address the breach and mitigate effects]

7. ADDITIONAL INFORMATION
[Any relevant additional details]

Submitted by: [Name, Title]
Date: [Date]
Reference: [Internal incident ID]
```

### 10.5 Media Statement Template

```
STATEMENT FROM GREENLANG
[Date]

GreenLang [is aware of / has identified / is responding to]
[brief factual description].

[What we know / What happened - factual, verified information only]

[Our response - actions taken]

[Commitment to customers and stakeholders]

We will provide updates as more information becomes available.
For the latest information, please visit [status page / blog /
newsroom URL].

Media Contact:
[Name]
media@greenlang.io
[Phone]
```

---

## 11. Post-Incident Communication

### 11.1 Internal Lessons Learned

Following incident closure:
- Post-incident review conducted within 5 business days
- Findings documented in incident record
- Lessons learned shared with relevant teams
- Process improvements tracked to completion

### 11.2 Customer Post-Incident Summary

For significant incidents, provide customers:
- Incident summary and timeline
- Root cause (appropriate level of detail)
- Remediation actions taken
- Preventive measures implemented
- SLA credit information (if applicable)

### 11.3 Public Post-Incident Report

For major incidents affecting many customers:
- Published on GreenLang blog within 2 weeks
- Technical summary appropriate for public audience
- Actions taken to prevent recurrence
- Acknowledgment and apology (if appropriate)

---

## 12. Roles and Responsibilities

| Role | Responsibilities |
|------|------------------|
| **Incident Commander** | Coordinates all communication activities during incident |
| **CEO** | Primary external spokesperson, approves material communications |
| **CISO** | Technical communication approval, regulatory assessment |
| **Legal Counsel** | Regulatory notification, statement approval, legal risk assessment |
| **DPO** | GDPR notification requirements, data subject communication |
| **VP Communications** | Media relations, social media, public messaging |
| **Customer Success** | Customer notification, account manager communication |
| **Engineering** | Technical status updates, status page management |

---

## 13. Exceptions

Exceptions to communication timelines or procedures require:
- CISO and Legal approval
- Documentation of exception rationale
- Alternative communication plan
- Review in post-incident analysis

Exceptions may be appropriate when:
- Communication would compromise ongoing investigation
- Law enforcement requests communication delay
- Premature disclosure would increase risk

---

## 14. Related Documents

| Document ID | Document Name |
|-------------|---------------|
| POL-006 | Incident Response Policy |
| POL-017 | Privacy Policy |
| PRO-006 | Incident Response Procedure |
| RUN-001 | Incident Response Runbook |
| TMP-018 | Communication Templates Library |

---

## 15. Definitions

| Term | Definition |
|------|------------|
| **Incident Commander** | Individual responsible for managing incident response |
| **Data Breach** | Unauthorized access to or disclosure of personal data |
| **Material Incident** | Incident with significant business or legal impact |
| **DPA** | Data Protection Authority (supervisory authority under GDPR) |
| **Status Page** | Public-facing service status dashboard |
| **Talking Points** | Pre-approved messaging for media or public communication |

---

## 16. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | Security Team | Initial policy creation |

---

*This policy is reviewed semi-annually. Questions should be directed to security@greenlang.io.*
