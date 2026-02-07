# POL-016: Security Awareness and Training Policy

**Document Control**

| Attribute | Value |
|-----------|-------|
| Document ID | POL-016 |
| Version | 1.0 |
| Classification | Internal |
| Owner | Chief Information Security Officer (CISO) |
| Approved By | VP of People Operations |
| Effective Date | 2026-02-06 |
| Last Updated | 2026-02-06 |
| Next Review | 2026-08-06 |
| Policy Tier | Tier 4 - Operational |

---

## 1. Purpose

This policy establishes the requirements for security awareness and training programs at GreenLang to ensure all personnel understand their security responsibilities and possess the knowledge and skills necessary to protect GreenLang assets, customer data, and systems. Effective security awareness is a critical component of our defense-in-depth strategy and regulatory compliance posture.

---

## 2. Scope

This policy applies to:

- All GreenLang employees (full-time, part-time, temporary)
- Contractors and consultants with access to GreenLang systems
- Third-party personnel working on-site or with GreenLang data
- Interns and volunteers
- Executive leadership and board members

---

## 3. Policy Statement

GreenLang is committed to fostering a security-conscious culture where all personnel understand the importance of information security and their role in protecting organizational assets. Security awareness and training shall be mandatory, role-appropriate, and continuously improved based on emerging threats and organizational needs.

---

## 4. Mandatory Training Requirements

### 4.1 Security Awareness Training (All Personnel)

| Requirement | Detail |
|-------------|--------|
| Audience | All employees, contractors, interns |
| Frequency | Annual (by employment anniversary date) |
| Duration | 60-90 minutes |
| Format | Interactive e-learning with knowledge checks |
| Passing Score | 80% minimum |

**Core Topics:**
- Information security fundamentals and CIA triad
- Password security and authentication best practices
- Phishing and social engineering recognition
- Data classification and handling requirements
- Acceptable use of company systems
- Incident reporting procedures
- Physical security awareness
- Remote work security practices
- GDPR and privacy awareness
- GreenLang-specific security policies

### 4.2 Secure Coding Training (Development Personnel)

| Requirement | Detail |
|-------------|--------|
| Audience | Software developers, DevOps engineers, QA engineers |
| Frequency | Annual |
| Duration | 4-8 hours (modular format) |
| Format | Interactive labs and hands-on exercises |
| Passing Score | 80% minimum |

**Core Topics:**
- OWASP Top 10 vulnerabilities and mitigations
- Secure coding practices for Python, JavaScript, TypeScript
- Input validation and output encoding
- Authentication and session management
- Cryptography implementation
- Secure API development
- Container and Kubernetes security
- Infrastructure as Code security
- Security testing tools and techniques
- Code review for security

### 4.3 Incident Response Training (Responders)

| Requirement | Detail |
|-------------|--------|
| Audience | Incident Response Team, Security Team, On-Call Engineers |
| Frequency | Quarterly |
| Duration | 2-4 hours per session |
| Format | Tabletop exercises and simulations |
| Assessment | Scenario-based evaluation |

**Core Topics:**
- Incident classification and prioritization
- Containment and eradication procedures
- Evidence collection and chain of custody
- Communication protocols
- Regulatory notification requirements
- Post-incident review processes
- Forensic analysis fundamentals
- Coordination with legal and PR teams

### 4.4 Compliance Training (Role-Specific)

| Requirement | Detail |
|-------------|--------|
| Audience | Finance, Legal, HR, Customer Success, Sales |
| Frequency | Annual |
| Duration | 2-3 hours |
| Format | Role-specific e-learning modules |
| Passing Score | 80% minimum |

**Topics by Role:**
- **Finance:** SOX compliance, financial data handling, fraud awareness
- **Legal:** Contract security requirements, DPA management, regulatory landscape
- **HR:** Personnel security, background checks, termination procedures
- **Customer Success:** Customer data handling, privacy requests, breach notification
- **Sales:** NDA requirements, demo environment security, prospect data handling

---

## 5. Training Frequency and Timing

### 5.1 New Hire Training

| Training Type | Timeline |
|---------------|----------|
| Security Awareness | Within 30 calendar days of start date |
| Role-Specific Training | Within 30 calendar days of start date |
| Secure Coding (if applicable) | Within 45 calendar days of start date |
| Compliance Training (if applicable) | Within 30 calendar days of start date |

New hires shall not receive access to production systems or customer data until mandatory security awareness training is completed.

### 5.2 Annual Refresher Training

All personnel must complete annual refresher training by their employment anniversary date. A 30-day grace period is permitted, after which non-compliance escalation procedures apply.

### 5.3 Role Change Training

| Trigger | Requirement |
|---------|-------------|
| Promotion to people manager | Leadership security responsibilities (14 days) |
| Transfer to development | Secure coding training (14 days) |
| Access to customer data | Data handling training (7 days) |
| Privileged access granted | Privileged access training (7 days) |
| Incident response role | IR training (14 days) |

---

## 6. Role-Based Training Matrix

| Role | Security Awareness | Secure Coding | IR Training | Compliance | Privileged Access |
|------|-------------------|---------------|-------------|------------|-------------------|
| Executive Leadership | Required | - | Awareness | Required | - |
| Engineering Manager | Required | Required | Awareness | - | As Needed |
| Software Developer | Required | Required | - | - | As Needed |
| DevOps/SRE Engineer | Required | Required | Required | - | Required |
| Security Engineer | Required | Required | Required | Required | Required |
| Customer Success | Required | - | Awareness | Required | - |
| Sales | Required | - | - | Required | - |
| HR | Required | - | - | Required | - |
| Finance | Required | - | - | Required | As Needed |
| Legal | Required | - | Awareness | Required | - |
| All Contractors | Required | As Needed | - | As Needed | As Needed |

---

## 7. Security Champions Program

### 7.1 Program Overview

Security Champions are nominated employees who serve as security advocates within their teams, providing peer guidance and promoting security best practices.

### 7.2 Selection Criteria

- Minimum 6 months tenure at GreenLang
- Demonstrated interest in security
- Strong communication skills
- Respected by peers
- Manager endorsement
- Commitment to 2-4 hours per week for security activities

### 7.3 Champion Responsibilities

- Serve as first point of contact for security questions within their team
- Promote security awareness and best practices
- Participate in security policy reviews
- Assist with security incident triage
- Provide feedback on security tools and processes
- Attend monthly Security Champions meetings
- Complete advanced security training

### 7.4 Champion Training Requirements

| Training | Frequency | Duration |
|----------|-----------|----------|
| Advanced Security Fundamentals | Upon appointment | 8 hours |
| Threat Landscape Updates | Monthly | 1 hour |
| Security Tool Training | Quarterly | 2 hours |
| Leadership and Influence | Annual | 4 hours |

### 7.5 Recognition and Incentives

- Security Champion certification and badge
- Priority access to security conferences and training
- Recognition in company communications
- Input into security roadmap decisions
- Professional development opportunities

---

## 8. Phishing Simulation Program

### 8.1 Program Requirements

| Attribute | Requirement |
|-----------|-------------|
| Frequency | Monthly campaigns |
| Coverage | All employees with email access |
| Target Click Rate | Less than 5% organization-wide |
| Reporting Target | Greater than 80% report rate |

### 8.2 Simulation Types

- **Credential Harvesting:** Fake login pages mimicking internal systems
- **Malicious Attachments:** Simulated malware documents
- **Business Email Compromise:** Executive impersonation attempts
- **Spear Phishing:** Targeted attacks using OSINT
- **Smishing/Vishing:** Text and voice-based simulations (quarterly)

### 8.3 Remedial Training

| Click Behavior | Action |
|----------------|--------|
| First click in 12 months | Immediate just-in-time training (5 minutes) |
| Second click in 12 months | Extended remedial training (30 minutes) + manager notification |
| Third click in 12 months | One-on-one coaching with Security team + performance documentation |
| Fourth click in 12 months | Escalation to HR + access review |

### 8.4 Metrics and Reporting

- Monthly phishing metrics reported to Security Council
- Department-level click rates shared with department heads
- Quarterly trend analysis and improvement recommendations
- Annual benchmarking against industry standards

---

## 9. Training Completion Tracking

### 9.1 Learning Management System (LMS)

All training completion shall be tracked in the corporate LMS with the following data:
- Employee identifier and department
- Training module completed
- Completion date and time
- Assessment score
- Certificate of completion

### 9.2 Compliance Reporting

| Report | Frequency | Recipients |
|--------|-----------|------------|
| Completion Dashboard | Real-time | Managers, HR |
| Overdue Training Report | Weekly | Managers, HR, CISO |
| Compliance Summary | Monthly | Executive Team |
| Audit-Ready Report | Quarterly | Compliance, Auditors |

### 9.3 Manager Accountability

Managers are responsible for:
- Ensuring team members complete required training on time
- Following up on overdue training
- Incorporating security training into onboarding plans
- Addressing repeated non-compliance

---

## 10. Competency Assessment

### 10.1 Assessment Standards

| Standard | Requirement |
|----------|-------------|
| Passing Score | 80% minimum for all assessments |
| Retake Policy | Two additional attempts within 7 days |
| Proctoring | Not required for standard training |
| Time Limit | None (untimed assessments) |

### 10.2 Assessment Methods

- Multiple choice knowledge checks
- Scenario-based questions
- Interactive simulations
- Practical exercises (secure coding)
- Tabletop exercise performance (IR training)

### 10.3 Failure Remediation

If an employee fails to achieve passing score after three attempts:
1. Manager notified immediately
2. One-on-one review session with Security team scheduled
3. Additional study materials provided
4. Fourth attempt permitted after review session
5. Continued failure escalated to HR for performance management

---

## 11. Non-Compliance Consequences

### 11.1 Escalation Timeline

| Overdue Period | Action |
|----------------|--------|
| 1-7 days | Automated reminder emails |
| 8-14 days | Manager notification + calendar block for training |
| 15-30 days | CISO notification + access restrictions may apply |
| 31+ days | HR escalation + system access suspension |

### 11.2 Access Restrictions

Employees with significantly overdue training (31+ days) may have:
- VPN access suspended
- Production system access revoked
- Customer data access removed
- Access restored upon training completion

### 11.3 Performance Impact

Repeated non-compliance with training requirements shall be documented and may impact:
- Performance reviews
- Eligibility for promotion
- Bonus consideration
- Continued employment (extreme cases)

---

## 12. Continuous Learning Program

### 12.1 Security Newsletters

- **Frequency:** Bi-weekly
- **Content:** Current threats, tips, policy reminders, recognition
- **Distribution:** All employees via email and Slack

### 12.2 Security Alerts

- **Trigger:** Emerging threats, active campaigns, policy updates
- **Format:** Concise, actionable guidance
- **Distribution:** All employees or targeted groups

### 12.3 Lunch and Learn Sessions

- **Frequency:** Monthly
- **Duration:** 30-45 minutes
- **Topics:** Deep dives on security topics, guest speakers, demos
- **Participation:** Voluntary, recorded for async viewing

### 12.4 Security Awareness Month

- **Timing:** October (Cybersecurity Awareness Month)
- **Activities:** Special training, competitions, guest speakers, swag
- **Goals:** Reinforce security culture, recognize champions

---

## 13. Roles and Responsibilities

| Role | Responsibilities |
|------|------------------|
| **CISO** | Program oversight, content approval, executive reporting |
| **Security Team** | Training development, delivery, phishing simulations, metrics |
| **HR/People Ops** | LMS administration, compliance tracking, escalation support |
| **Managers** | Team compliance, follow-up on overdue training, performance management |
| **Employees** | Complete required training on time, apply knowledge, report incidents |
| **Security Champions** | Peer support, awareness promotion, feedback collection |

---

## 14. Exceptions

Exceptions to training requirements must be:
- Requested in writing with business justification
- Approved by the CISO
- Time-limited (maximum 90 days)
- Documented in the exception register
- Reviewed for continued applicability

Exceptions shall not be granted for foundational security awareness training.

---

## 15. Related Documents

| Document ID | Document Name |
|-------------|---------------|
| POL-001 | Information Security Policy |
| POL-006 | Incident Response Policy |
| POL-010 | SDLC Security Policy |
| POL-017 | Privacy Policy |
| STD-001 | Secure Coding Standards |
| PRO-016 | New Hire Security Onboarding Procedure |

---

## 16. Definitions

| Term | Definition |
|------|------------|
| **LMS** | Learning Management System - platform for delivering and tracking training |
| **Phishing Simulation** | Authorized simulated phishing attacks to test employee awareness |
| **Security Champion** | Employee designated to promote security within their team |
| **Just-in-Time Training** | Brief, immediate training delivered after a learning moment |
| **Click Rate** | Percentage of employees who click on simulated phishing links |

---

## 17. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | Security Team | Initial policy creation |

---

*This policy is reviewed semi-annually. Questions should be directed to security@greenlang.io.*
