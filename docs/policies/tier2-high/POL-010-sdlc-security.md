# POL-010: Software Development Lifecycle (SDLC) Security Policy

## Document Control

| Field | Value |
|-------|-------|
| Policy ID | POL-010 |
| Version | 1.0 |
| Effective Date | 2026-02-06 |
| Last Review | 2026-02-06 |
| Next Review | 2027-02-06 |
| Owner | Chief Technology Officer (CTO) |
| Approver | Chief Information Security Officer (CISO) |
| Classification | Internal |

---

## 1. Purpose

This Software Development Lifecycle (SDLC) Security Policy establishes security requirements and practices that must be integrated throughout the software development process at GreenLang. Security must be a fundamental consideration from requirements through deployment, not an afterthought.

This policy ensures that:
- Security is embedded in every phase of software development
- Vulnerabilities are identified and remediated early when cost is lowest
- Code meets secure coding standards before production deployment
- Dependencies are managed to minimize supply chain risk
- Security testing is automated and comprehensive
- Releases are signed and verified for integrity
- Technical debt is tracked and prioritized appropriately

---

## 2. Scope

### 2.1 Applicability

This policy applies to:
- All GreenLang software development teams
- Contractors and third parties developing GreenLang software
- All internally developed applications and services
- All code deployed to production environments
- Open source contributions representing GreenLang

### 2.2 Covered Activities

This policy covers:
- Application development (Python, TypeScript, Go)
- Infrastructure as Code (Terraform, Kubernetes manifests)
- Configuration management
- API development and integration
- Database schema design and migrations
- Machine learning model development
- Mobile application development
- DevOps and CI/CD pipeline development

### 2.3 Development Environments

This policy applies to code destined for:
- Production environments
- Staging environments
- Shared development environments
- Customer-facing demo environments

---

## 3. Policy Statement

### 3.1 Secure Coding Standards

#### 3.1.1 OWASP Top 10 Awareness

All developers must understand and code defensively against OWASP Top 10 vulnerabilities:

| Rank | Vulnerability | Required Defense |
|------|--------------|------------------|
| A01 | Broken Access Control | Deny by default, validate all access server-side |
| A02 | Cryptographic Failures | Use approved algorithms, never custom crypto |
| A03 | Injection | Parameterized queries, input validation, output encoding |
| A04 | Insecure Design | Threat modeling, secure design patterns |
| A05 | Security Misconfiguration | Hardened defaults, no default credentials |
| A06 | Vulnerable Components | SCA scanning, dependency updates |
| A07 | Identification & Auth Failures | MFA, strong passwords, secure session management |
| A08 | Software & Data Integrity | Signed releases, verified dependencies |
| A09 | Logging & Monitoring Failures | Comprehensive audit logging, alerting |
| A10 | Server-Side Request Forgery | Input validation, allowlists, network segmentation |

#### 3.1.2 Language-Specific Guidelines

**Python:**
- Use type hints for all public functions
- Avoid `eval()`, `exec()`, and `pickle` with untrusted data
- Use `secrets` module for random generation, not `random`
- Parameterize all database queries (SQLAlchemy ORM preferred)
- Use `defusedxml` for XML parsing
- Follow PEP 8 style guidelines

**TypeScript/JavaScript:**
- Enable strict TypeScript mode (`strict: true`)
- Use prepared statements for database queries
- Sanitize HTML output (DOMPurify for client-side)
- Avoid `eval()`, `new Function()`, `innerHTML` with user input
- Use `===` instead of `==` for comparisons
- Use ESLint security plugins

**Go:**
- Use standard library crypto packages
- Avoid unsafe pointer operations
- Use context with timeouts for all I/O
- Parameterize SQL queries
- Follow effective Go guidelines

**Infrastructure as Code (Terraform/Kubernetes):**
- No hardcoded secrets in configuration
- Use least-privilege IAM policies
- Enable encryption at rest and in transit
- Define resource limits (CPU, memory)
- Use network policies for segmentation

#### 3.1.3 Input Validation

All user input must be validated:
- Validate on the server side (never trust client validation)
- Use allowlists over denylists when possible
- Validate data type, length, format, and range
- Reject invalid input, do not attempt to sanitize
- Log validation failures for monitoring

#### 3.1.4 Output Encoding

All output must be properly encoded:
- HTML encode for web page output
- JavaScript encode for script contexts
- URL encode for URL parameters
- SQL parameterization for database queries
- JSON encoding for API responses
- Use context-appropriate encoding libraries

### 3.2 Security Requirements Phase

#### 3.2.1 Threat Modeling for New Features

Threat modeling is required for:
- New applications or services
- New features handling sensitive data
- Significant architectural changes
- External-facing API additions
- Authentication or authorization changes
- Third-party integrations

Threat modeling process:
1. Document architecture and data flows
2. Identify assets and trust boundaries
3. Enumerate threats using STRIDE methodology
4. Assess risk (likelihood x impact)
5. Define mitigations for high/critical risks
6. Document residual risks and acceptance

#### 3.2.2 Security Acceptance Criteria

User stories involving sensitive functionality must include security acceptance criteria:

```
As a [user]
I want to [action]
So that [benefit]

Security Acceptance Criteria:
- [ ] Input validated against expected schema
- [ ] Authorization checked before data access
- [ ] Sensitive data encrypted in transit/at rest
- [ ] Action logged for audit trail
- [ ] Error messages do not leak sensitive info
- [ ] Rate limiting applied
```

#### 3.2.3 Privacy by Design

For features processing personal data:
- Data minimization: Collect only required data
- Purpose limitation: Use data only for stated purpose
- Retention limits: Define and enforce retention periods
- Access controls: Restrict access to need-to-know
- Encryption: Protect data at rest and in transit
- Consent: Implement consent management where required

### 3.3 Code Review Standards

#### 3.3.1 All Code Reviewed Before Merge

- Every code change requires at least one approving review
- Authors cannot approve their own code
- Reviews must be from qualified reviewers (not AI-only)
- Review comments must be addressed before merge
- Approvals expire if code changes substantially

#### 3.3.2 Security-Focused Review Requirements

Enhanced security review required for:

| Change Type | Review Requirement |
|-------------|-------------------|
| Authentication/authorization logic | Security team review |
| Cryptographic operations | Security team review |
| Code handling Confidential/Restricted data | Security team review |
| New external dependencies | Security team review |
| Infrastructure changes to production | Platform + Security review |
| API changes (public or internal) | API owner + standard review |
| Database schema changes | DBA + standard review |

#### 3.3.3 Code Review Security Checklist

Reviewers must verify:

**Authentication & Authorization:**
- [ ] All endpoints require authentication (unless intentionally public)
- [ ] Authorization checks performed server-side
- [ ] Principle of least privilege applied

**Input/Output:**
- [ ] All input validated before use
- [ ] Output properly encoded for context
- [ ] No command injection vectors

**Data Protection:**
- [ ] Sensitive data not logged
- [ ] PII handled per privacy requirements
- [ ] Secrets not hardcoded

**Error Handling:**
- [ ] Errors handled gracefully
- [ ] Error messages do not reveal sensitive information
- [ ] Exceptions logged appropriately

**Dependencies:**
- [ ] New dependencies justified and vetted
- [ ] No known critical/high vulnerabilities

### 3.4 Security Testing Requirements

Reference: PRD-SEC-007: Security Scanning Pipeline

#### 3.4.1 Static Application Security Testing (SAST)

**Requirement:** SAST scan on every commit to main branches

| Tool | Scope | Blocking |
|------|-------|----------|
| Bandit | Python code | Critical/High block merge |
| Semgrep | All languages | Critical/High block merge |
| TFSec | Terraform | Critical/High block merge |
| Checkov | IaC | Critical/High block merge |

**Configuration:**
- Scans run automatically via pre-commit hooks and CI
- Results uploaded to GitHub Security tab (SARIF)
- False positives documented and suppressed with justification
- New findings must be triaged within 24 hours

#### 3.4.2 Dynamic Application Security Testing (DAST)

**Requirement:** DAST scan on staging deployment

| Type | Frequency | Scope |
|------|-----------|-------|
| Baseline scan | Every staging deployment | Critical paths, authentication |
| Full scan | Nightly | All endpoints |
| API scan | Every staging deployment | All API endpoints |

**Configuration:**
- OWASP ZAP configured with GreenLang scan policy
- Authenticated scans using test credentials
- Results triaged by Security team
- Critical/High findings block production deployment

#### 3.4.3 Software Composition Analysis (SCA)

**Requirement:** SCA scan on every build

| Tool | Scope | Database |
|------|-------|----------|
| Trivy | All dependencies, containers | NVD, GitHub Advisory |
| Snyk | Dependencies | Snyk vulnerability DB |
| pip-audit | Python | PyPI Advisory DB |
| npm audit | JavaScript | npm Advisory DB |

**Configuration:**
- Scans run on PR and main branch builds
- SBOM generated for every release (CycloneDX format)
- License compliance verified (see 3.5.1)
- Vulnerability SLAs enforced (see 3.5.3)

#### 3.4.4 Penetration Testing

**Requirement:** Annual penetration testing minimum

| Type | Frequency | Scope |
|------|-----------|-------|
| External penetration test | Annual | Production, externally accessible |
| Internal penetration test | Annual | Internal services, network |
| Web application test | Semi-annual | Customer-facing applications |
| Red team exercise | Biennial | Full scope, including social |

**Process:**
- Engage approved third-party firm
- Provide test scope and rules of engagement
- Findings reported to Security team
- Critical/High findings remediated within SLA
- Retesting to verify remediation

### 3.5 Dependency Management

#### 3.5.1 Approved Sources Only

Dependencies must come from approved sources:

| Language | Approved Sources |
|----------|------------------|
| Python | PyPI (pinned versions), internal mirror |
| JavaScript | npm registry (pinned), internal mirror |
| Go | Go modules proxy, internal mirror |
| Containers | ECR, approved Docker Hub images |
| Terraform | Terraform Registry, internal modules |

Prohibited:
- Direct GitHub/GitLab dependency links in production
- Unvetted or unmaintained packages
- Packages with restrictive licenses (see below)

#### 3.5.2 License Compliance

Approved licenses for production use:
- MIT, Apache 2.0, BSD (2/3-clause)
- ISC, Unlicense, CC0
- MPL 2.0 (with legal review)

Requires legal review:
- LGPL (dynamic linking only)
- GPL (usually prohibited in proprietary)
- AGPL (prohibited in SaaS without legal)
- Commons Clause, SSPL (prohibited)

All dependencies scanned for license compliance before use.

#### 3.5.3 Vulnerability Scanning and Update SLAs

| Severity | Remediation SLA | Action |
|----------|-----------------|--------|
| Critical (CVSS 9.0+) | 24 hours | Immediate patch, emergency change if needed |
| High (CVSS 7.0-8.9) | 7 days | Prioritize remediation |
| Medium (CVSS 4.0-6.9) | 30 days | Plan remediation |
| Low (CVSS 0.1-3.9) | 90 days | Address in normal cycle |

**Automated Processes:**
- Dependabot/Renovate creates PRs for updates
- Security patches auto-merged for passing tests (minor/patch only)
- Major version updates require manual review
- Weekly dependency review meetings

### 3.6 Security Gates

#### 3.6.1 No Critical/High Vulnerabilities

Deployment blocked if:
- SAST finds Critical or High severity issues
- SCA finds Critical or High severity vulnerabilities (not suppressed)
- DAST finds Critical or High severity issues in staging
- Container scan finds Critical or High issues

Gate enforcement:
- CI/CD pipeline fails on gate violation
- Security team notified of blocked deployments
- Override requires CISO or Security Director approval
- All overrides logged and time-limited

#### 3.6.2 Security Review Sign-Off

Security team sign-off required for:
- New production services
- Major architecture changes
- Changes to authentication/authorization
- New external integrations
- Changes to data handling for sensitive data
- Production infrastructure changes

Sign-off process:
1. Developer requests review via Jira
2. Security reviews design/code
3. Security provides findings/recommendations
4. Developer addresses findings
5. Security signs off (comment in PR/ticket)
6. Merge/deploy proceeds

### 3.7 Deployment Security

#### 3.7.1 Signed Artifacts

All production artifacts must be signed:

| Artifact Type | Signing Method |
|---------------|---------------|
| Container images | Cosign (Sigstore) |
| Helm charts | GPG signature |
| Python packages | GPG signature (internal PyPI) |
| SBOM | Cosign attestation |

**Process:**
- CI/CD signs artifacts during build
- Private keys stored in KMS/Vault
- Public keys distributed to deployment systems
- Signature verification before deployment

#### 3.7.2 Image Verification

Before deployment, verify:
- Image signature is valid
- Image from approved registry
- Image scanned and meets security gates
- Image provenance matches build system
- SBOM attestation present

Unsigned or unverified images rejected automatically.

#### 3.7.3 Deployment Integrity

Production deployments must:
- Use immutable infrastructure (no runtime changes)
- Deploy from CI/CD only (no manual deploys)
- Use GitOps with version-controlled manifests
- Include automated health checks
- Support automated rollback

### 3.8 Security Debt Management

#### 3.8.1 Backlog Tracking

Security debt tracked in central backlog:
- Security findings added as tickets (Jira)
- Categorized by type (vulnerability, tech debt, compliance)
- Severity assigned per standard scale
- SLA assigned based on severity
- Age tracked for reporting

#### 3.8.2 Prioritization Criteria

Security debt prioritized by:

| Factor | Weight |
|--------|--------|
| Exploitability (actively exploited, PoC, theoretical) | 30% |
| Impact (data exposure, availability, integrity) | 30% |
| Exposure (internet-facing, internal, air-gapped) | 20% |
| Data sensitivity (Restricted, Confidential, Internal, Public) | 20% |

Priority calculation guides sprint allocation:
- Critical: Immediate remediation (sprint 0)
- High: Current sprint
- Medium: Next 2 sprints
- Low: Backlog (quarterly review)

#### 3.8.3 Security Debt Metrics

Track and report monthly:
- Total security debt items (by severity)
- Average age of open items
- Items opened vs. closed
- Overdue items (past SLA)
- Debt trend over time

Targets:
- Zero Critical items >24 hours
- Zero High items >7 days
- Debt count decreasing quarter-over-quarter
- <5% items overdue

#### 3.8.4 Debt Reduction Sprints

Quarterly security debt reduction:
- Minimum 10% sprint capacity allocated to security debt
- Security team identifies priority items
- Engineering addresses during sprint
- Progress reported to leadership

---

## 4. Roles and Responsibilities

### 4.1 Developers

- Write secure code following standards
- Complete security training annually
- Address security findings in owned code
- Participate in code reviews
- Report security concerns to Security team
- Keep dependencies updated

### 4.2 Tech Leads

- Ensure team follows secure coding standards
- Conduct security-focused code reviews
- Triage security findings for team
- Allocate time for security remediation
- Participate in threat modeling
- Escalate security concerns

### 4.3 Security Champions

- Advocate for security within teams
- First point of contact for security questions
- Participate in Security Champion program
- Share security knowledge with team
- Assist with security reviews
- Attend monthly Security Champion meetings

### 4.4 Information Security Team

- Define security standards and requirements
- Conduct security reviews and threat modeling
- Manage security scanning tools
- Triage and prioritize vulnerabilities
- Provide security training
- Coordinate penetration testing
- Maintain security gates

### 4.5 Platform/DevOps Team

- Implement and maintain CI/CD security gates
- Manage artifact signing infrastructure
- Configure security scanning in pipelines
- Maintain container security baselines
- Manage secrets management integration
- Respond to pipeline security incidents

### 4.6 Product Management

- Include security requirements in planning
- Prioritize security-related work
- Support time allocation for security debt
- Consider security in feature decisions
- Communicate security features to customers

---

## 5. Procedures

### 5.1 Starting a New Project

1. Register project with Security team
2. Conduct initial threat modeling session
3. Define security requirements with Product
4. Configure CI/CD with security scanning
5. Set up dependency management (Dependabot/Renovate)
6. Complete Security team review before first release
7. Document architecture and security controls

### 5.2 Adding a New Dependency

1. Verify package is from approved source
2. Check license compatibility
3. Run security scan (SCA)
4. Review package reputation (downloads, maintenance, vulnerabilities)
5. Add with pinned version
6. Document justification in PR description
7. Request Security review if sensitive functionality

### 5.3 Responding to Vulnerability Alert

1. Acknowledge alert within 4 hours
2. Assess severity and applicability
3. Determine affected systems/versions
4. Identify remediation (patch, upgrade, workaround)
5. Apply remediation per SLA
6. Verify fix through testing
7. Close ticket with evidence
8. Conduct retrospective for Critical/High

### 5.4 Requesting Security Review

1. Create security review request (Jira)
2. Attach design documents or PRs
3. Identify areas of concern
4. Security team assigns reviewer
5. Review completed within 3 business days
6. Address findings and re-request if needed
7. Obtain sign-off before proceeding

---

## 6. Exceptions

### 6.1 Exception Criteria

Exceptions to security requirements may be granted for:
- Critical business deadline with documented risk acceptance
- Legacy system with compensating controls
- Third-party component limitations
- Research/experimental projects (non-production)

### 6.2 Exception Process

1. Submit exception request with justification
2. Identify compensating controls
3. Define exception scope and duration
4. Security assesses risk
5. Approval authority:
   - Low risk: Security Manager
   - Medium risk: CISO
   - High risk: CISO + CTO
6. Document exception and review date
7. Implement monitoring for exceptions

### 6.3 Non-Negotiable Requirements

No exceptions for:
- Credentials/secrets in code
- Known Critical vulnerabilities in production
- Unsigned artifacts in production
- Disabled security scanning
- Bypassing code review

---

## 7. Enforcement

### 7.1 Technical Enforcement

Policy enforced through:
- Pre-commit hooks blocking policy violations
- CI/CD gates blocking non-compliant code
- Branch protection requiring reviews
- Automated dependency updates
- Signature verification at deployment

### 7.2 Non-Compliance Consequences

Violations may result in:
- Deployment blocked until remediation
- Mandatory security training
- Performance impact for repeated violations
- Escalation to management
- Termination for willful violations

### 7.3 Metrics and Reporting

Track and report monthly:
- SAST/DAST finding counts (by severity)
- Mean time to remediate vulnerabilities
- Code review coverage percentage
- Security gate failure rate
- Dependency vulnerability count
- Security debt trend
- Penetration test findings status

---

## 8. Related Documents

| Document | Description |
|----------|-------------|
| POL-001: Information Security Policy | Master security policy |
| POL-007: Change Management Policy | Change control procedures |
| POL-009: Password and Authentication Policy | Credential management |
| PRD-SEC-006: Secrets Management | Vault integration |
| PRD-SEC-007: Security Scanning Pipeline | Scanning infrastructure |
| PRD-INFRA-007: CI/CD Pipelines | Pipeline configuration |
| Secure Coding Guide | Detailed coding standards |
| Threat Modeling Guide | Threat modeling procedures |

---

## 9. Definitions

| Term | Definition |
|------|------------|
| **SAST** | Static Application Security Testing - analyzing source code |
| **DAST** | Dynamic Application Security Testing - testing running applications |
| **SCA** | Software Composition Analysis - scanning dependencies |
| **SBOM** | Software Bill of Materials - inventory of components |
| **OWASP** | Open Web Application Security Project |
| **CVE** | Common Vulnerabilities and Exposures identifier |
| **CVSS** | Common Vulnerability Scoring System |
| **STRIDE** | Threat modeling framework (Spoofing, Tampering, Repudiation, Info Disclosure, DoS, Elevation) |
| **Security Champion** | Developer advocate for security within their team |
| **Security Gate** | Automated check that blocks non-compliant code |
| **Technical Debt** | Deferred work that accumulates maintenance cost |
| **Cosign** | Tool for signing container images (Sigstore project) |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | CTO | Initial policy release |

---

## Appendix A: Security Review Request Template

```
=== SECURITY REVIEW REQUEST ===

Project/Service:
Requestor:
Date:
Target Review Date:

REVIEW TYPE
[ ] New service/application
[ ] Major feature addition
[ ] Architecture change
[ ] External integration
[ ] Data handling change
[ ] Other: _______________

DESCRIPTION
Brief description of what needs review:

DOCUMENTATION
[ ] Design document attached
[ ] Architecture diagram attached
[ ] PR link (if code review):
[ ] Threat model attached (if existing):

DATA CLASSIFICATION
Highest data classification involved:
[ ] Public [ ] Internal [ ] Confidential [ ] Restricted

SPECIFIC CONCERNS
Areas where you want Security to focus:

TIMELINE
When do you need this review completed?
Is there flexibility?

REVIEWER ASSIGNMENT (Security team fills in)
Assigned to:
Target completion:
```

---

## Appendix B: Code Review Security Checklist

```markdown
## Security Review Checklist

### Authentication & Authorization
- [ ] Endpoints require appropriate authentication
- [ ] Authorization verified for all operations
- [ ] No privilege escalation paths
- [ ] Session management is secure

### Input Validation
- [ ] All input validated server-side
- [ ] Validation uses allowlists where possible
- [ ] No unsafe deserialization
- [ ] File uploads validated and sandboxed

### Output Encoding
- [ ] HTML output encoded
- [ ] JSON responses properly formatted
- [ ] No XSS vulnerabilities
- [ ] Error messages don't leak info

### Data Protection
- [ ] PII handled per policy
- [ ] Sensitive data encrypted
- [ ] No secrets in code
- [ ] Logs don't contain sensitive data

### Cryptography
- [ ] Uses approved algorithms
- [ ] Keys properly managed
- [ ] Random numbers from secure source
- [ ] No custom crypto implementations

### Dependencies
- [ ] New deps from approved sources
- [ ] License compatible
- [ ] No known vulnerabilities
- [ ] Version pinned

### Infrastructure
- [ ] Least privilege IAM
- [ ] Network segmentation appropriate
- [ ] Resource limits defined
- [ ] Secrets from Vault

Reviewer: _______________
Date: _______________
Result: [ ] Approved [ ] Changes Requested
```

---

## Appendix C: STRIDE Threat Modeling Quick Reference

| Threat | Description | Example | Mitigation |
|--------|-------------|---------|------------|
| **Spoofing** | Impersonating someone/something | Fake login page | Strong authentication |
| **Tampering** | Modifying data | SQL injection | Input validation, integrity checks |
| **Repudiation** | Denying actions | "I didn't do that" | Audit logging |
| **Information Disclosure** | Exposing sensitive data | Data breach | Encryption, access control |
| **Denial of Service** | Disrupting availability | DDoS attack | Rate limiting, redundancy |
| **Elevation of Privilege** | Gaining unauthorized access | Privilege escalation | Least privilege, authorization |

---

**Document Classification: Internal**
**Policy Owner: Chief Technology Officer**
**Copyright 2026 GreenLang Climate OS. All Rights Reserved.**
