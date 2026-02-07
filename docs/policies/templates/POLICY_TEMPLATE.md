# [Policy Name]

**Document Control**

| Attribute | Value |
|-----------|-------|
| Document ID | POL-NNN |
| Version | 1.0 |
| Classification | [Public / Internal / Confidential / Restricted] |
| Owner | [Title of Policy Owner] |
| Approved By | [Title of Approver] |
| Effective Date | YYYY-MM-DD |
| Review Date | YYYY-MM-DD |

---

## 1. Purpose

[State the purpose of this policy in 2-3 sentences. Explain why this policy exists and what it aims to achieve.]

**Example:**
> This policy establishes the requirements for controlling access to GreenLang information systems and data. It ensures that access is granted based on business need and the principle of least privilege, protecting sensitive information from unauthorized access.

---

## 2. Scope

### 2.1 Applicability

This policy applies to:
- [List of personnel covered - e.g., all employees, contractors, third parties]
- [List of systems covered - e.g., all production systems, cloud infrastructure]
- [List of data covered - e.g., customer data, financial records]

### 2.2 Exclusions

This policy does not apply to:
- [Any explicit exclusions, if applicable]

**Example:**
> This policy applies to all GreenLang employees, contractors, and third-party vendors who access company systems. It covers all production and development environments, including cloud infrastructure (AWS, Azure), SaaS applications, and on-premises systems.

---

## 3. Policy Statement

### 3.1 [First Major Requirement Area]

[State the policy requirement clearly and unambiguously. Use "shall" for mandatory requirements and "should" for recommendations.]

**3.1.1** [Specific requirement]

**3.1.2** [Specific requirement]

**3.1.3** [Specific requirement]

### 3.2 [Second Major Requirement Area]

**3.2.1** [Specific requirement]

**3.2.2** [Specific requirement]

### 3.3 [Third Major Requirement Area]

**3.3.1** [Specific requirement]

**3.3.2** [Specific requirement]

**Example:**
> ### 3.1 Access Provisioning
>
> **3.1.1** All access requests shall be submitted through the IT Service Management system and require manager approval before provisioning.
>
> **3.1.2** Access shall be granted based on the principle of least privilege, providing only the minimum permissions necessary for job functions.
>
> **3.1.3** Privileged access shall require additional approval from the system owner and security team.

---

## 4. Roles and Responsibilities

| Role | Responsibilities |
|------|------------------|
| **[Role 1 - e.g., CISO]** | [List key responsibilities for this role] |
| **[Role 2 - e.g., IT Operations]** | [List key responsibilities for this role] |
| **[Role 3 - e.g., Managers]** | [List key responsibilities for this role] |
| **[Role 4 - e.g., All Employees]** | [List key responsibilities for this role] |

**Example:**

| Role | Responsibilities |
|------|------------------|
| **CISO** | Overall accountability for access control policy; approve privileged access requests; review access violations |
| **IT Operations** | Provision and deprovision access; maintain access control systems; generate access reports |
| **Managers** | Approve access requests for direct reports; conduct quarterly access reviews; report access concerns |
| **All Employees** | Protect access credentials; report suspected unauthorized access; complete security training |

---

## 5. Procedures

### 5.1 [Procedure 1 Name]

**Objective:** [What this procedure accomplishes]

**Steps:**

1. [First step with specific instructions]
2. [Second step with specific instructions]
3. [Third step with specific instructions]
4. [Fourth step with specific instructions]

**Inputs:** [Required inputs or prerequisites]

**Outputs:** [Expected outputs or deliverables]

**Frequency:** [How often this procedure is performed]

### 5.2 [Procedure 2 Name]

**Objective:** [What this procedure accomplishes]

**Steps:**

1. [First step]
2. [Second step]
3. [Third step]

**Example:**
> ### 5.1 Access Request Procedure
>
> **Objective:** Provision appropriate system access for new or changing job roles.
>
> **Steps:**
>
> 1. Employee submits access request via ServiceNow, specifying systems and access level needed
> 2. Manager reviews request and approves/denies within 2 business days
> 3. For privileged access, Security Team reviews and approves/denies within 1 business day
> 4. IT Operations provisions approved access within 1 business day
> 5. System generates confirmation email to requestor and manager
>
> **Inputs:** Completed access request form, manager approval
>
> **Outputs:** Provisioned access, audit log entry
>
> **Frequency:** As needed (event-driven)

---

## 6. Compliance and Enforcement

### 6.1 Compliance Monitoring

[Describe how compliance with this policy will be monitored]

- [Monitoring method 1]
- [Monitoring method 2]
- [Audit frequency]

### 6.2 Non-Compliance

Violations of this policy may result in:

| Severity | Examples | Consequences |
|----------|----------|--------------|
| **Low** | [Minor violations] | Verbal warning, additional training |
| **Medium** | [Moderate violations] | Written warning, restricted access |
| **High** | [Serious violations] | Suspension, termination consideration |
| **Critical** | [Severe violations] | Immediate termination, legal action |

### 6.3 Reporting Violations

Suspected policy violations should be reported to:
- Security Team: security@greenlang.io
- Anonymous Hotline: [hotline number/URL]
- Direct Manager

---

## 7. Exceptions

### 7.1 Exception Process

Exceptions to this policy must be:

1. Documented using the Policy Exception Request form
2. Approved by the Policy Owner and [appropriate authority]
3. Time-limited (maximum [duration])
4. Reviewed [frequency] while active

### 7.2 Exception Criteria

Exceptions may be granted when:
- [Criterion 1 - e.g., technical limitation prevents compliance]
- [Criterion 2 - e.g., business necessity with compensating controls]
- [Criterion 3 - e.g., regulatory requirement conflict]

### 7.3 Compensating Controls

All exceptions must include documented compensating controls that:
- Address the risk created by the exception
- Are monitored for effectiveness
- Are documented in the exception request

---

## 8. Related Documents

| Document | Description |
|----------|-------------|
| [POL-XXX: Related Policy] | [Brief description of relationship] |
| [STD-XXX: Related Standard] | [Brief description of relationship] |
| [PROC-XXX: Related Procedure] | [Brief description of relationship] |
| [External Reference] | [Link or citation to external standard/regulation] |

**Example:**

| Document | Description |
|----------|-------------|
| POL-001: Information Security Policy | Parent policy establishing security framework |
| POL-004: Data Classification Policy | Defines data categories for access decisions |
| STD-003: Password Standard | Password requirements for authentication |
| ISO 27001:2022 A.9 | Access control requirements |

---

## 9. Definitions

| Term | Definition |
|------|------------|
| **[Term 1]** | [Clear definition of the term as used in this policy] |
| **[Term 2]** | [Clear definition of the term as used in this policy] |
| **[Term 3]** | [Clear definition of the term as used in this policy] |

**Example:**

| Term | Definition |
|------|------------|
| **Access Control** | The selective restriction of access to data and systems based on authorization |
| **Least Privilege** | The principle of providing only the minimum access necessary to perform job functions |
| **Privileged Access** | Administrative or elevated access that can modify system configurations or access sensitive data |
| **Provisioning** | The process of creating and configuring user accounts and access rights |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | YYYY-MM-DD | [Author Name] | Initial release |
| 1.1 | YYYY-MM-DD | [Author Name] | [Summary of changes] |
| 2.0 | YYYY-MM-DD | [Author Name] | [Summary of major revision] |

---

## Appendices

### Appendix A: [Appendix Title]

[Include supplementary information, forms, or detailed procedures that support the policy but would clutter the main body]

### Appendix B: [Appendix Title]

[Additional appendices as needed]

---

## Document Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Policy Owner | | | |
| Approver | | | |
| Legal Review (if required) | | | |

---

## Formatting Guidelines

### Writing Style

1. **Use Active Voice**: "Employees shall protect their credentials" not "Credentials shall be protected by employees"

2. **Be Specific**: "Review access quarterly" not "Review access regularly"

3. **Use Consistent Terminology**: Define terms and use them consistently throughout

4. **Use Shall/Should/May**:
   - **Shall** = Mandatory requirement
   - **Should** = Recommendation (best practice)
   - **May** = Optional/permissive

### Document Structure

1. **Headings**: Use hierarchical numbering (1, 1.1, 1.1.1)

2. **Tables**: Use for structured data, role matrices, reference lists

3. **Lists**: Use bullets for unordered items, numbers for sequential steps

4. **Cross-References**: Link to related documents using document IDs

### Formatting Conventions

| Element | Convention |
|---------|------------|
| Policy requirements | Bold key terms |
| Definitions | Italics on first use |
| Document references | Document ID in brackets [POL-001] |
| Dates | YYYY-MM-DD format |
| Times | 24-hour format with timezone |

---

*This template is confidential and intended for internal use only. Unauthorized distribution is prohibited.*

---

## Template Checklist

Before submitting a policy for review, ensure:

- [ ] All sections are completed (or marked N/A with justification)
- [ ] Document control table is filled in
- [ ] Scope is clearly defined
- [ ] All requirements use shall/should/may appropriately
- [ ] Roles and responsibilities are assigned
- [ ] Procedures are specific and actionable
- [ ] Related documents are listed and linked
- [ ] Definitions include all specialized terms
- [ ] Revision history is current
- [ ] Approval signatures are obtained (for final version)
