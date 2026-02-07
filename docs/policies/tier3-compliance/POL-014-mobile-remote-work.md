# POL-014: Mobile Device and Remote Work Policy

**Document Control**

| Attribute | Value |
|-----------|-------|
| Document ID | POL-014 |
| Version | 1.0 |
| Classification | Confidential |
| Policy Tier | Tier 3 - Compliance |
| Owner | Chief Information Security Officer (CISO) |
| Approved By | Director of IT |
| Effective Date | 2026-02-06 |
| Last Review | 2026-02-06 |
| Next Review | 2027-02-06 |

---

## 1. Purpose

This policy establishes requirements for the secure use of mobile devices and remote work arrangements at GreenLang. The purpose is to enable flexible work while protecting company information, maintaining productivity, and ensuring compliance with security and privacy requirements.

As a distributed technology company, GreenLang supports remote work as a core part of our operations. This policy defines the security controls required to protect company data when accessed outside traditional office environments, whether through mobile devices, home offices, or while traveling.

This policy supports compliance with SOC 2 Type II (endpoint security controls), ISO 27001:2022 (A.6.7 Teleworking, A.8.1 User endpoint devices), and GDPR requirements for data protection during remote processing.

---

## 2. Scope

This policy applies to:

- **Personnel**: All employees, contractors, and third parties who access GreenLang systems remotely or use mobile devices for work
- **Devices**: Company-issued laptops, smartphones, tablets; personal devices used for work (BYOD)
- **Locations**: Home offices, co-working spaces, public locations, travel destinations
- **Activities**: Remote access to company systems, email, collaboration tools, and data processing
- **Data**: All company data accessed, processed, or stored on mobile devices or during remote work

### 2.1 Out of Scope

- Office-based work (covered by POL-013 Physical Security Policy)
- Server and infrastructure management (covered by operational procedures)
- Customer devices and systems (governed by customer agreements)

---

## 3. Policy Statement

GreenLang enables secure remote work and mobile device usage to support business agility and employee flexibility. All remote access must occur through approved, secured channels, and all devices accessing company data must meet minimum security requirements. Employees are responsible for maintaining a secure remote work environment and reporting security incidents promptly.

### 3.1 Approved Mobile Devices

#### 3.1.1 Company-Issued Devices (Preferred)

Company-issued devices are the preferred method for accessing company data and systems:

| Device Type | Supported Platforms | Provisioning | Support |
|-------------|---------------------|--------------|---------|
| **Laptops** | macOS 13+, Windows 11 Pro/Enterprise | IT-provisioned, pre-configured | Full IT support |
| **Smartphones** | iOS 16+, Android 13+ | IT-provisioned or COPE | Full IT support |
| **Tablets** | iPadOS 16+, Android 13+ | IT-provisioned | Full IT support |

Company devices include:
- Pre-installed security software (endpoint protection, MDM)
- Full-disk encryption enabled
- Automatic security updates
- Remote wipe capability
- VPN client pre-configured

#### 3.1.2 Personal Devices (BYOD)

Personal devices may be used with approval and MDM enrollment:

| Device Type | Minimum Requirements | Enrollment Required |
|-------------|---------------------|---------------------|
| **Laptops** | macOS 12+, Windows 10 Pro, Linux (approved distros) | Endpoint agent required |
| **Smartphones** | iOS 15+, Android 12+ | MDM enrollment required |
| **Tablets** | iPadOS 15+, Android 12+ | MDM enrollment required |

BYOD devices require:
- Written acknowledgment of BYOD agreement
- MDM enrollment before accessing company data
- Compliance with minimum security requirements
- Understanding that company may remotely wipe work container

### 3.2 Mobile Device Management (MDM) Requirements

All devices accessing company data must comply with MDM policies:

#### 3.2.1 Required Security Controls

| Control | Company Device | BYOD Device |
|---------|----------------|-------------|
| **Screen Lock** | 6+ digit PIN or biometric | 6+ digit PIN or biometric |
| **Encryption** | Full disk encryption required | Device encryption required |
| **Remote Wipe** | Enabled (full device) | Enabled (work container) |
| **Automatic Updates** | Enforced (security updates) | Required within 7 days |
| **Jailbreak/Root Detection** | Blocked | Blocked |
| **App Installation** | Managed app store preferred | Work apps managed |
| **VPN** | Auto-connect for company resources | Required for company access |
| **Camera in Secure Areas** | May be disabled | May be disabled |

#### 3.2.2 MDM Enforcement Actions

| Compliance State | Action | Timeline |
|------------------|--------|----------|
| **Non-compliant: No passcode** | Block company data access | Immediate |
| **Non-compliant: Outdated OS** | Warning, then block | 7-day warning, then block |
| **Non-compliant: Jailbroken** | Block access, wipe work data | Immediate |
| **Lost/Stolen** | Remote wipe | Within 1 hour of report |
| **Terminated Employee** | Remote wipe work data | Same day |

#### 3.2.3 MDM Privacy Notice

MDM on BYOD devices:
- **Can** see: Device model, OS version, compliance status, installed work apps
- **Can** do: Install/remove work apps, enforce passcode, wipe work container
- **Cannot** see: Personal photos, personal emails, browsing history, location (unless device lost)
- **Cannot** do: Read personal messages, access personal apps, wipe personal data

### 3.3 BYOD Security Requirements

#### 3.3.1 Containerization and Data Separation

- Work data must be stored in managed container only
- Work and personal data must remain separated
- Company has no access to personal data
- Personal apps cannot access work data
- Copy/paste between work and personal may be restricted

#### 3.3.2 Company Wipe Rights

By enrolling in BYOD:
- Employee acknowledges company may wipe work container at any time
- Wipe occurs upon: termination, device loss, security incident, non-compliance
- Personal data is not affected by work container wipe
- Employee responsible for personal data backup

#### 3.3.3 BYOD Reimbursement

- Eligible employees may receive monthly stipend for BYOD usage
- Stipend amount determined by HR policy
- Stipend does not grant company ownership of device

### 3.4 Remote Work Eligibility

#### 3.4.1 Role-Based Eligibility

| Role Category | Remote Eligibility | Approval Required |
|---------------|-------------------|-------------------|
| **Engineering** | Full remote or hybrid | Manager |
| **Product/Design** | Full remote or hybrid | Manager |
| **Customer Support** | Hybrid preferred (timezone coverage) | Manager + Director |
| **Finance** | Hybrid (access to sensitive systems) | Manager + CFO |
| **HR** | Hybrid (employee relations) | Manager + CHRO |
| **Security Operations** | Hybrid (incident response) | Manager + CISO |
| **Executive** | Flexible | Self-determined |

#### 3.4.2 Remote Work Agreement

All remote workers must:
1. Acknowledge remote work policy
2. Confirm suitable home office environment
3. Agree to security requirements
4. Provide emergency contact information
5. Review remote work annually

### 3.5 Home Office Security

#### 3.5.1 Workspace Requirements

| Requirement | Description | Verification |
|-------------|-------------|--------------|
| **Dedicated Workspace** | Area for work activities with privacy | Self-attestation |
| **Physical Security** | Ability to secure device when away | Self-attestation |
| **Screen Privacy** | Screen not visible to non-employees | Self-attestation |
| **Video Background** | Professional background or blur for calls | Manager review |
| **Document Handling** | Secure disposal of printed documents | Self-attestation |

#### 3.5.2 Network Security Requirements

| Requirement | Standard | Enforcement |
|-------------|----------|-------------|
| **Wi-Fi Encryption** | WPA3 required (WPA2 minimum with complex password) | VPN blocks weak encryption |
| **Router Password** | Changed from default | Self-attestation |
| **Router Firmware** | Updated regularly | Self-attestation |
| **Network Segmentation** | Separate guest and IoT networks recommended | Guidance provided |
| **ISP Router** | Disable remote management | Guidance provided |

#### 3.5.3 Prohibited Home Office Activities

- Processing Restricted data without CISO approval
- Allowing family members to use work devices
- Storing work credentials on shared family devices
- Printing Restricted or Confidential documents
- Using work devices on unsecured networks

### 3.6 VPN Usage Requirements

#### 3.6.1 When VPN is Required

| Access Type | VPN Required | Notes |
|-------------|--------------|-------|
| **Internal applications** | Yes (always-on) | Auto-connect on company network access |
| **Email (Outlook/Gmail)** | Yes | Included in always-on VPN |
| **Slack/Teams** | Yes | Included in always-on VPN |
| **Public websites** | Optional | Split-tunnel for non-sensitive browsing |
| **Cloud applications (SSO)** | Yes | Zero-trust access via VPN |
| **Developer resources** | Yes | Always required for code access |

#### 3.6.2 VPN Configuration

- VPN client pre-installed on all company devices
- Always-on VPN enforced for company resource access
- Split-tunneling allowed only for approved non-sensitive traffic
- VPN disconnection requires re-authentication
- VPN logs retained for 90 days

#### 3.6.3 VPN Troubleshooting

If VPN is unavailable:
1. Check internet connectivity
2. Restart VPN client
3. Try alternate VPN endpoint
4. Contact IT support
5. Do not attempt to access company resources without VPN

### 3.7 Public Wi-Fi Restrictions

#### 3.7.1 Public Wi-Fi Policy

| Location | Permission | Requirements |
|----------|------------|--------------|
| **Coffee shops, hotels** | Permitted | VPN required, no sensitive work |
| **Airports, airplanes** | Permitted | VPN required, screen privacy filter |
| **Conference Wi-Fi** | Permitted | VPN required, verify network name |
| **Unsecured open networks** | Discouraged | VPN required, brief use only |
| **Unknown networks** | Prohibited | Use mobile hotspot instead |

#### 3.7.2 Public Wi-Fi Security Measures

When using public Wi-Fi:
- Always enable VPN before any company access
- Verify network name with venue staff (evil twin prevention)
- Use screen privacy filter in visible areas
- Avoid accessing Restricted data
- Prefer mobile hotspot for sensitive work
- Do not leave device unattended
- Disable auto-connect to open networks

#### 3.7.3 Prohibited Activities on Public Wi-Fi

Even with VPN:
- Accessing Restricted data
- Financial transactions
- Password changes
- Code deployments to production
- Privileged administrative access

### 3.8 Lost or Stolen Device Procedures

#### 3.8.1 Reporting Timeline

| Action | Timeline | Contact |
|--------|----------|---------|
| **Report loss/theft** | Within 1 hour of discovery | IT Help Desk + Security |
| **File police report** | Within 24 hours (if stolen) | Local police |
| **Provide police report number** | Within 48 hours | IT Help Desk |

#### 3.8.2 Reporting Information Required

- Device type and model
- Serial number (if known)
- Time and location last seen
- Time and location discovered missing
- Circumstances (theft, lost, unknown)
- Whether device was locked
- Police report number (if applicable)

#### 3.8.3 Company Response

Upon report of lost/stolen device:
1. **Immediate**: Remote wipe initiated (within 1 hour of report)
2. **Immediate**: All active sessions terminated
3. **Same day**: Password reset for all accounts accessed from device
4. **Same day**: Review access logs for suspicious activity
5. **Within 48 hours**: Issue replacement device (if company-owned)
6. **Within 7 days**: Security review and incident closure

#### 3.8.4 Employee Responsibilities

- Report loss/theft immediately (even if device might be found)
- Cooperate with investigation
- File police report for theft
- Do not attempt to recover device yourself (safety risk)
- Await instruction before using replacement device

### 3.9 Data Protection on Mobile Devices

#### 3.9.1 Data Classification Restrictions

| Classification | Mobile Access | Storage Allowed | Offline Access |
|----------------|---------------|-----------------|----------------|
| **Restricted** | Prohibited (exceptions require CISO approval) | Never | Never |
| **Confidential** | Permitted with MDM | Work container only | Manager approval |
| **Internal** | Permitted with MDM | Work container only | Permitted |
| **Public** | Permitted | Any | Permitted |

#### 3.9.2 Data Handling Requirements

- Do not save Confidential data to device local storage (use cloud)
- Do not email Confidential data to personal email accounts
- Do not copy company data to personal cloud storage (iCloud, Google Drive personal)
- Do not take photos of confidential screens or documents
- Use approved file sharing methods only (company SharePoint, approved apps)

### 3.10 Remote Work Monitoring Disclosure

#### 3.10.1 Monitoring Scope

GreenLang may monitor the following on company devices and during remote work:
- Network traffic through company VPN
- Access to company applications and systems
- Endpoint security status (compliance with MDM)
- Login times and session duration
- File access audit logs (for Confidential/Restricted data)

#### 3.10.2 What is NOT Monitored

GreenLang does not monitor:
- Personal device content outside work container
- Personal browsing (when VPN split-tunnel enabled)
- Physical workspace (no remote camera access)
- Keystroke logging (except for security investigation with approval)
- Personal communications

#### 3.10.3 Monitoring Notice

- Employees acknowledge monitoring in employment agreement
- Monitoring conducted for security and compliance purposes only
- Access to monitoring data restricted to Security and HR (with justification)
- Monitoring data retained per data retention policy

---

## 4. Roles and Responsibilities

| Role | Responsibilities |
|------|------------------|
| **CISO** | Policy ownership, exception approval for Restricted data access |
| **IT Department** | MDM administration, VPN management, device provisioning |
| **Security Operations** | Incident response for lost/stolen devices, monitoring review |
| **HR** | Remote work agreements, policy acknowledgment tracking |
| **Managers** | Remote work approval, productivity management, security awareness |
| **Employees** | Policy compliance, incident reporting, secure workspace maintenance |

---

## 5. Procedures

### 5.1 Enrolling a BYOD Device

1. Submit BYOD request via IT portal
2. Review and sign BYOD agreement
3. IT provides enrollment instructions
4. Install MDM profile and required apps
5. Complete enrollment verification
6. Begin using device for work

### 5.2 Setting Up Home Office

1. Complete remote work agreement
2. Review home office security checklist
3. Configure home network per guidelines
4. Install VPN and verify connectivity
5. Confirm compliance with manager
6. Complete remote work acknowledgment

### 5.3 Reporting a Lost/Stolen Device

1. Call IT Help Desk immediately: +1-XXX-XXX-XXXX
2. Provide device information and circumstances
3. IT initiates remote wipe
4. File police report (theft)
5. Provide police report number to IT
6. Obtain replacement device (if applicable)

---

## 6. Exceptions

Exceptions to this policy require:

1. Written business justification
2. Security risk assessment
3. Approval from CISO (for data classification exceptions) or IT Director (for device exceptions)
4. Compensating controls documented
5. Time-limited approval (maximum 6 months)

Exceptions for accessing Restricted data on mobile devices are rarely granted and require executive sponsor.

---

## 7. Enforcement

Violations of this policy may result in:

- Warning and mandatory security training (first offense)
- Temporary suspension of remote work privileges
- Revocation of BYOD enrollment
- Disciplinary action up to termination
- Financial liability for data breach caused by negligence

Non-compliant devices will be automatically blocked from company resources until compliance is restored.

---

## 8. Related Documents

| Document ID | Document Name |
|-------------|---------------|
| POL-002 | Acceptable Use Policy |
| POL-003 | Access Control Policy |
| POL-004 | Data Classification Policy |
| POL-011 | Encryption and Key Management Policy |
| POL-013 | Physical Security Policy |
| STD-MDM-001 | Mobile Device Management Standard |
| STD-VPN-001 | VPN Configuration Standard |
| AGR-BYOD-001 | BYOD User Agreement |
| AGR-REMOTE-001 | Remote Work Agreement |

---

## 9. Definitions

| Term | Definition |
|------|------------|
| **BYOD** | Bring Your Own Device - using personal devices for work purposes |
| **COPE** | Corporate-Owned, Personally-Enabled - company device with personal use allowed |
| **Containerization** | Technology separating work and personal data on a device |
| **MDM** | Mobile Device Management - software for managing and securing mobile devices |
| **Remote Wipe** | Ability to erase data from a device remotely |
| **Split Tunneling** | VPN configuration allowing some traffic to bypass VPN |
| **Work Container** | Encrypted partition on device containing only work data |
| **WPA3** | Wi-Fi Protected Access 3 - current wireless security standard |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | Security Team | Initial policy creation |

---

**Document Classification: Confidential**

*This policy is the property of GreenLang Climate OS. Unauthorized distribution, copying, or disclosure is prohibited.*
