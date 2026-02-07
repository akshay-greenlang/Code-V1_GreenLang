# POL-015: Media Protection Policy

**Document Control**

| Attribute | Value |
|-----------|-------|
| Document ID | POL-015 |
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

This policy establishes requirements for the protection, handling, and disposal of removable media and portable storage devices at GreenLang. The purpose is to prevent data leakage, protect against malware introduction, and ensure proper handling of physical media throughout its lifecycle.

Removable media presents significant security risks including data exfiltration, malware infection, and loss of sensitive information. This policy defines controls to mitigate these risks while allowing necessary business use of approved media types.

This policy supports compliance with SOC 2 Type II (CC6.7 - Media), ISO 27001:2022 (A.7.10 Storage media), and GDPR requirements for protecting personal data on portable storage.

---

## 2. Scope

This policy applies to:

- **Media Types**: USB flash drives, external hard drives, SD cards, optical media (CD/DVD/Blu-ray), magnetic tapes, and any other portable storage devices
- **Personnel**: All employees, contractors, and third parties who handle removable media containing GreenLang data
- **Data**: All company data stored on or transferred via removable media
- **Systems**: All company systems that interact with removable media

### 2.1 Out of Scope

- Internal hard drives and SSDs (covered by asset management and encryption policies)
- Cloud storage (covered by data protection policy)
- Backup tapes managed by data center provider (governed by vendor contracts)

---

## 3. Policy Statement

GreenLang restricts the use of removable media to minimize data loss and security risks. Only company-issued, encrypted removable media may be used for authorized business purposes. All media must be properly labeled, tracked, and disposed of according to the data classification of information stored.

### 3.1 Approved Removable Media

#### 3.1.1 Authorized Media Types

Only the following media types are approved for use with GreenLang data:

| Media Type | Approval Status | Encryption | Use Case |
|------------|-----------------|------------|----------|
| **Company-issued encrypted USB** | Approved | Hardware AES-256 | Data transfer, field use |
| **Company-issued encrypted external drive** | Approved | Hardware AES-256 | Large data transfer, backup |
| **Optical media (read-only distribution)** | Approved with authorization | N/A (read-only content) | Software distribution, customer delivery |
| **Backup tapes** | Approved (IT only) | Hardware encryption | Disaster recovery |

#### 3.1.2 Company-Issued Media Specifications

| Media Type | Vendor/Model | Encryption | Capacity | Assigned To |
|------------|--------------|------------|----------|-------------|
| USB Flash Drive | Kingston IronKey D300S | FIPS 140-2 L3, AES-256 | 32GB, 64GB | Individuals |
| External Drive | Apricorn Aegis Padlock | FIPS 140-2 L2, AES-256 | 1TB, 2TB | Teams/Departments |
| External SSD | Samsung T7 Shield + BitLocker | AES-256 | 1TB, 2TB | Individuals (approved) |

### 3.2 Use Case Restrictions

#### 3.2.1 Permitted Use Cases

| Use Case | Media Type | Approval Required | Logging |
|----------|------------|-------------------|---------|
| **Data transfer between air-gapped systems** | Encrypted USB | IT approval | Required |
| **Customer data delivery** | Encrypted USB/drive | Manager + Security | Required |
| **System recovery media** | Optical/USB | IT only | Asset tracked |
| **Backup for travel** | Encrypted USB | Manager | Required |
| **Conference presentation** | Encrypted USB | None (Internal data only) | Recommended |

#### 3.2.2 Restricted Use Cases (Require CISO Approval)

- Transferring Restricted data
- Transferring data to third parties
- Taking media outside company premises for extended periods
- Using media in foreign countries (export control consideration)

#### 3.2.3 Prohibited Use Cases

- Storing Restricted data on any removable media without CISO exception
- Using personal USB drives for company data
- Connecting unknown or found USB devices
- Bypassing endpoint protection for media access
- Copying entire databases to removable media

### 3.3 Encryption Requirements

#### 3.3.1 Hardware Encryption (Required)

All approved removable media must use hardware-based encryption:

| Requirement | Specification |
|-------------|---------------|
| **Algorithm** | AES-256 (minimum) |
| **Mode** | XTS-AES for full disk encryption |
| **Certification** | FIPS 140-2 Level 2 minimum |
| **Authentication** | PIN/password with brute-force protection |
| **Lockout** | Device wipe after 10 failed attempts |

#### 3.3.2 Software Encryption (Secondary)

Software encryption may supplement hardware encryption:

| Use Case | Tool | Requirements |
|----------|------|--------------|
| Individual files | 7-Zip AES-256 | Complex password, share separately |
| Volumes | VeraCrypt | AES-256, complex password |
| Windows drives | BitLocker | TPM + PIN or password |
| macOS drives | FileVault | Recovery key stored in IT |

#### 3.3.3 Password Requirements for Encrypted Media

- Minimum 12 characters
- Mix of upper, lower, numbers, symbols
- Not derived from device serial number
- Changed annually or upon personnel change
- Stored in approved password manager (not on the media)

### 3.4 Labeling Requirements

#### 3.4.1 Required Labels

All removable media must be labeled with:

| Field | Example | Location |
|-------|---------|----------|
| **Asset Tag** | GL-USB-2026-0001 | Physical label on device |
| **Classification** | CONFIDENTIAL | Physical label on device |
| **Owner** | Security Team | Physical label on device |
| **Department** | Engineering | Physical label on device |
| **Issue Date** | 2026-02 | Asset management system |

#### 3.4.2 Classification Label Colors

| Classification | Label Color | Border |
|----------------|-------------|--------|
| **Restricted** | Red background, white text | Red |
| **Confidential** | Orange background, black text | Orange |
| **Internal** | Blue background, white text | Blue |
| **Public** | Green background, black text | Green |

#### 3.4.3 Labeling Procedure

1. IT issues asset tag upon media provisioning
2. Appropriate classification label applied based on intended use
3. Owner and department added
4. Media registered in asset management system
5. Label verified before media release to user

### 3.5 Storage Requirements

#### 3.5.1 When Not in Use

| Classification | Storage Requirement |
|----------------|---------------------|
| **Restricted** | Locked safe, access logged |
| **Confidential** | Locked cabinet or drawer |
| **Internal** | Secured workspace |
| **Public** | No special requirement |

#### 3.5.2 Storage Location Standards

- Media containing Confidential or higher data must be stored in locked container
- Keys or combinations controlled by media owner or designated custodian
- Shared media stored in department-controlled locked cabinet
- Server room media stored in media fireproof safe
- Environmental controls: cool, dry, away from magnetic fields

#### 3.5.3 Inventory Requirements

| Frequency | Activity | Responsibility |
|-----------|----------|----------------|
| **Continuous** | Track issuance and return | IT Asset Management |
| **Quarterly** | Physical inventory check | Department managers |
| **Annually** | Full reconciliation | IT + Internal Audit |

### 3.6 Transfer Procedures

#### 3.6.1 Internal Transfers

Transferring media within GreenLang:

1. Log transfer in asset management (from, to, date, reason)
2. Verify recipient is authorized for data classification
3. Hand-deliver or use internal mail with tracking
4. Recipient acknowledges receipt
5. Update asset ownership

#### 3.6.2 External Transfers

Transferring media to external parties:

1. Obtain manager and Security approval
2. Verify recipient organization security posture
3. Execute or verify NDA covers data
4. Encrypt media and provide password via separate channel
5. Use tracked courier (FedEx, UPS with signature)
6. Recipient confirms receipt
7. Document in transfer log

#### 3.6.3 Chain of Custody

For Restricted or Confidential data transfers:

| Step | Documentation |
|------|---------------|
| **Origin** | Date, time, custodian, data description |
| **Transfer** | Method, tracking number, carrier |
| **Receipt** | Date, time, recipient, condition |
| **Storage** | Location, access controls |
| **Return/Destruction** | Date, method, verification |

### 3.7 Sanitization Requirements

#### 3.7.1 NIST 800-88 Compliance

All media sanitization must follow NIST Special Publication 800-88 Rev. 1:

| Method | Description | Use Case |
|--------|-------------|----------|
| **Clear** | Overwrite with non-sensitive data | Reuse within organization |
| **Purge** | Advanced overwrite or block erase | Reuse or transfer |
| **Destroy** | Physical destruction | Disposal of Restricted media |

#### 3.7.2 Sanitization by Media Type

| Media Type | Clear Method | Purge Method | Destroy Method |
|------------|--------------|--------------|----------------|
| **USB Flash** | ATA Secure Erase | Cryptographic erase | Shredding |
| **HDD** | 3-pass overwrite | NIST purge pattern | Degaussing + shredding |
| **SSD** | Block erase | Cryptographic erase | Shredding |
| **Optical (CD/DVD)** | N/A | N/A | Shredding only |
| **Magnetic Tape** | N/A | Degaussing | Shredding |

#### 3.7.3 Sanitization Verification

- Sanitization logged with date, method, technician, verification
- Sample verification using forensic tools
- Certificate of destruction for third-party services
- Retention of sanitization records for 7 years

### 3.8 Disposal Procedures

#### 3.8.1 Disposal Methods by Classification

| Classification | Disposal Method | Verification |
|----------------|-----------------|--------------|
| **Restricted** | Physical destruction (witnessed) | Certificate + photo |
| **Confidential** | Physical destruction or verified purge | Certificate |
| **Internal** | Purge + recycle or destroy | Log entry |
| **Public** | Recycle | None required |

#### 3.8.2 Physical Destruction Methods

| Method | Equipment | Media Types |
|--------|-----------|-------------|
| **Shredding** | NSA/CSS EPL-listed shredder | All solid-state media |
| **Degaussing** | NSA-approved degausser | Magnetic media (HDD, tape) |
| **Incineration** | Licensed facility | Optical media |
| **Disintegration** | Industrial disintegrator | High-security destruction |

#### 3.8.3 Third-Party Destruction Services

When using third-party destruction:
- Vendor must be certified (NAID AAA preferred)
- Chain of custody maintained until destruction
- Witnessed destruction or video documentation
- Certificate of destruction obtained
- Vendor audited annually

### 3.9 Prohibited Media Types

The following media types are prohibited for use with company data:

| Media Type | Prohibition Reason |
|------------|--------------------|
| **Personal USB drives** | Uncontrolled, unencrypted, untraceable |
| **Unencrypted USB drives** | Data exposure risk |
| **Promotional USB drives** | Potential malware vector |
| **Found or unknown USB devices** | Malware risk (USB drop attacks) |
| **Consumer external drives** | Typically unencrypted |
| **Writable optical media (CD-R, DVD-R)** | Cannot be reliably encrypted or sanitized |
| **Memory cards (SD, microSD)** | Easy to lose, difficult to secure |
| **Floppy disks** | Obsolete, unreliable |

#### 3.9.1 Found Device Procedure

If an unknown USB device or media is found:
1. Do not connect to any computer
2. Report to IT Security immediately
3. Turn over device to Security
4. Security will analyze in isolated environment
5. Device disposed of or returned to owner if identified

### 3.10 Exception Process

#### 3.10.1 Exception Request Requirements

Requests for exceptions must include:
1. Business justification (why standard methods insufficient)
2. Specific media type and use case
3. Data classification of affected data
4. Duration of exception needed
5. Proposed compensating controls
6. Risk acceptance acknowledgment

#### 3.10.2 Approval Authority

| Exception Type | Approval Required |
|----------------|-------------------|
| Using non-standard encrypted media | IT Director |
| Storing Confidential data on media | Manager + CISO |
| Storing Restricted data on media | CISO + Data Owner |
| Using unencrypted media | Prohibited (no exceptions) |
| Taking media internationally | CISO + Legal |

#### 3.10.3 Exception Documentation

Approved exceptions must:
- Be documented in exception register
- Have defined expiration date (maximum 12 months)
- Be reviewed quarterly
- Include compensating controls
- Be revoked upon change in circumstances

---

## 4. Roles and Responsibilities

| Role | Responsibilities |
|------|------------------|
| **CISO** | Policy ownership, exception approval, annual review |
| **IT Department** | Media provisioning, encryption configuration, asset tracking |
| **IT Security** | Media policy enforcement, sanitization, incident response |
| **Department Managers** | Approve team media requests, quarterly inventory |
| **Employees** | Proper handling, labeling, storage, reporting loss |
| **Facilities** | Secure storage infrastructure, destruction services |

---

## 5. Procedures

### 5.1 Requesting Removable Media

1. Submit request via IT service portal
2. Specify use case, data classification, duration
3. Manager approves request
4. IT provisions and configures encrypted media
5. Media issued with asset tag and classification label
6. Employee acknowledges receipt and responsibilities

### 5.2 Returning Media

1. Verify all data has been copied to primary storage
2. Notify IT of intent to return
3. IT verifies sanitization or performs sanitization
4. Media returned to IT inventory
5. Asset record updated

### 5.3 Reporting Lost or Stolen Media

1. Report immediately (within 1 hour) to IT Security
2. Provide: asset tag, data classification, approximate loss time/location
3. If theft suspected, file police report
4. IT Security assesses data exposure risk
5. If Confidential or higher data, initiate incident response
6. Document in incident management system

---

## 6. Exceptions

Exceptions to this policy must follow the exception process in Section 3.10.

Standard exceptions are not granted for:
- Using personal USB devices for company data
- Using unencrypted media for any classification
- Bypassing endpoint protection controls

---

## 7. Enforcement

Violations of this policy may result in:

- Confiscation of unauthorized media
- Mandatory security awareness retraining
- Written warning
- Disciplinary action up to termination
- Legal action if data breach results

Systems are configured to:
- Block unauthorized USB devices
- Log all media connections
- Alert Security on policy violations

---

## 8. Related Documents

| Document ID | Document Name |
|-------------|---------------|
| POL-004 | Data Classification Policy |
| POL-011 | Encryption and Key Management Policy |
| POL-014 | Mobile Device and Remote Work Policy |
| POL-017 | Asset Management Policy |
| STD-MEDIA-001 | Media Handling Standard |
| PRO-SANIT-001 | Media Sanitization Procedure |
| PRO-DEST-001 | Media Destruction Procedure |

---

## 9. Definitions

| Term | Definition |
|------|------------|
| **Air-gapped** | System physically isolated from networks |
| **Chain of Custody** | Documentation tracking possession of media |
| **Cryptographic Erase** | Destroying encryption key to render data unrecoverable |
| **Degaussing** | Using magnetic field to erase magnetic media |
| **Hardware Encryption** | Encryption performed by dedicated hardware on device |
| **NAID** | National Association for Information Destruction |
| **NSA/CSS EPL** | NSA Evaluated Products List for media destruction |
| **Removable Media** | Portable storage device that can be removed from systems |
| **Sanitization** | Process of removing data from media |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | Security Team | Initial policy creation |

---

**Document Classification: Confidential**

*This policy is the property of GreenLang Climate OS. Unauthorized distribution, copying, or disclosure is prohibited.*
