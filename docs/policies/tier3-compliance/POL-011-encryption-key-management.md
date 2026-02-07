# POL-011: Encryption and Key Management Policy

**Document Control**

| Attribute | Value |
|-----------|-------|
| Document ID | POL-011 |
| Version | 1.0 |
| Classification | Confidential |
| Policy Tier | Tier 3 - Compliance |
| Owner | Chief Information Security Officer (CISO) |
| Approved By | Director of Security |
| Effective Date | 2026-02-06 |
| Last Review | 2026-02-06 |
| Next Review | 2027-02-06 |

---

## 1. Purpose

This policy establishes comprehensive requirements for the use of cryptographic controls and the management of cryptographic keys within GreenLang Climate OS. The purpose is to ensure the confidentiality, integrity, and authenticity of information assets through the proper implementation of encryption technologies and the secure lifecycle management of cryptographic keys.

Encryption serves as a critical control for protecting sensitive data, including customer emissions data, financial information, personally identifiable information (PII), and regulatory submissions. Proper key management ensures that cryptographic protections remain effective throughout the data lifecycle and that keys are available when needed while protected from unauthorized access.

This policy aligns with industry best practices including NIST Special Publication 800-57 (Key Management), NIST SP 800-131A (Transitioning Cryptographic Algorithms), and supports compliance with SOC 2 Type II, ISO 27001:2022, and GDPR encryption requirements.

---

## 2. Scope

This policy applies to:

- **Systems**: All GreenLang production, staging, and development systems that process, store, or transmit data requiring cryptographic protection
- **Data**: All data classified as Restricted, Confidential, or Internal per POL-004 Data Classification Policy
- **Personnel**: All employees, contractors, and third parties who manage, configure, or have access to cryptographic systems
- **Technologies**: All encryption software, hardware security modules (HSMs), key management systems (KMS), and certificate authorities
- **Locations**: Cloud environments (AWS), on-premises infrastructure, endpoint devices, and third-party systems processing GreenLang data

### 2.1 Out of Scope

- Personal encryption used for non-business purposes on personal devices
- Third-party systems where GreenLang does not control encryption implementation (governed by vendor contracts)

---

## 3. Policy Statement

GreenLang is committed to protecting information assets through the appropriate use of cryptographic controls. All data must be encrypted according to its classification level, using only approved algorithms and protocols. Cryptographic keys must be managed throughout their lifecycle with appropriate controls for generation, storage, distribution, rotation, and destruction.

### 3.1 Encryption Requirements by Data Classification

#### 3.1.1 Restricted Data

Restricted data requires the highest level of cryptographic protection:

| Protection Layer | Requirement |
|-----------------|-------------|
| **At Rest** | AES-256-GCM encryption with unique data encryption keys (DEKs) |
| **In Transit** | TLS 1.3 with forward secrecy (mandatory) |
| **Database Fields** | Application-level field encryption for sensitive attributes |
| **Backups** | AES-256-GCM with separate backup encryption keys |
| **Key Storage** | Hardware Security Module (HSM) for key encryption keys (KEKs) |

Examples: Customer API credentials, payment card data, HSM master keys, root certificates.

#### 3.1.2 Confidential Data

Confidential data requires strong encryption with managed keys:

| Protection Layer | Requirement |
|-----------------|-------------|
| **At Rest** | AES-256 encryption (GCM or CBC mode with HMAC) |
| **In Transit** | TLS 1.3 preferred, TLS 1.2 minimum |
| **Database** | Transparent Data Encryption (TDE) or volume encryption |
| **Backups** | AES-256 encryption |
| **Key Storage** | AWS KMS or HashiCorp Vault |

Examples: Customer emissions data, financial reports, employee PII, audit logs.

#### 3.1.3 Internal Data

Internal data requires transport encryption and optional storage encryption:

| Protection Layer | Requirement |
|-----------------|-------------|
| **At Rest** | Volume-level encryption (recommended) |
| **In Transit** | TLS 1.3 preferred, TLS 1.2 minimum |
| **Backups** | Encrypted backup storage |
| **Key Storage** | AWS KMS or application-managed |

Examples: Internal documentation, non-sensitive configuration, development data.

#### 3.1.4 Public Data

Public data requires transport encryption only:

| Protection Layer | Requirement |
|-----------------|-------------|
| **At Rest** | No encryption required |
| **In Transit** | TLS 1.3 preferred, TLS 1.2 minimum (HTTPS) |

Examples: Marketing materials, public documentation, press releases.

### 3.2 Approved Cryptographic Algorithms

Only the following cryptographic algorithms and configurations are approved for use:

#### 3.2.1 Symmetric Encryption

| Algorithm | Key Size | Mode | Use Case |
|-----------|----------|------|----------|
| **AES-256-GCM** | 256-bit | GCM (AEAD) | Primary choice for all encryption (provides confidentiality + integrity) |
| **AES-256-CBC** | 256-bit | CBC + HMAC-SHA256 | Legacy systems only (must include separate MAC) |
| **ChaCha20-Poly1305** | 256-bit | AEAD | Mobile/embedded where AES-NI unavailable |

**Prohibited**: DES, 3DES, RC4, Blowfish, AES-ECB mode.

#### 3.2.2 Asymmetric Encryption

| Algorithm | Key Size | Use Case |
|-----------|----------|----------|
| **RSA** | 4096-bit minimum | Key exchange, digital signatures (legacy compatibility) |
| **ECDSA P-384** | 384-bit | Digital signatures, TLS certificates |
| **ECDH P-384** | 384-bit | Key agreement, TLS key exchange |
| **Ed25519** | 256-bit | Digital signatures (internal systems) |
| **X25519** | 256-bit | Key agreement (internal systems) |

**Prohibited**: RSA < 2048-bit, DSA, ECDSA P-192.

#### 3.2.3 Hashing Algorithms

| Algorithm | Output Size | Use Case |
|-----------|-------------|----------|
| **SHA-256** | 256-bit | General hashing, HMAC, file integrity |
| **SHA-384** | 384-bit | Digital signatures, certificate hashing |
| **SHA-512** | 512-bit | High-security applications |
| **BLAKE2b** | Variable | Password hashing (with Argon2) |
| **Argon2id** | Variable | Password storage (with salt + pepper) |

**Prohibited**: MD5, SHA-1 (except for non-security purposes).

### 3.3 Key Generation

All cryptographic keys must be generated using approved methods:

#### 3.3.1 Requirements

- **Random Number Generation**: Use only Cryptographically Secure Pseudo-Random Number Generators (CSPRNG)
  - AWS KMS key generation
  - HSM hardware random number generators
  - /dev/urandom (Linux) or CryptGenRandom (Windows)
  - Python: `secrets` module (not `random`)
- **Root and Master Keys**: Must be generated within FIPS 140-2 Level 3 (or higher) HSMs
- **Data Encryption Keys**: May be generated via software CSPRNG when protected by HSM-stored KEKs
- **Key Uniqueness**: Each key must be unique; key reuse across environments is prohibited

#### 3.3.2 Key Generation Procedures

1. Identify key purpose, classification, and required algorithm
2. Select appropriate generation mechanism (HSM for KEKs, KMS for DEKs)
3. Generate key with full entropy
4. Immediately store in approved key management system
5. Document key metadata (ID, purpose, creation date, owner, expiration)
6. Enable key in production only after testing in non-production

### 3.4 Key Storage

Reference: SEC-006 (Secrets Management) for detailed implementation.

#### 3.4.1 Approved Key Storage Systems

| Key Type | Primary Storage | Backup Storage |
|----------|-----------------|----------------|
| **Root Keys / Master Keys** | AWS CloudHSM (FIPS 140-2 L3) | Offline HSM backup |
| **Key Encryption Keys (KEKs)** | AWS KMS | Cross-region replica |
| **Data Encryption Keys (DEKs)** | HashiCorp Vault | Encrypted backup in S3 |
| **TLS Private Keys** | Vault PKI / AWS ACM | HSM escrow |
| **Application Secrets** | HashiCorp Vault | Encrypted backup |

#### 3.4.2 Storage Requirements

- Keys must never be stored in plaintext
- Keys must never be stored in source code, configuration files, or logs
- Key storage systems must implement access controls and audit logging
- Encryption of keys at rest using higher-level keys (key hierarchy)
- Geographic separation of backup keys from primary keys

### 3.5 Key Rotation Schedules

All cryptographic keys must be rotated according to the following schedule:

| Key Type | Maximum Lifetime | Rotation Trigger | Notification |
|----------|------------------|------------------|--------------|
| **Data Encryption Keys (DEKs)** | 1 year | Automatic | 30 days before expiry |
| **Key Encryption Keys (KEKs)** | 2 years | Automatic | 60 days before expiry |
| **Root Keys / Master Keys** | 5 years | Manual (ceremony) | 180 days before expiry |
| **TLS Certificates** | 1 year (public), 2 years (internal) | Automatic via ACME/Vault | 30 days before expiry |
| **API Keys** | 90 days | Automatic | 14 days before expiry |
| **Service Account Keys** | 1 year | Automatic | 30 days before expiry |

#### 3.5.1 Emergency Rotation

Immediate key rotation is required when:
- Key compromise is suspected or confirmed
- Personnel with key access leave the organization
- Vulnerability discovered in cryptographic algorithm
- Regulatory or audit requirement

### 3.6 Key Backup and Recovery

#### 3.6.1 Backup Requirements

- All production keys must have documented backup procedures
- Backups must be encrypted with a separate backup KEK
- Backup keys must be stored in geographically separate locations
- Backup encryption keys must be protected by HSM
- Recovery procedures must be tested quarterly

#### 3.6.2 Recovery Procedures

1. Initiate key recovery request through ticketing system
2. Obtain approval from two authorized personnel (dual control)
3. Retrieve backup from secure storage
4. Decrypt backup using backup KEK (requires HSM access)
5. Verify key integrity using stored checksums
6. Import key into target key management system
7. Document recovery in audit log

### 3.7 Key Destruction

#### 3.7.1 Cryptographic Erasure

When data protected by a key must be permanently destroyed, cryptographic erasure may be used:

1. Verify no systems require access to encrypted data
2. Document destruction decision and obtain approval
3. Destroy all copies of the encryption key
4. Verify destruction across all key storage locations
5. Update key inventory to reflect destruction
6. Retain destruction record for audit purposes (7 years)

#### 3.7.2 Key Destruction Methods

| Storage Type | Destruction Method |
|--------------|--------------------|
| **HSM Keys** | Secure zeroization via HSM command |
| **KMS Keys** | Schedule key deletion (7-30 day waiting period) |
| **Vault Secrets** | Destroy secret versions, then delete secret |
| **File-based Keys** | Secure overwrite (DoD 5220.22-M) then delete |
| **Memory** | Explicit zeroing before deallocation |

### 3.8 Certificate Management

Reference: SEC-004 (TLS/SSL Certificate Management) for detailed procedures.

#### 3.8.1 Certificate Requirements

- All public-facing services must use certificates from trusted public CAs
- Internal services may use certificates from GreenLang internal CA (managed via Vault PKI)
- Certificates must use approved key algorithms (ECDSA P-384 preferred, RSA-4096 if required)
- Subject Alternative Names (SANs) must be minimized to required domains
- Wildcard certificates are discouraged; use when managing >10 subdomains

#### 3.8.2 Certificate Lifecycle

1. **Request**: Submit CSR through Vault PKI or CA portal
2. **Validation**: Domain validation (DV) minimum; Organization validation (OV) for customer-facing
3. **Issuance**: Automated via ACME (Let's Encrypt) or Vault PKI
4. **Deployment**: Automated certificate deployment via Kubernetes secrets or AWS ACM
5. **Monitoring**: Certificate expiry monitoring via Prometheus/Grafana alerts
6. **Renewal**: Automatic renewal 30 days before expiry
7. **Revocation**: Immediate revocation if private key compromised

### 3.9 TLS/SSL Standards

#### 3.9.1 Protocol Requirements

| Context | Minimum Version | Preferred Version |
|---------|-----------------|-------------------|
| **Public APIs** | TLS 1.2 | TLS 1.3 |
| **Internal Services** | TLS 1.2 | TLS 1.3 |
| **Database Connections** | TLS 1.2 | TLS 1.3 |
| **Email (SMTP/IMAP)** | TLS 1.2 | TLS 1.3 |

**Prohibited**: SSL 2.0, SSL 3.0, TLS 1.0, TLS 1.1.

#### 3.9.2 Cipher Suite Configuration

TLS 1.3 cipher suites (in order of preference):
```
TLS_AES_256_GCM_SHA384
TLS_CHACHA20_POLY1305_SHA256
TLS_AES_128_GCM_SHA256
```

TLS 1.2 cipher suites (in order of preference):
```
ECDHE-ECDSA-AES256-GCM-SHA384
ECDHE-RSA-AES256-GCM-SHA384
ECDHE-ECDSA-CHACHA20-POLY1305
ECDHE-RSA-CHACHA20-POLY1305
ECDHE-ECDSA-AES128-GCM-SHA256
ECDHE-RSA-AES128-GCM-SHA256
```

#### 3.9.3 Additional TLS Requirements

- Perfect Forward Secrecy (PFS) required (ECDHE key exchange)
- HSTS header required for all HTTPS services (max-age >= 1 year)
- OCSP Stapling enabled for all public certificates
- Session tickets disabled or rotated frequently
- Compression disabled (CRIME/BREACH mitigation)

### 3.10 Encryption Audit Requirements

#### 3.10.1 Continuous Monitoring

- Key usage logging enabled for all KMS and Vault operations
- Certificate expiry alerts configured (30, 14, 7, 1 day)
- TLS configuration scanning (weekly via SSL Labs API)
- Encryption-at-rest verification (monthly)

#### 3.10.2 Periodic Audits

| Audit Activity | Frequency | Owner |
|----------------|-----------|-------|
| Key inventory reconciliation | Quarterly | Security Engineering |
| Key rotation compliance | Monthly | Security Operations |
| Certificate inventory review | Monthly | Platform Engineering |
| Encryption configuration review | Quarterly | Security Engineering |
| Key access review | Quarterly | Security Operations |
| Algorithm deprecation review | Annual | CISO |
| Full cryptographic audit | Annual | External auditor |

---

## 4. Roles and Responsibilities

| Role | Responsibilities |
|------|------------------|
| **CISO** | Policy ownership, exception approval, algorithm deprecation decisions |
| **Security Engineering** | KMS/HSM/Vault implementation, key lifecycle automation, encryption architecture |
| **Security Operations** | Key rotation monitoring, incident response for key compromise, access reviews |
| **Platform Engineering** | TLS configuration, certificate deployment, encryption-at-rest implementation |
| **Development Teams** | Proper use of encryption APIs, secure handling of keys in application code |
| **Compliance** | Audit support, evidence collection, regulatory alignment |

---

## 5. Procedures

### 5.1 Requesting a New Encryption Key

1. Submit key request via Security team ticketing system
2. Specify: key purpose, classification, algorithm, rotation requirements
3. Security Engineering reviews and approves request
4. Key generated in appropriate system (HSM/KMS/Vault)
5. Key ID and access instructions provided to requestor
6. Key documented in encryption key inventory

### 5.2 Emergency Key Rotation

1. Identify compromised or suspected-compromised key
2. Notify Security Operations immediately (PagerDuty)
3. Initiate emergency rotation procedure
4. Generate new key with same parameters
5. Re-encrypt affected data (if necessary)
6. Revoke/destroy compromised key
7. Conduct post-incident review

### 5.3 Algorithm Migration

When algorithms are deprecated:
1. Security Engineering publishes migration timeline
2. Affected teams notified 90 days in advance
3. New keys generated with approved algorithm
4. Data re-encrypted or re-signed as required
5. Old keys scheduled for destruction after migration verification

---

## 6. Exceptions

Exceptions to this policy require:

1. Written business justification
2. Risk assessment documenting compensating controls
3. Approval from CISO (or designee)
4. Time-limited exception (maximum 12 months)
5. Documented remediation plan

Exception requests must be submitted via the security exception process and will be reviewed within 5 business days.

---

## 7. Enforcement

Violations of this policy may result in:

- Revocation of system access
- Disciplinary action up to and including termination
- Legal action if laws or regulations are violated
- Vendor contract termination for third-party violations

Suspected violations should be reported to security@greenlang.io or via the anonymous reporting hotline.

---

## 8. Related Documents

| Document ID | Document Name |
|-------------|---------------|
| POL-004 | Data Classification Policy |
| POL-010 | Encryption Policy (Tier 2 overview) |
| SEC-004 | TLS/SSL Certificate Management Standard |
| SEC-006 | Secrets Management Standard |
| STD-CRYPTO-001 | Cryptographic Implementation Standard |
| PRO-KEY-001 | Key Ceremony Procedure |
| PRO-KEY-002 | Key Recovery Procedure |

---

## 9. Definitions

| Term | Definition |
|------|------------|
| **AEAD** | Authenticated Encryption with Associated Data - encryption mode providing confidentiality and integrity |
| **CSPRNG** | Cryptographically Secure Pseudo-Random Number Generator |
| **DEK** | Data Encryption Key - key used to encrypt actual data |
| **Forward Secrecy** | Property ensuring session keys cannot be compromised even if long-term keys are |
| **HSM** | Hardware Security Module - tamper-resistant hardware for key storage |
| **KEK** | Key Encryption Key - key used to encrypt other keys |
| **KMS** | Key Management Service - cloud service for key management |
| **OCSP** | Online Certificate Status Protocol - protocol for checking certificate revocation |
| **PFS** | Perfect Forward Secrecy - see Forward Secrecy |
| **TLS** | Transport Layer Security - protocol for encrypted network communications |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | Security Team | Initial policy creation |

---

**Document Classification: Confidential**

*This policy is the property of GreenLang Climate OS. Unauthorized distribution, copying, or disclosure is prohibited.*
