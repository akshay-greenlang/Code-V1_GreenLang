# GL-011 FUELCRAFT Security Policy

**Document Classification:** REGULATORY SENSITIVE
**Agent:** GL-011 FUELCRAFT - FuelManagementOrchestrator
**Version:** 1.0.0
**Last Updated:** 2025-12-01
**Owner:** GreenLang Foundation Security Team
**Review Cycle:** Quarterly

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Security Classification](#2-security-classification)
3. [Defense in Depth Strategy](#3-defense-in-depth-strategy)
4. [Network Segmentation](#4-network-segmentation)
5. [Zero Trust Architecture](#5-zero-trust-architecture)
6. [Encryption Standards](#6-encryption-standards)
7. [Key Management](#7-key-management)
8. [IEC 62443-4-2 Compliance](#8-iec-62443-4-2-compliance)
9. [OWASP Top 10 Mitigation](#9-owasp-top-10-mitigation)
10. [SOC 2 Type II Controls](#10-soc-2-type-ii-controls)
11. [Vulnerability Management](#11-vulnerability-management)
12. [Incident Response](#12-incident-response)
13. [Access Control Policy](#13-access-control-policy)
14. [Data Protection Standards](#14-data-protection-standards)
15. [Authentication and Authorization](#15-authentication-and-authorization)
16. [Secrets Management](#16-secrets-management)
17. [Network Security](#17-network-security)
18. [Audit and Logging](#18-audit-and-logging)
19. [Container Security](#19-container-security)
20. [Kubernetes Security](#20-kubernetes-security)
21. [API Security](#21-api-security)
22. [Cryptographic Standards](#22-cryptographic-standards)
23. [Security Testing](#23-security-testing)
24. [Compliance Verification](#24-compliance-verification)
25. [Policy Exceptions](#25-policy-exceptions)
26. [Document Control](#26-document-control)

---

## 1. Executive Summary

### 1.1 Purpose

This Security Policy document establishes the comprehensive security requirements, controls, and procedures for the GL-011 FUELCRAFT FuelManagementOrchestrator. The agent handles REGULATORY SENSITIVE fuel procurement, storage, and lifecycle data subject to EPA fuel standards, ISO 50001 energy management requirements, and financial audit regulations.

FUELCRAFT is a mission-critical agent that manages:
- **Fuel procurement contracts** with multi-million dollar values
- **Inventory management** across distributed storage facilities
- **Quality assurance** ensuring regulatory compliance (ASTM, ISO)
- **Financial tracking** subject to SOX and GAAP requirements
- **Environmental compliance** for emissions and spill prevention

### 1.2 Scope

This policy applies to:
- All GL-011 FUELCRAFT components, including Python application code, configuration files, and deployment manifests
- All environments: development, staging, and production
- All personnel with access to GL-011 systems or data
- All third-party integrations (fuel suppliers, ERP systems, financial systems, tank monitoring systems)
- All data processed, stored, or transmitted by the agent including:
  - Fuel procurement contracts and pricing
  - Inventory levels and quality data
  - Financial transactions and accounting records
  - Environmental compliance data
  - Supplier and vendor information

### 1.3 Security Objectives

| Objective | Description | Priority | Business Impact |
|-----------|-------------|----------|-----------------|
| Confidentiality | Protect fuel pricing, contracts, and supplier data from unauthorized disclosure | Critical | Competitive advantage loss, contract breaches |
| Integrity | Ensure accuracy and completeness of fuel calculations, inventory, and financial data | Critical | Financial losses, regulatory violations, operational disruptions |
| Availability | Maintain 99.95% uptime for continuous fuel operations | Critical | Production downtime, fuel shortages, safety risks |
| Auditability | Provide complete audit trails with cryptographic verification | Critical | Regulatory compliance, financial audits, fraud prevention |
| Non-repudiation | Ensure provenance tracking with SHA-256 hashing for all transactions | High | Legal disputes, financial audits, compliance verification |
| Compliance | Meet EPA, ISO 50001, SOX, and industry standards | Critical | Regulatory fines, operational shutdowns, legal liability |

### 1.4 Key Security Features

GL-011 FUELCRAFT implements the following core security features:

1. **Zero-Hallucination Calculations**: All fuel calculations use deterministic formulas with no LLM involvement in numeric computations
   - Fuel energy content calculations (HHV, LHV)
   - Inventory volume adjustments (temperature, pressure corrections)
   - Cost calculations and financial reconciliation
   - Quality parameter validation (sulfur content, API gravity, viscosity)

2. **SHA-256 Provenance Hashing**: Complete audit trail with cryptographic verification
   - Every fuel transaction hashed
   - Immutable audit chain
   - Tamper detection
   - Regulatory evidence collection

3. **Zero Hardcoded Credentials**: All secrets loaded from environment variables or Kubernetes secrets
   - No credentials in source code
   - Validated at startup
   - Automated secret rotation
   - External Secrets Operator integration

4. **Thread-Safe Operations**: Concurrent request handling with proper synchronization
   - Distributed locks for inventory updates
   - ACID transactions for financial operations
   - Optimistic concurrency control
   - Deadlock prevention

5. **Input Validation**: Pydantic-based validation with field constraints
   - Type safety
   - Range validation
   - Business rule enforcement
   - Injection prevention

6. **Role-Based Access Control (RBAC)**: Five-tier access model
   - fuel_admin: Full administrative access
   - fuel_manager: Procurement and inventory management
   - fuel_operator: Day-to-day operations
   - fuel_analyst: Reporting and analysis
   - fuel_viewer: Read-only access

### 1.5 Regulatory Context

FUELCRAFT operates in a heavily regulated environment:

**Environmental Regulations:**
- EPA Clean Air Act fuel standards
- EPA Spill Prevention, Control, and Countermeasure (SPCC) rules
- State-level fuel quality requirements
- Underground Storage Tank (UST) regulations

**Financial Regulations:**
- Sarbanes-Oxley Act (SOX) internal controls
- Generally Accepted Accounting Principles (GAAP)
- International Financial Reporting Standards (IFRS)
- Commodity trading regulations

**Energy Management:**
- ISO 50001 Energy Management Systems
- ASTM fuel quality standards
- Industry-specific standards (IATA for aviation fuel, etc.)

**Data Protection:**
- SOC 2 Type II security and availability
- ISO 27001 information security
- GDPR for European operations
- State privacy laws (CCPA, CPRA)

### 1.6 Threat Landscape

FUELCRAFT faces unique security threats:

**Financial Threats:**
- Fraudulent procurement orders
- Price manipulation
- Invoice fraud
- Unauthorized fund transfers
- Insider trading on commodity information

**Operational Threats:**
- Inventory data tampering
- Quality data falsification
- Supply chain disruption
- Tank level manipulation
- Safety system interference

**Compliance Threats:**
- Audit trail destruction
- Regulatory report manipulation
- Quality certification forgery
- Environmental violation concealment

**Cyber Threats:**
- Ransomware targeting fuel operations
- APT groups seeking commodity intelligence
- Supplier impersonation attacks
- API exploitation
- Credential theft

### 1.7 Security Architecture Philosophy

FUELCRAFT's security architecture is built on these principles:

1. **Defense in Depth**: Multiple layers of security controls
2. **Zero Trust**: Never trust, always verify
3. **Least Privilege**: Minimal permissions required
4. **Fail Secure**: Default deny, graceful degradation
5. **Security by Design**: Security embedded in architecture
6. **Assume Breach**: Continuous monitoring and detection
7. **Cryptographic Verification**: All critical operations hashed
8. **Immutable Audit**: Tamper-evident logging

---

## 2. Security Classification

### 2.1 Data Classification Levels

| Level | Description | Examples | Handling Requirements | Legal Implications |
|-------|-------------|----------|----------------------|-------------------|
| REGULATORY SENSITIVE | Data subject to EPA, financial audit, and energy management regulations | Fuel contracts, pricing, quality data, financial transactions, environmental compliance records | Encryption required (AES-256), 7-year retention, audit logging, access controls, geographic restrictions | Regulatory fines, legal liability, operational shutdowns |
| CONFIDENTIAL | Business-sensitive operational data | Supplier relationships, procurement strategies, inventory forecasts, cost optimization models | Encryption required, access controls, 5-year retention | Competitive disadvantage, contract breaches |
| INTERNAL | Non-public operational information | Configuration settings, performance metrics, system documentation | Access controls, standard logging, 3-year retention | Operational risk |
| PUBLIC | Information cleared for public release | API documentation, general capabilities, public reports | No restrictions | None |

### 2.2 GL-011 Data Classification Matrix

| Data Category | Classification | Encryption at Rest | Encryption in Transit | Retention Period | Regulatory Requirement |
|--------------|----------------|-------------------|----------------------|------------------|----------------------|
| Fuel Procurement Contracts | REGULATORY SENSITIVE | AES-256-GCM | TLS 1.3 | 7 years | SOX, Contract Law |
| Fuel Pricing Data | REGULATORY SENSITIVE | AES-256-GCM | TLS 1.3 | 7 years | Commodity Regulations |
| Inventory Levels | REGULATORY SENSITIVE | AES-256-GCM | TLS 1.3 | 7 years | EPA SPCC, ISO 50001 |
| Quality Assurance Data | REGULATORY SENSITIVE | AES-256-GCM | TLS 1.3 | 7 years | ASTM, EPA Fuel Standards |
| Financial Transactions | REGULATORY SENSITIVE | AES-256-GCM | TLS 1.3 | 7 years | SOX, GAAP |
| Environmental Compliance | REGULATORY SENSITIVE | AES-256-GCM | TLS 1.3 | 7 years | EPA, State Regulations |
| Audit Trails | REGULATORY SENSITIVE | AES-256-GCM | TLS 1.3 | 7 years | SOX, SOC 2 |
| Provenance Hashes | REGULATORY SENSITIVE | AES-256-GCM | TLS 1.3 | 7 years | Internal Controls |
| Supplier Information | CONFIDENTIAL | AES-256-GCM | TLS 1.3 | 5 years | Business Confidentiality |
| Optimization Models | CONFIDENTIAL | AES-256-GCM | TLS 1.3 | 5 years | Trade Secrets |
| Tank Monitoring Data | CONFIDENTIAL | AES-256-GCM | TLS 1.3 | 3 years | Operational |
| Performance Metrics | INTERNAL | Optional | TLS 1.3 | 1 year | Internal Use |
| Health Check Data | INTERNAL | No | TLS 1.3 | 30 days | Operational |
| API Documentation | PUBLIC | No | TLS 1.3 | Indefinite | None |

### 2.3 Regulatory Data Handling

Fuel data processed by GL-011 is subject to the following regulatory frameworks:

#### 2.3.1 EPA Fuel Standards

**Clean Air Act Requirements:**
- Fuel quality monitoring (sulfur content, volatility, oxygenate levels)
- Reformulated gasoline (RFG) compliance
- Renewable Fuel Standard (RFS) tracking
- Data must be retained for minimum 5 years after production/import
- Electronic records must be tamper-evident
- Quarterly reporting to EPA

**SPCC (Spill Prevention, Control, and Countermeasure) Rules:**
- Oil storage facility capacity tracking
- Spill prevention plan compliance
- Secondary containment verification
- Regular inspection records
- Immediate spill reporting requirements

#### 2.3.2 Financial Regulations

**Sarbanes-Oxley Act (SOX) Compliance:**
- Internal controls over financial reporting
- Audit trail for all financial transactions
- Segregation of duties
- Management certification of controls
- Independent auditor verification
- Data retention: 7 years minimum

**GAAP/IFRS Requirements:**
- Inventory valuation methods (FIFO, LIFO, weighted average)
- Lower of cost or market (LCM) reporting
- Financial statement accuracy
- Audit trail for all adjustments
- Quarterly financial close processes

#### 2.3.3 Energy Management

**ISO 50001 Requirements:**
- Energy baseline establishment
- Energy performance indicators (EnPIs)
- Significant energy use (SEU) tracking
- Energy review and audits
- Management review documentation
- Continual improvement evidence

**ASTM Fuel Quality Standards:**
- ASTM D975 (Diesel Fuel)
- ASTM D4814 (Spark-Ignition Engine Fuel)
- ASTM D1655 (Aviation Turbine Fuel)
- Quality test results and certificates
- Chain of custody documentation

### 2.4 Data Sovereignty and Geographic Restrictions

**Data Localization Requirements:**

| Region | Requirement | Implementation | Compliance |
|--------|-------------|----------------|------------|
| European Union | GDPR - data must stay in EU | EU-based Kubernetes clusters | ✅ Compliant |
| China | Data must be stored in China | China-region deployment | ✅ Compliant |
| Russia | Data localization law | Russian data center | ⚠️ Planned |
| India | Local storage for critical data | India-region deployment | ✅ Compliant |
| Brazil | LGPD compliance | Brazil data center | ✅ Compliant |

**Cross-Border Data Transfer Controls:**
- Standard Contractual Clauses (SCCs) for EU data
- Binding Corporate Rules (BCRs) for intra-company transfers
- Adequacy decisions verification
- Transfer Impact Assessments (TIAs)
- Encryption for all international transfers

### 2.5 Data Lifecycle Management

**Data States and Protection:**

| Data State | Security Controls | Monitoring |
|------------|------------------|------------|
| **Data in Transit** | TLS 1.3, mutual TLS for B2B | SSL/TLS inspection, DLP |
| **Data at Rest** | AES-256-GCM, encrypted volumes | Access logging, integrity checks |
| **Data in Use** | Memory encryption (Intel SGX), isolated processing | Runtime monitoring, container security |
| **Data in Backup** | Encrypted backups, offline copies | Backup integrity verification |
| **Data in Archive** | Encrypted cold storage, immutable | Archive verification, retention tracking |

**Data Destruction:**

| Data Type | Destruction Method | Verification | Timeline |
|-----------|-------------------|--------------|----------|
| Production data (end of retention) | Cryptographic erasure, overwrite (DoD 5220.22-M) | Certificate of destruction | Within 30 days of retention expiry |
| Backup media | Physical destruction (shredding, degaussing) | Chain of custody, video evidence | Within 90 days |
| Development/test data | Secure deletion, database truncation | Automated verification | Quarterly cleanup |
| Log files | Automated purge, archival | Retention policy enforcement | Per retention schedule |

---

## 3. Defense in Depth Strategy

### 3.1 Overview

GL-011 FUELCRAFT implements a comprehensive defense-in-depth strategy with multiple layers of security controls. This approach ensures that if one security layer is compromised, additional layers provide continued protection.

### 3.2 Security Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 7: Governance & Compliance                                     │
│ - Security policies, procedures, training                            │
│ - Compliance audits, certifications                                  │
│ - Risk management, business continuity                               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 6: Data Security                                               │
│ - Data classification, encryption (AES-256-GCM)                      │
│ - DLP (Data Loss Prevention), tokenization                           │
│ - Provenance hashing (SHA-256), immutable audit logs                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 5: Application Security                                        │
│ - Input validation (Pydantic), output encoding                       │
│ - Authentication (JWT, API keys), RBAC authorization                 │
│ - Secure coding practices, SAST/DAST scanning                        │
│ - OWASP Top 10 mitigation                                            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 4: Platform Security (Kubernetes)                              │
│ - Pod Security Standards (Restricted), RBAC                          │
│ - Network Policies, Service Mesh (Istio)                             │
│ - Secrets management, admission controllers                          │
│ - Runtime security (Falco), vulnerability scanning                   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 3: Host Security                                               │
│ - OS hardening (CIS benchmarks), minimal attack surface              │
│ - Host-based IDS/IPS, file integrity monitoring                      │
│ - Patch management, vulnerability scanning                           │
│ - Secure boot, TPM verification                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 2: Network Security                                            │
│ - Network segmentation, VLANs, microsegmentation                     │
│ - Firewalls (NGFWs), WAF, IDS/IPS                                    │
│ - TLS 1.3, mTLS, VPNs                                                │
│ - DDoS protection, rate limiting                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 1: Physical & Environmental Security                           │
│ - Data center access controls, biometrics                            │
│ - Video surveillance, security guards                                │
│ - Environmental controls (fire, flood, temperature)                  │
│ - Redundant power, connectivity                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Layer 1: Physical & Environmental Security

**Data Center Security:**

| Control | Implementation | Verification | Responsible Party |
|---------|----------------|--------------|-------------------|
| Perimeter Security | 8-foot fencing, anti-ramming barriers | Quarterly inspection | Facility Management |
| Access Control | Biometric + badge + PIN (3-factor) | Access log audit (monthly) | Security Operations |
| Video Surveillance | 24/7 recording, 90-day retention | Camera health checks (daily) | Security Operations |
| Security Guards | 24/7 on-site personnel, background checks | Guard logs, incident reports | Security Operations |
| Visitor Management | Escort required, visitor logs, NDA | Visitor audit (monthly) | Facility Management |
| Man Trap Entry | Dual-door airlock, weight sensors | Functional test (quarterly) | Facility Management |
| Fire Suppression | FM-200 gas system, smoke detectors | Annual inspection, testing | Facility Management |
| Environmental Monitoring | Temperature, humidity, water leak sensors | Continuous monitoring | NOC Team |
| Backup Power | N+1 UPS, dual-feed generators | Monthly load testing | Facility Management |
| Redundant Connectivity | Diverse fiber paths, multiple ISPs | Failover testing (quarterly) | Network Operations |

**Equipment Security:**

- Server racks: Locked, keycard access
- Hot/cold aisle containment
- Cable management and security
- Equipment disposal: NIST 800-88 media sanitization

### 3.4 Layer 2: Network Security

**Network Segmentation:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ Internet Zone (Untrusted)                                            │
│ - Public DNS                                                         │
│ - No direct access to internal resources                            │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ Firewall (Deny All, Allow Specific)
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ DMZ (Demilitarized Zone)                                             │
│ - Load balancers, reverse proxies                                    │
│ - WAF (Web Application Firewall)                                     │
│ - Rate limiting, DDoS protection                                     │
│ - TLS termination                                                    │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ Firewall + IDS/IPS
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Application Zone (Trusted)                                           │
│ - GL-011 FUELCRAFT pods                                             │
│ - Service mesh (Istio) with mTLS                                     │
│ - Network policies (deny by default)                                │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ Internal Firewall
                            ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Data Zone (Highly Restricted)                                        │
│ - PostgreSQL database (encrypted)                                    │
│ - Redis cache (encrypted)                                            │
│ - Secret stores (Vault, AWS Secrets Manager)                         │
│ - No direct internet access                                          │
└─────────────────────────────────────────────────────────────────────┘
```

**Firewall Rules (Stateful Inspection):**

| Rule # | Source | Destination | Port/Protocol | Action | Logging |
|--------|--------|-------------|---------------|--------|---------|
| 1 | Internet | DMZ Load Balancer | 443/TCP (HTTPS) | ALLOW | Yes |
| 2 | DMZ | Application Zone (GL-011) | 8011/TCP | ALLOW | Yes |
| 3 | Application Zone | Data Zone (PostgreSQL) | 5432/TCP | ALLOW | Yes |
| 4 | Application Zone | Data Zone (Redis) | 6379/TCP | ALLOW | Yes |
| 5 | Application Zone | Internet (Anthropic API) | 443/TCP (HTTPS) | ALLOW | Yes |
| 6 | Monitoring Zone | Application Zone | 9011/TCP (metrics) | ALLOW | Yes |
| 7 | Data Zone | Internet | ANY | DENY | Yes |
| 8 | ANY | ANY | ANY | DENY | Yes |

**Web Application Firewall (WAF) Rules:**

- **OWASP ModSecurity Core Rule Set (CRS) 4.0**
- SQL Injection prevention
- Cross-Site Scripting (XSS) prevention
- Local File Inclusion (LFI) / Remote File Inclusion (RFI) prevention
- Command injection prevention
- Geo-blocking for high-risk countries
- Rate limiting: 100 requests/minute per IP
- Custom rules for FUELCRAFT-specific attack patterns

**Intrusion Detection/Prevention System (IDS/IPS):**

| Detection Type | Tool | Action | Alert |
|----------------|------|--------|-------|
| Network anomalies | Suricata | Block + Alert | SOC team (PagerDuty) |
| Port scanning | Zeek | Alert | Security team (Slack) |
| Malware signatures | Snort | Block + Alert | SOC team (PagerDuty) |
| DDoS attacks | Cloudflare | Rate limit + Alert | NOC team |
| Brute force login | Fail2Ban | Block IP (1 hour) | Security log |

**DDoS Protection:**

- **Cloudflare Enterprise**: Global CDN, rate limiting, challenge pages
- **AWS Shield Advanced**: Layer 3/4 protection, always-on detection
- **Rate limiting**: 100 req/min per IP, 1000 req/min globally
- **Connection limits**: Max 10,000 concurrent connections per pod
- **SYN flood protection**: SYN cookies enabled

### 3.5 Layer 3: Host Security

**Operating System Hardening (CIS Benchmarks):**

| Control | Implementation | Verification |
|---------|----------------|--------------|
| Minimal OS | Amazon Linux 2023 (hardened) | Monthly vulnerability scan |
| Disable unnecessary services | SSH disabled, only HTTPS | Service inventory audit |
| Kernel hardening | Secure sysctl parameters | Configuration management |
| File system permissions | Least privilege, no world-writable | File integrity monitoring |
| Audit logging | auditd enabled, 90-day retention | Log review (weekly) |
| SELinux/AppArmor | Enforcing mode | Policy compliance check |
| Automatic security updates | Unattended-upgrades, weekly | Patch compliance report |
| Time synchronization | NTP/Chrony, authenticated | Time drift monitoring |

**Secure sysctl Parameters:**

```bash
# /etc/sysctl.d/99-security.conf
# IP forwarding disabled
net.ipv4.ip_forward = 0

# ICMP redirect disabled
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0

# Source packet routing disabled
net.ipv4.conf.all.accept_source_route = 0

# SYN cookies enabled (SYN flood protection)
net.ipv4.tcp_syncookies = 1

# Log martian packets
net.ipv4.conf.all.log_martians = 1

# Ignore ICMP ping
net.ipv4.icmp_echo_ignore_all = 1

# Randomize memory addresses (ASLR)
kernel.randomize_va_space = 2

# Restrict kernel pointers
kernel.kptr_restrict = 2

# Disable core dumps
kernel.core_uses_pid = 1
fs.suid_dumpable = 0
```

**File Integrity Monitoring (FIM):**

- **Tool**: AIDE (Advanced Intrusion Detection Environment)
- **Monitored paths**:
  - `/bin`, `/sbin`, `/usr/bin`, `/usr/sbin`
  - `/etc` (configuration files)
  - `/lib`, `/lib64` (system libraries)
  - `/boot` (bootloader and kernel)
- **Scan frequency**: Daily (2 AM UTC)
- **Alerting**: Email to security team, SIEM integration
- **Baseline**: Established after initial hardening, re-baseline after approved changes

**Host-Based Intrusion Detection (HIDS):**

- **Tool**: OSSEC
- **Capabilities**:
  - Log analysis (syslog, auth logs)
  - File integrity monitoring
  - Rootkit detection
  - Real-time alerting
  - Active response (block IPs)
- **Alert levels**:
  - Level 10+: PagerDuty alert to SOC
  - Level 7-9: Slack alert to security team
  - Level 0-6: Logged to SIEM

**Vulnerability Scanning:**

| Scanner | Frequency | Scope | SLA |
|---------|-----------|-------|-----|
| Nessus Professional | Weekly | All hosts | Critical: 48h, High: 7d |
| OpenVAS | Daily | Production hosts | Critical: 24h |
| Amazon Inspector | Continuous | AWS resources | Critical: 24h |

**Patch Management:**

| Patch Type | Testing | Deployment Window | Rollback Plan |
|-----------|---------|-------------------|---------------|
| Critical security | 24-hour test environment | Within 48 hours | Automated rollback |
| High security | 3-day test environment | Within 7 days | Automated rollback |
| Medium security | 1-week test environment | Within 30 days | Manual rollback |
| Low security | 2-week test environment | Within 90 days | Manual rollback |

### 3.6 Layer 4: Platform Security (Kubernetes)

**Kubernetes Security Architecture:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ Kubernetes Control Plane (Managed by AWS EKS)                        │
│ - API Server: RBAC, admission controllers, audit logging             │
│ - etcd: Encrypted at rest, mTLS, restricted access                   │
│ - Scheduler: Pod security admission, resource quotas                 │
│ - Controller Manager: Service account token rotation                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────────┐
│ Worker Nodes (EC2 instances)                                         │
│ ┌─────────────────────────────────────────────────────────────┐     │
│ │ Pod: GL-011 FUELCRAFT                                        │     │
│ │ ┌─────────────────────────────────────────────────────────┐ │     │
│ │ │ Container: fuelcraft-app                                 │ │     │
│ │ │ - Non-root user (UID 10011)                              │ │     │
│ │ │ - Read-only root filesystem                              │ │     │
│ │ │ - No privilege escalation                                │ │     │
│ │ │ - Capabilities dropped (ALL)                             │ │     │
│ │ │ - Seccomp: RuntimeDefault                                │ │     │
│ │ │ - AppArmor/SELinux enforcing                             │ │     │
│ │ └─────────────────────────────────────────────────────────┘ │     │
│ │ Network Policy: Ingress/Egress restrictions                  │     │
│ │ Service Account: gl-011-fuelcraft (least privilege RBAC)     │     │
│ └─────────────────────────────────────────────────────────────┘     │
│                                                                       │
│ ┌─────────────────────────────────────────────────────────────┐     │
│ │ Service Mesh (Istio)                                         │     │
│ │ - mTLS between all services                                  │     │
│ │ - Authorization policies                                     │     │
│ │ - Traffic encryption                                         │     │
│ │ - Request authentication                                     │     │
│ └─────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

**Pod Security Standards (PSS) - Restricted Profile:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gl-011-fuelcraft
  namespace: greenlang-agents
  labels:
    app: gl-011-fuelcraft
    security-tier: regulatory-sensitive
spec:
  # Security Context (Pod-level)
  securityContext:
    runAsNonRoot: true
    runAsUser: 10011
    runAsGroup: 10011
    fsGroup: 10011
    fsGroupChangePolicy: "OnRootMismatch"
    seccompProfile:
      type: RuntimeDefault
    supplementalGroups: []

  # Service Account
  serviceAccountName: gl-011-fuelcraft
  automountServiceAccountToken: true

  # Containers
  containers:
    - name: fuelcraft-app
      image: greenlang/gl-011-fuelcraft:1.0.0
      imagePullPolicy: Always

      # Security Context (Container-level)
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        runAsNonRoot: true
        runAsUser: 10011
        runAsGroup: 10011
        capabilities:
          drop:
            - ALL
        seccompProfile:
          type: RuntimeDefault
        # AppArmor (if supported)
        # appArmorProfile:
        #   type: RuntimeDefault

      # Resource limits (prevent DoS)
      resources:
        requests:
          cpu: "500m"
          memory: "1Gi"
          ephemeral-storage: "1Gi"
        limits:
          cpu: "2000m"
          memory: "4Gi"
          ephemeral-storage: "5Gi"

      # Liveness probe (auto-restart on failure)
      livenessProbe:
        httpGet:
          path: /health
          port: 8011
          scheme: HTTP
        initialDelaySeconds: 30
        periodSeconds: 30
        timeoutSeconds: 5
        failureThreshold: 3

      # Readiness probe (traffic routing)
      readinessProbe:
        httpGet:
          path: /ready
          port: 8011
          scheme: HTTP
        initialDelaySeconds: 10
        periodSeconds: 10
        timeoutSeconds: 3
        failureThreshold: 3

      # Startup probe (slow startup grace period)
      startupProbe:
        httpGet:
          path: /health
          port: 8011
        initialDelaySeconds: 0
        periodSeconds: 5
        failureThreshold: 30  # 150 seconds max startup time

      # Volume mounts (read-only except temp)
      volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/cache
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: secrets
          mountPath: /app/secrets
          readOnly: true

      # Environment variables (non-sensitive)
      env:
        - name: AGENT_ID
          value: "GL-011"
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: ENABLE_METRICS
          value: "true"

      # Secrets from Kubernetes Secrets (sensitive)
      envFrom:
        - secretRef:
            name: gl-011-secrets
            optional: false

  # Volumes
  volumes:
    - name: tmp
      emptyDir:
        sizeLimit: 1Gi
    - name: cache
      emptyDir:
        sizeLimit: 2Gi
    - name: config
      configMap:
        name: gl-011-config
        defaultMode: 0440
    - name: secrets
      secret:
        secretName: gl-011-secrets
        defaultMode: 0440

  # Pod anti-affinity (distribute across nodes)
  affinity:
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          podAffinityTerm:
            labelSelector:
              matchExpressions:
                - key: app
                  operator: In
                  values:
                    - gl-011-fuelcraft
            topologyKey: kubernetes.io/hostname

  # Tolerations (none - run only on dedicated nodes)
  tolerations: []

  # Priority (high priority for critical service)
  priorityClassName: high-priority

  # DNS policy
  dnsPolicy: ClusterFirst

  # Restart policy
  restartPolicy: Always

  # Termination grace period
  terminationGracePeriodSeconds: 30
```

**Kubernetes RBAC (Role-Based Access Control):**

```yaml
---
# ServiceAccount
apiVersion: v1
kind: ServiceAccount
metadata:
  name: gl-011-fuelcraft
  namespace: greenlang-agents
  labels:
    app: gl-011-fuelcraft
    security-tier: regulatory-sensitive
automountServiceAccountToken: true

---
# Role (Namespace-scoped)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: gl-011-role
  namespace: greenlang-agents
rules:
  # Allow reading ConfigMaps
  - apiGroups: [""]
    resources: ["configmaps"]
    resourceNames: ["gl-011-config"]
    verbs: ["get", "watch", "list"]

  # Allow reading Secrets
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["gl-011-secrets", "gl-011-api-keys", "gl-011-tls"]
    verbs: ["get"]

  # Allow reading own Pod information
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list"]

  # Allow reading Pod logs (for debugging)
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get", "list"]

  # Allow creating Events (for audit trail)
  - apiGroups: [""]
    resources: ["events"]
    verbs: ["create", "patch"]

---
# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: gl-011-rolebinding
  namespace: greenlang-agents
subjects:
  - kind: ServiceAccount
    name: gl-011-fuelcraft
    namespace: greenlang-agents
roleRef:
  kind: Role
  name: gl-011-role
  apiGroup: rbac.authorization.k8s.io
```

**Network Policies (Zero Trust Networking):**

```yaml
---
# Network Policy: Ingress (Allow from specific sources only)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: gl-011-ingress-policy
  namespace: greenlang-agents
spec:
  podSelector:
    matchLabels:
      app: gl-011-fuelcraft
  policyTypes:
    - Ingress
  ingress:
    # Allow from NGINX Ingress Controller
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
          podSelector:
            matchLabels:
              app.kubernetes.io/name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8011

    # Allow from Prometheus (metrics scraping)
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
          podSelector:
            matchLabels:
              app: prometheus
      ports:
        - protocol: TCP
          port: 9011

    # Allow from Istio sidecar (service mesh)
    - from:
        - podSelector:
            matchLabels:
              app: gl-011-fuelcraft
      ports:
        - protocol: TCP
          port: 15090  # Envoy stats
        - protocol: TCP
          port: 15021  # Health checks

---
# Network Policy: Egress (Allow to specific destinations only)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: gl-011-egress-policy
  namespace: greenlang-agents
spec:
  podSelector:
    matchLabels:
      app: gl-011-fuelcraft
  policyTypes:
    - Egress
  egress:
    # Allow DNS resolution
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53

    # Allow to PostgreSQL database
    - to:
        - namespaceSelector:
            matchLabels:
              name: greenlang-data
          podSelector:
            matchLabels:
              app: postgresql
      ports:
        - protocol: TCP
          port: 5432

    # Allow to Redis cache
    - to:
        - namespaceSelector:
            matchLabels:
              name: greenlang-data
          podSelector:
            matchLabels:
              app: redis
      ports:
        - protocol: TCP
          port: 6379

    # Allow to Kubernetes API server (for leader election, etc.)
    - to:
        - namespaceSelector:
            matchLabels:
              name: default
      ports:
        - protocol: TCP
          port: 443

    # Allow HTTPS egress for external APIs (Anthropic, fuel suppliers)
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
            except:
              - 169.254.169.254/32  # Block AWS metadata service
              - 10.0.0.0/8          # Block internal networks
              - 172.16.0.0/12
              - 192.168.0.0/16
      ports:
        - protocol: TCP
          port: 443  # HTTPS only

    # Allow to Istio control plane
    - to:
        - namespaceSelector:
            matchLabels:
              name: istio-system
      ports:
        - protocol: TCP
          port: 15012  # Istiod XDS
```

**Admission Controllers:**

| Controller | Purpose | Enforcement |
|------------|---------|-------------|
| **PodSecurity** | Enforce Pod Security Standards (Restricted) | Enforcing |
| **ResourceQuota** | Prevent resource exhaustion | Enforcing |
| **LimitRanger** | Set default resource limits | Enforcing |
| **ImagePolicyWebhook** | Verify container image signatures | Enforcing |
| **NamespaceLifecycle** | Prevent operations on terminating namespaces | Enforcing |
| **ServiceAccount** | Automatic service account injection | Enforcing |
| **AlwaysPullImages** | Always pull images (prevent local tampering) | Enforcing |
| **OPA/Gatekeeper** | Custom policy enforcement | Enforcing |

**OPA Gatekeeper Policies (Custom Constraints):**

```yaml
---
# Constraint: Require security labels on all pods
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequiredLabels
metadata:
  name: require-security-tier-label
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
    namespaces:
      - greenlang-agents
  parameters:
    labels:
      - key: "security-tier"
        allowedRegex: "^(public|internal|confidential|regulatory-sensitive)$"

---
# Constraint: Block privileged containers
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sPSPPrivilegedContainer
metadata:
  name: block-privileged-containers
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]

---
# Constraint: Require read-only root filesystem
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sPSPReadOnlyRootFilesystem
metadata:
  name: require-readonly-rootfs
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
    namespaces:
      - greenlang-agents

---
# Constraint: Block host network, PID, IPC
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sPSPHostNamespace
metadata:
  name: block-host-namespace
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]

---
# Constraint: Require resource limits
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sContainerLimits
metadata:
  name: require-resource-limits
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
    namespaces:
      - greenlang-agents
  parameters:
    cpu: "2000m"
    memory: "4Gi"
```

**Secrets Management (External Secrets Operator):**

```yaml
---
# SecretStore (AWS Secrets Manager backend)
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secretsmanager
  namespace: greenlang-agents
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa

---
# ExternalSecret (Pull secrets from AWS Secrets Manager)
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: gl-011-secrets
  namespace: greenlang-agents
spec:
  refreshInterval: 1h  # Refresh every hour
  secretStoreRef:
    name: aws-secretsmanager
    kind: SecretStore
  target:
    name: gl-011-secrets
    creationPolicy: Owner
    deletionPolicy: Retain
  data:
    # Anthropic API Key
    - secretKey: ANTHROPIC_API_KEY
      remoteRef:
        key: greenlang/gl-011/anthropic-api-key
        version: latest

    # Database credentials
    - secretKey: DATABASE_URL
      remoteRef:
        key: greenlang/gl-011/database-url

    # Redis credentials
    - secretKey: REDIS_URL
      remoteRef:
        key: greenlang/gl-011/redis-url

    # Encryption key
    - secretKey: ENCRYPTION_KEY
      remoteRef:
        key: greenlang/gl-011/encryption-key

    # JWT signing key
    - secretKey: JWT_SECRET_KEY
      remoteRef:
        key: greenlang/gl-011/jwt-secret-key

    # Fuel supplier API keys
    - secretKey: FUEL_SUPPLIER_API_KEY
      remoteRef:
        key: greenlang/gl-011/fuel-supplier-api-key

    # ERP system credentials
    - secretKey: ERP_USERNAME
      remoteRef:
        key: greenlang/gl-011/erp-username
    - secretKey: ERP_PASSWORD
      remoteRef:
        key: greenlang/gl-011/erp-password

    # Tank monitoring system credentials
    - secretKey: TANK_MONITOR_API_KEY
      remoteRef:
        key: greenlang/gl-011/tank-monitor-api-key
```

**Runtime Security (Falco):**

```yaml
---
# Falco Rule: Detect shell execution in container
- rule: Shell Execution in Container
  desc: Detect shell execution in GL-011 FUELCRAFT container
  condition: >
    spawned_process and
    container.name = "fuelcraft-app" and
    (proc.name in (sh, bash, zsh, fish))
  output: >
    Shell executed in FUELCRAFT container
    (user=%user.name command=%proc.cmdline container=%container.name)
  priority: WARNING
  tags: [container, shell, T1059]

---
# Falco Rule: Detect file modification in read-only filesystem
- rule: Write to Read-Only Filesystem
  desc: Detect write attempts to read-only filesystem
  condition: >
    open_write and
    container.name = "fuelcraft-app" and
    fd.name startswith /app and
    not fd.name startswith /tmp and
    not fd.name startswith /app/cache
  output: >
    Write to read-only filesystem detected
    (file=%fd.name user=%user.name container=%container.name)
  priority: ERROR
  tags: [filesystem, integrity]

---
# Falco Rule: Detect privilege escalation
- rule: Privilege Escalation Attempt
  desc: Detect setuid or privilege escalation in container
  condition: >
    spawned_process and
    container.name = "fuelcraft-app" and
    (proc.name in (sudo, su, setuid))
  output: >
    Privilege escalation attempt
    (command=%proc.cmdline user=%user.name container=%container.name)
  priority: CRITICAL
  tags: [privilege_escalation, T1548]

---
# Falco Rule: Detect secrets access
- rule: Secrets File Access
  desc: Detect access to Kubernetes secrets
  condition: >
    open_read and
    container.name = "fuelcraft-app" and
    fd.name startswith /app/secrets and
    not proc.name in (fuelcraft-app, python)
  output: >
    Unauthorized secrets access
    (file=%fd.name process=%proc.name user=%user.name)
  priority: CRITICAL
  tags: [secrets, credential_access]

---
# Falco Rule: Detect network connection to unexpected destination
- rule: Unexpected Egress Connection
  desc: Detect network connection to non-whitelisted destination
  condition: >
    outbound and
    container.name = "fuelcraft-app" and
    not fd.sip in (postgresql_ip, redis_ip) and
    not fd.sport = 443  # Allow HTTPS
  output: >
    Unexpected egress connection
    (destination=%fd.sip:%fd.sport container=%container.name)
  priority: WARNING
  tags: [network, exfiltration]
```

### 3.7 Layer 5: Application Security

**Secure Coding Practices:**

| Practice | Implementation | Verification |
|----------|----------------|--------------|
| Input Validation | Pydantic models with field constraints | Unit tests, SAST |
| Output Encoding | Automatic JSON encoding, no raw HTML | DAST, code review |
| Parameterized Queries | SQLAlchemy ORM, prepared statements | Code review, SAST |
| Error Handling | Generic error messages, detailed logging | Code review, testing |
| Secure Defaults | Fail-secure, deny by default | Security testing |
| Principle of Least Privilege | Minimal permissions, RBAC | Access control testing |
| Defense in Depth | Multiple security layers | Penetration testing |
| Cryptographic Verification | SHA-256 hashing for all transactions | Audit log verification |

**Input Validation (Pydantic Models):**

```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal
from datetime import datetime
from decimal import Decimal

class FuelProcurementRequest(BaseModel):
    """
    Fuel procurement request with strict validation.

    Security controls:
    - Type safety (Pydantic)
    - Range validation
    - Business rule enforcement
    - Injection prevention
    """

    # Fuel type (enum for strict validation)
    fuel_type: Literal[
        "diesel",
        "gasoline",
        "jet_fuel",
        "natural_gas",
        "coal",
        "biomass",
        "fuel_oil"
    ] = Field(
        ...,
        description="Type of fuel to procure"
    )

    # Quantity (positive, realistic range)
    quantity_gallons: Decimal = Field(
        ...,
        gt=0,
        le=1_000_000,  # Max 1M gallons per order
        decimal_places=2,
        description="Quantity in gallons"
    )

    # Price (positive, realistic range)
    price_per_gallon: Decimal = Field(
        ...,
        gt=0,
        le=100,  # Max $100/gallon (sanity check)
        decimal_places=4,
        description="Price per gallon in USD"
    )

    # Supplier ID (alphanumeric, max 50 chars)
    supplier_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=r"^[A-Z0-9\-]+$",
        description="Supplier identifier"
    )

    # Delivery location (alphanumeric, max 100 chars)
    delivery_location: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9\s\-,\.]+$",
        description="Delivery location"
    )

    # Delivery date (future date only)
    delivery_date: datetime = Field(
        ...,
        description="Requested delivery date"
    )

    # Contract reference (optional, alphanumeric)
    contract_reference: Optional[str] = Field(
        None,
        max_length=50,
        pattern=r"^[A-Z0-9\-]+$",
        description="Contract reference number"
    )

    # Quality specifications
    sulfur_content_ppm: Optional[Decimal] = Field(
        None,
        ge=0,
        le=5000,  # Max 5000 ppm
        description="Sulfur content in ppm"
    )

    api_gravity: Optional[Decimal] = Field(
        None,
        ge=0,
        le=100,
        description="API gravity (petroleum)"
    )

    cetane_number: Optional[Decimal] = Field(
        None,
        ge=0,
        le=100,
        description="Cetane number (diesel)"
    )

    # Authorization
    authorized_by: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z\s\-\.]+$",
        description="Person authorizing procurement"
    )

    @field_validator('delivery_date')
    @classmethod
    def validate_future_date(cls, v: datetime) -> datetime:
        """Ensure delivery date is in the future."""
        if v <= datetime.utcnow():
            raise ValueError("Delivery date must be in the future")

        # Max 1 year in advance
        max_date = datetime.utcnow().replace(year=datetime.utcnow().year + 1)
        if v > max_date:
            raise ValueError("Delivery date cannot be more than 1 year in advance")

        return v

    @model_validator(mode='after')
    def validate_quality_specs(self) -> 'FuelProcurementRequest':
        """Validate quality specifications based on fuel type."""

        # Diesel fuel quality requirements
        if self.fuel_type == "diesel":
            if self.sulfur_content_ppm is None:
                raise ValueError("Sulfur content required for diesel")
            if self.sulfur_content_ppm > 15:
                # ULSD (Ultra-Low Sulfur Diesel) requirement
                raise ValueError("Sulfur content must be ≤15 ppm for ULSD")
            if self.cetane_number is None:
                raise ValueError("Cetane number required for diesel")
            if self.cetane_number < 40:
                raise ValueError("Cetane number must be ≥40 for diesel")

        # Gasoline quality requirements
        elif self.fuel_type == "gasoline":
            if self.sulfur_content_ppm is not None and self.sulfur_content_ppm > 10:
                # EPA Tier 3 requirement
                raise ValueError("Sulfur content must be ≤10 ppm for gasoline")

        # Jet fuel quality requirements
        elif self.fuel_type == "jet_fuel":
            if self.sulfur_content_ppm is not None and self.sulfur_content_ppm > 3000:
                raise ValueError("Sulfur content must be ≤3000 ppm for jet fuel")

        return self

    @field_validator('supplier_id', 'contract_reference')
    @classmethod
    def validate_no_sql_injection(cls, v: Optional[str]) -> Optional[str]:
        """Prevent SQL injection in alphanumeric fields."""
        if v is None:
            return v

        # Block SQL keywords
        sql_keywords = [
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE',
            'ALTER', 'EXEC', 'UNION', 'OR', 'AND', 'WHERE', '--', ';'
        ]
        v_upper = v.upper()
        for keyword in sql_keywords:
            if keyword in v_upper:
                raise ValueError(f"Invalid characters detected: {keyword}")

        return v

    @field_validator('authorized_by', 'delivery_location')
    @classmethod
    def validate_no_xss(cls, v: str) -> str:
        """Prevent XSS in text fields."""
        # Block HTML/JavaScript
        dangerous_chars = ['<', '>', '"', "'", '&', '/', '\\', '{', '}']
        for char in dangerous_chars:
            if char in v:
                raise ValueError(f"Invalid character detected: {char}")

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "fuel_type": "diesel",
                "quantity_gallons": 50000.00,
                "price_per_gallon": 3.25,
                "supplier_id": "SUPPLIER-001",
                "delivery_location": "Tank Farm A, Houston TX",
                "delivery_date": "2025-12-15T08:00:00Z",
                "contract_reference": "CONTRACT-2025-001",
                "sulfur_content_ppm": 10,
                "cetane_number": 45,
                "authorized_by": "John Smith"
            }
        }
```

**SQL Injection Prevention:**

```python
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from models import FuelProcurement

async def get_fuel_procurement_by_id(
    session: AsyncSession,
    procurement_id: str
) -> Optional[FuelProcurement]:
    """
    Retrieve fuel procurement by ID.

    Security: Uses SQLAlchemy ORM with parameterized queries.
    No SQL injection possible.
    """
    # ✅ SECURE: Parameterized query
    stmt = select(FuelProcurement).where(
        FuelProcurement.id == procurement_id
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()

# ❌ INSECURE (Never do this):
# query = f"SELECT * FROM fuel_procurement WHERE id = '{procurement_id}'"
# This is vulnerable to SQL injection!
```

**Command Injection Prevention:**

```python
import subprocess
from typing import List

def execute_fuel_report_generation(
    report_type: str,
    start_date: str,
    end_date: str
) -> bytes:
    """
    Execute report generation script.

    Security: Uses subprocess with argument list (no shell=True).
    Command injection prevented.
    """
    # Validate inputs (allowlist)
    allowed_report_types = ["daily", "weekly", "monthly", "annual"]
    if report_type not in allowed_report_types:
        raise ValueError(f"Invalid report type: {report_type}")

    # ✅ SECURE: Argument list, no shell
    cmd: List[str] = [
        "/usr/bin/python3",
        "/app/scripts/generate_report.py",
        "--type", report_type,
        "--start", start_date,
        "--end", end_date
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        check=True,
        timeout=300,  # 5-minute timeout
        shell=False  # Critical: prevents shell injection
    )

    return result.stdout

# ❌ INSECURE (Never do this):
# cmd = f"python /app/scripts/generate_report.py --type {report_type}"
# subprocess.run(cmd, shell=True)  # Vulnerable to command injection!
```

**Path Traversal Prevention:**

```python
from pathlib import Path
import os

def read_fuel_quality_certificate(
    certificate_id: str
) -> bytes:
    """
    Read fuel quality certificate file.

    Security: Prevents path traversal attacks.
    """
    # Base directory for certificates
    base_dir = Path("/app/data/certificates")

    # Sanitize certificate ID (alphanumeric only)
    if not certificate_id.isalnum():
        raise ValueError("Invalid certificate ID")

    # Construct file path
    file_path = base_dir / f"{certificate_id}.pdf"

    # ✅ SECURE: Verify path is within base directory
    try:
        file_path = file_path.resolve()
        base_dir = base_dir.resolve()

        # Check if resolved path is within base directory
        if not str(file_path).startswith(str(base_dir)):
            raise ValueError("Path traversal detected")
    except Exception as e:
        raise ValueError(f"Invalid path: {e}")

    # Check file exists and is a file (not directory)
    if not file_path.is_file():
        raise FileNotFoundError("Certificate not found")

    # Read file
    with open(file_path, 'rb') as f:
        return f.read()

# ❌ INSECURE (Never do this):
# file_path = f"/app/data/certificates/{certificate_id}.pdf"
# This allows: certificate_id = "../../etc/passwd"
```

**XML External Entity (XXE) Prevention:**

```python
import defusedxml.ElementTree as ET
from typing import Dict, Any

def parse_fuel_delivery_xml(xml_data: str) -> Dict[str, Any]:
    """
    Parse fuel delivery XML data.

    Security: Uses defusedxml to prevent XXE attacks.
    """
    # ✅ SECURE: defusedxml prevents XXE
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML: {e}")

    # Extract data
    delivery_data = {
        "delivery_id": root.find("delivery_id").text,
        "fuel_type": root.find("fuel_type").text,
        "quantity": float(root.find("quantity").text),
        "timestamp": root.find("timestamp").text
    }

    return delivery_data

# ❌ INSECURE (Never do this):
# import xml.etree.ElementTree as ET
# ET.fromstring(xml_data)  # Vulnerable to XXE!
```

**Deserialization Security:**

```python
import json
from typing import Dict, Any

def deserialize_fuel_data(data: str) -> Dict[str, Any]:
    """
    Deserialize fuel data.

    Security: Uses JSON (safe), not pickle (unsafe).
    """
    # ✅ SECURE: JSON deserialization is safe
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

# ❌ INSECURE (Never do this):
# import pickle
# pickle.loads(data)  # Can execute arbitrary code!
```

**LDAP Injection Prevention:**

```python
import ldap
from ldap import filter as ldap_filter

def authenticate_user_ldap(username: str, password: str) -> bool:
    """
    Authenticate user via LDAP.

    Security: LDAP filter escaping to prevent injection.
    """
    # ✅ SECURE: Escape LDAP filter
    escaped_username = ldap_filter.escape_filter_chars(username)

    # Construct filter
    search_filter = f"(uid={escaped_username})"

    # Connect to LDAP
    conn = ldap.initialize("ldap://ldap.example.com")

    try:
        # Bind with user credentials
        conn.simple_bind_s(
            f"uid={escaped_username},ou=users,dc=example,dc=com",
            password
        )
        return True
    except ldap.INVALID_CREDENTIALS:
        return False
    finally:
        conn.unbind_s()

# ❌ INSECURE (Never do this):
# search_filter = f"(uid={username})"
# Allows: username = "*)(uid=*))(|(uid=*"
```

### 3.8 Layer 6: Data Security

**Data Encryption Architecture:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ Data in Transit                                                      │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ Client → Load Balancer: TLS 1.3 (AES-256-GCM)                   │ │
│ │ Load Balancer → Pod: TLS 1.3 (mTLS with Istio)                  │ │
│ │ Pod → Database: TLS 1.3 (client certificate auth)               │ │
│ │ Pod → External API: TLS 1.3 (certificate pinning)               │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Data in Use                                                          │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ Application Memory: Encrypted memory (Intel SGX)                │ │
│ │ Sensitive data: Encrypted until needed, immediate scrubbing     │ │
│ │ No sensitive data in logs, core dumps, error messages           │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Data at Rest                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ Database: AES-256-GCM (column-level encryption)                 │ │
│ │ EBS volumes: AWS KMS encryption                                 │ │
│ │ S3 buckets: SSE-S3 or SSE-KMS                                   │ │
│ │ Backups: GPG encryption (4096-bit RSA)                          │ │
│ │ Secrets: AWS Secrets Manager (AES-256-GCM)                      │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**Column-Level Encryption (Database):**

```python
from cryptography.fernet import Fernet
from sqlalchemy import Column, String, LargeBinary, TypeDecorator
from sqlalchemy.ext.declarative import declarative_base
import os

Base = declarative_base()

class EncryptedString(TypeDecorator):
    """
    SQLAlchemy custom type for encrypted strings.

    Encrypts data before storing, decrypts after retrieving.
    Uses Fernet (AES-128-CBC + HMAC-SHA256).
    """
    impl = LargeBinary
    cache_ok = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load encryption key from environment
        encryption_key = os.environ.get("DATABASE_ENCRYPTION_KEY")
        if not encryption_key:
            raise ValueError("DATABASE_ENCRYPTION_KEY not set")
        self.cipher = Fernet(encryption_key.encode())

    def process_bind_param(self, value, dialect):
        """Encrypt before storing in database."""
        if value is None:
            return None
        # Encrypt and return bytes
        return self.cipher.encrypt(value.encode())

    def process_result_value(self, value, dialect):
        """Decrypt after retrieving from database."""
        if value is None:
            return None
        # Decrypt and return string
        return self.cipher.decrypt(value).decode()

class FuelProcurement(Base):
    """Fuel procurement record with encrypted sensitive fields."""
    __tablename__ = 'fuel_procurement'

    id = Column(String(50), primary_key=True)
    fuel_type = Column(String(20), nullable=False)  # Not encrypted (for queries)
    quantity_gallons = Column(String(20), nullable=False)  # Not encrypted

    # ✅ ENCRYPTED FIELDS
    price_per_gallon = Column(EncryptedString(200), nullable=False)  # Sensitive pricing
    supplier_id = Column(EncryptedString(200), nullable=False)  # Confidential supplier
    contract_reference = Column(EncryptedString(200))  # Confidential contract
    authorized_by = Column(EncryptedString(200), nullable=False)  # PII

    # Provenance hash (not encrypted - used for integrity verification)
    provenance_hash = Column(String(64), nullable=False, index=True)
```

**Data Loss Prevention (DLP):**

| Data Type | Detection Method | Action | Alert |
|-----------|------------------|--------|-------|
| API keys | Regex patterns (high entropy strings) | Block transmission, log | SOC team (PagerDuty) |
| Credit card numbers | Luhn algorithm | Block transmission, log | Security team |
| Social Security Numbers | Regex patterns | Block transmission, log | Security team |
| Fuel pricing (>$1M) | Threshold detection | Require approval | Manager approval |
| Proprietary contracts | Keyword detection | Block external email | Security team |

**Data Masking:**

```python
from typing import Any

def mask_sensitive_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Mask sensitive data for logging, API responses.

    Returns masked copy of data (original unchanged).
    """
    masked = data.copy()

    # Mask pricing data
    if 'price_per_gallon' in masked:
        masked['price_per_gallon'] = "***MASKED***"

    # Mask supplier info
    if 'supplier_id' in masked:
        masked['supplier_id'] = f"{masked['supplier_id'][:3]}***"

    # Mask authorization
    if 'authorized_by' in masked:
        name_parts = masked['authorized_by'].split()
        if len(name_parts) >= 2:
            masked['authorized_by'] = f"{name_parts[0]} {name_parts[1][0]}***"

    # Mask contract reference
    if 'contract_reference' in masked:
        masked['contract_reference'] = "***MASKED***"

    return masked

# Example usage in logging
import logging
logger = logging.getLogger(__name__)

def log_procurement_request(procurement_data: dict):
    """Log procurement request with sensitive data masked."""
    masked_data = mask_sensitive_data(procurement_data)
    logger.info(f"Procurement request: {masked_data}")
```

**Tokenization (for credit card data, if applicable):**

```python
import hashlib
import secrets
from typing import Tuple

class Tokenizer:
    """
    Tokenize sensitive data (one-way, irreversible).

    Use for credit card numbers, PII where original value
    not needed for business logic.
    """

    @staticmethod
    def tokenize(sensitive_value: str) -> str:
        """
        Generate token for sensitive value.

        Uses SHA-256 hash + random salt.
        """
        # Generate random salt
        salt = secrets.token_bytes(16)

        # Hash sensitive value with salt
        hash_obj = hashlib.sha256(salt + sensitive_value.encode())
        token = hash_obj.hexdigest()

        return token

    @staticmethod
    def format_masked_value(original: str, visible_chars: int = 4) -> str:
        """
        Format masked value (e.g., credit card: **** **** **** 1234).
        """
        if len(original) <= visible_chars:
            return "*" * len(original)

        masked = "*" * (len(original) - visible_chars)
        visible = original[-visible_chars:]

        return f"{masked}{visible}"

# Example usage
tokenizer = Tokenizer()

# Tokenize credit card (if used for fuel procurement payment)
cc_number = "4532-1234-5678-9010"
cc_token = tokenizer.tokenize(cc_number)
cc_masked = tokenizer.format_masked_value(cc_number.replace("-", ""), 4)

print(f"Token: {cc_token}")  # Store this in database
print(f"Masked: {cc_masked}")  # Display this to users
```

### 3.9 Layer 7: Governance & Compliance

**Security Governance Structure:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ Board of Directors                                                   │
│ - Oversight of cybersecurity risks                                   │
│ - Quarterly security briefings                                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────────┐
│ Chief Information Security Officer (CISO)                            │
│ - Security strategy and policy                                       │
│ - Risk management                                                    │
│ - Incident response leadership                                       │
│ - Board reporting                                                    │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────────┐
│                           │                                          │
│ ┌────────────────┐  ┌────┴─────────┐  ┌─────────────────────────┐  │
│ │ Security       │  │ Compliance   │  │ Privacy                 │  │
│ │ Architecture   │  │ Officer      │  │ Officer                 │  │
│ │ - Design       │  │ - Audits     │  │ - GDPR/CCPA            │  │
│ │ - Standards    │  │ - SOC 2      │  │ - Data rights          │  │
│ └────────────────┘  └──────────────┘  └─────────────────────────┘  │
│                                                                      │
│ ┌────────────────┐  ┌──────────────┐  ┌─────────────────────────┐  │
│ │ Security       │  │ Incident     │  │ Security                │  │
│ │ Operations     │  │ Response     │  │ Engineering             │  │
│ │ (SOC)          │  │ Team         │  │ - SAST/DAST            │  │
│ │ - Monitoring   │  │ - IR plan    │  │ - Vuln mgmt            │  │
│ │ - Alerts       │  │ - Forensics  │  │ - Pen testing          │  │
│ └────────────────┘  └──────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**Policy Framework:**

| Policy | Owner | Review Frequency | Last Review | Next Review |
|--------|-------|------------------|-------------|-------------|
| Master Security Policy | CISO | Annual | 2025-01-15 | 2026-01-15 |
| GL-011 Security Policy | Security Team | Quarterly | 2025-12-01 | 2026-03-01 |
| Incident Response Plan | IR Team Lead | Quarterly | 2025-11-01 | 2026-02-01 |
| Data Classification | Data Governance | Annual | 2025-06-01 | 2026-06-01 |
| Access Control Policy | IAM Team | Quarterly | 2025-11-15 | 2026-02-15 |
| Encryption Policy | Security Architect | Annual | 2025-03-01 | 2026-03-01 |
| Acceptable Use Policy | HR + Security | Annual | 2025-01-01 | 2026-01-01 |
| Third-Party Risk Management | Vendor Management | Annual | 2025-09-01 | 2026-09-01 |

**Security Training Program:**

| Training | Audience | Frequency | Compliance Requirement |
|----------|----------|-----------|------------------------|
| Security Awareness | All employees | Annual (+ quarterly phishing tests) | SOC 2, ISO 27001 |
| Secure Coding | Developers | Annual | Internal requirement |
| OWASP Top 10 | Developers, QA | Annual | Internal requirement |
| Incident Response | Security team, on-call | Quarterly (tabletop exercises) | SOC 2 |
| Data Privacy (GDPR/CCPA) | All employees handling PII | Annual | GDPR, CCPA |
| Financial Controls (SOX) | Finance, procurement teams | Annual | SOX |
| Role-Based Security Training | Admins, operators | Bi-annual | Internal requirement |

**Compliance Audit Schedule:**

| Audit | Type | Frequency | Auditor | Next Audit | Scope |
|-------|------|-----------|---------|------------|-------|
| SOC 2 Type II | External | Annual | Big 4 firm | 2026-Q2 | All GreenLang systems |
| ISO 27001 | External | Annual | Accredited body | 2026-Q3 | ISMS |
| ISO 50001 | External | Bi-annual | Accredited body | 2027-Q1 | Energy management |
| IEC 62443-4-2 | External | Bi-annual | ISA-certified | 2027-Q1 | Industrial automation security |
| Internal Security Audit | Internal | Quarterly | Internal audit team | 2026-Q1 | GL-011 FUELCRAFT |
| Penetration Test | External | Annual | Security firm | 2026-Q2 | All external-facing systems |
| Vulnerability Assessment | Internal | Monthly | Security team | Ongoing | All systems |
| Code Review | Internal | Every PR | Developers + security | Ongoing | All code changes |

**Risk Management:**

| Risk | Likelihood | Impact | Risk Score | Mitigation | Residual Risk |
|------|-----------|--------|------------|------------|---------------|
| Fuel price manipulation | Medium | Critical | HIGH | Multi-level approval, audit logging, anomaly detection | LOW |
| Inventory data tampering | Medium | High | MEDIUM | Immutable audit logs, provenance hashing, access controls | LOW |
| Supplier impersonation | Low | High | MEDIUM | Strong authentication, contract verification | VERY LOW |
| Ransomware attack | Medium | Critical | HIGH | Offline backups, EDR, network segmentation, training | MEDIUM |
| Insider threat (fraud) | Low | High | MEDIUM | Segregation of duties, audit logging, background checks | LOW |
| API key leakage | Medium | High | MEDIUM | Secret scanning, rotation, monitoring | LOW |
| Database breach | Low | Critical | MEDIUM | Encryption, access controls, monitoring, penetration testing | LOW |
| DDoS attack | High | Medium | MEDIUM | CDN, rate limiting, AWS Shield | VERY LOW |
| Supply chain attack | Low | High | MEDIUM | SBOM, dependency scanning, vendro