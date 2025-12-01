# GL-011 FUELCRAFT Threat Model

**Document Classification:** CONFIDENTIAL - SECURITY PLANNING
**Agent:** GL-011 FUELCRAFT - FuelManagementOrchestrator
**Threat Model Version:** 1.0.0
**Last Updated:** 2025-12-01
**Threat Modeling Framework:** STRIDE + PASTA
**Security Architects:** GreenLang Foundation Security Team

---

## Executive Summary

### Threat Modeling Overview

This threat model analyzes GL-011 FUELCRAFT FuelManagementOrchestrator using the **STRIDE threat classification** framework combined with **PASTA (Process for Attack Simulation and Threat Analysis)** methodology. The analysis identifies potential threats, attack vectors, and recommended mitigations to protect fuel procurement, inventory management, and lifecycle operations.

**STRIDE Categories:**
- **S**poofing - Identity impersonation
- **T**ampering - Unauthorized modification
- **R**epudiation - Denying actions
- **I**nformation Disclosure - Data leakage
- **D**enial of Service - Availability attacks
- **E**levation of Privilege - Unauthorized access escalation

### Threat Summary

| Threat Category | Total Threats | Critical | High | Medium | Low |
|----------------|---------------|----------|------|--------|-----|
| **Spoofing** | 8 | 1 | 2 | 3 | 2 |
| **Tampering** | 12 | 2 | 4 | 4 | 2 |
| **Repudiation** | 5 | 0 | 1 | 2 | 2 |
| **Information Disclosure** | 15 | 2 | 5 | 6 | 2 |
| **Denial of Service** | 10 | 1 | 3 | 4 | 2 |
| **Elevation of Privilege** | 7 | 1 | 2 | 3 | 1 |
| **TOTAL** | **57** | **7** | **17** | **22** | **11** |

### Risk Heat Map

```
         ┌─────────────────────────────────────────┐
         │          IMPACT (Consequence)            │
┌────────┼──────────┬──────────┬──────────┬─────────┤
│        │  LOW     │  MEDIUM  │  HIGH    │ CRITICAL│
├────────┼──────────┼──────────┼──────────┼─────────┤
│ VERY   │    0     │    2     │    3     │    1    │ 6
│ HIGH   │    ●     │    ●●    │   ●●●    │   ●     │
├────────┼──────────┼──────────┼──────────┼─────────┤
│ HIGH   │    3     │    8     │    9     │    6    │ 26
│        │   ●●●    │  ●●●●●●●●│ ●●●●●●●●●│  ●●●●●●│
├────────┼──────────┼──────────┼──────────┼─────────┤
│ MEDIUM │    5     │    7     │    3     │    0    │ 15
│        │  ●●●●●   │ ●●●●●●●  │   ●●●    │         │
├────────┼──────────┼──────────┼──────────┼─────────┤
│ LOW    │    3     │    5     │    2     │    0    │ 10
│        │   ●●●    │  ●●●●●   │    ●●    │         │
└────────┴──────────┴──────────┴──────────┴─────────┘
 Total:     11         22          17          7      57
```

**Risk Acceptance Threshold:** Medium and below (with mitigations)
**High-Risk Threats:** 24 (7 critical + 17 high) - **ALL REQUIRE MITIGATION**

---

## 1. System Architecture and Data Flow

### 1.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  External Actors                                                  │
│  - Fuel Procurement Manager                                       │
│  - Fuel Operators                                                 │
│  - Financial Auditors                                             │
│  - Fuel Suppliers (via API)                                       │
│  - ERP Systems                                                    │
└────────────────┬─────────────────────────────────────────────────┘
                 │ HTTPS / API Calls
                 ↓
┌──────────────────────────────────────────────────────────────────┐
│  Entry Points (Trust Boundary #1)                                │
│  - Load Balancer (AWS ALB)                                        │
│  - WAF (Web Application Firewall)                                 │
│  - DDoS Protection (Cloudflare)                                   │
│  - Rate Limiting                                                  │
└────────────────┬─────────────────────────────────────────────────┘
                 │ TLS 1.3
                 ↓
┌──────────────────────────────────────────────────────────────────┐
│  GL-011 FUELCRAFT Application Layer                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ API Endpoints                                                │ │
│  │ - /api/v1/fuel/procure      (POST) - Fuel procurement       │ │
│  │ - /api/v1/fuel/inventory    (GET)  - Inventory query        │ │
│  │ - /api/v1/fuel/quality      (POST) - Quality assurance      │ │
│  │ - /api/v1/fuel/optimize     (POST) - Cost optimization      │ │
│  │ - /api/v1/fuel/compliance   (GET)  - Compliance reports     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Security Controls                                            │ │
│  │ - Authentication (API Key + JWT)                             │ │
│  │ - Authorization (RBAC - 5 roles)                             │ │
│  │ - Input Validation (Pydantic)                                │ │
│  │ - Provenance Hashing (SHA-256)                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└────────────────┬─────────────────────────────────────────────────┘
                 │ Encrypted connections
                 ↓
┌──────────────────────────────────────────────────────────────────┐
│  Data Layer (Trust Boundary #2)                                  │
│  - PostgreSQL (Fuel data, contracts, transactions)               │
│  - Redis (Cache for performance)                                 │
│  - S3 (Reports, audit logs)                                      │
│  - Secrets Manager (API keys, credentials)                       │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Trust Boundaries

| Boundary | Description | Security Controls |
|----------|-------------|-------------------|
| **TB-1: Internet → DMZ** | External users/systems → Load Balancer | WAF, DDoS protection, TLS 1.3, rate limiting |
| **TB-2: DMZ → Application** | Load Balancer → GL-011 Pods | Authentication (API Key/JWT), Network Policies |
| **TB-3: Application → Data** | GL-011 Pods → Databases | Encryption (TLS), Access Control Lists, Audit Logging |
| **TB-4: Application → External APIs** | GL-011 → Supplier APIs, ERP | mTLS, Certificate Pinning, URL Validation |

### 1.3 Critical Data Flows

| Flow ID | Data Flow | Classification | Threats |
|---------|-----------|----------------|---------|
| **DF-001** | Procurement Manager → API → FUELCRAFT → DB | REGULATORY SENSITIVE | Spoofing, Tampering, Info Disclosure |
| **DF-002** | FUELCRAFT → Supplier API (pricing query) | CONFIDENTIAL | Info Disclosure, Spoofing, MitM |
| **DF-003** | FUELCRAFT → ERP System (PO creation) | REGULATORY SENSITIVE | Tampering, Repudiation, Info Disclosure |
| **DF-004** | FUELCRAFT → Database (inventory update) | REGULATORY SENSITIVE | Tampering, Repudiation, Info Disclosure |
| **DF-005** | Auditor → API → FUELCRAFT → Audit Logs | REGULATORY SENSITIVE | Info Disclosure, Repudiation |

---

## 2. STRIDE Threat Analysis

### 2.1 Spoofing Threats

#### Threat S-001: API Key Theft and Reuse

**Category:** Spoofing
**Risk Rating:** HIGH
**CVSS Score:** 8.1 (High)

**Description:**
Attacker steals API key (via phishing, malware, or network sniffing) and impersonates legitimate fuel procurement manager to create fraudulent procurement orders.

**Attack Scenario:**
1. Attacker sends phishing email to procurement manager
2. Manager clicks malicious link, malware steals API key from browser/disk
3. Attacker uses stolen API key to authenticate to FUELCRAFT
4. Attacker creates large fraudulent fuel procurement order ($500K+)
5. Fraudulent order processed, fuel delivered to attacker-controlled location

**Impact:**
- **Financial:** $500K+ loss per fraudulent order
- **Reputation:** Supplier trust damaged
- **Regulatory:** Audit trail compromised

**Likelihood:** MEDIUM (phishing attacks common, API keys if not rotated)

**Existing Controls:**
- API keys hashed (SHA-256) in database
- TLS 1.3 prevents network sniffing
- Audit logging tracks all API usage

**Recommended Mitigations:**

| Priority | Mitigation | Implementation | Cost | Effectiveness |
|----------|-----------|----------------|------|---------------|
| **HIGH** | API Key Rotation (90-day mandatory) | Automated rotation script | Low | High |
| **HIGH** | API Key IP Allowlisting | Restrict API keys to specific IPs | Low | High |
| **MEDIUM** | Multi-Factor Authentication for procurement | Add MFA for high-value operations | Medium | Very High |
| **MEDIUM** | Anomaly Detection | Alert on unusual procurement patterns | High | High |
| **LOW** | Security Awareness Training | Phishing awareness for managers | Low | Medium |

**Residual Risk (After Mitigation):** LOW

---

#### Threat S-002: Supplier Impersonation via API

**Category:** Spoofing
**Risk Rating:** MEDIUM
**CVSS Score:** 6.5 (Medium)

**Description:**
Attacker impersonates fuel supplier by compromising supplier API credentials or creating fake API endpoint.

**Attack Scenario:**
1. Attacker compromises supplier's API credentials
2. Attacker responds to FUELCRAFT price queries with artificially low prices
3. FUELCRAFT creates procurement order based on fake pricing
4. Attacker intercepts delivery or payment

**Impact:**
- **Financial:** $100K-500K loss
- **Operational:** Fuel delivery disruption
- **Compliance:** Contract violations

**Likelihood:** LOW (requires supplier compromise)

**Recommended Mitigations:**

| Priority | Mitigation | Implementation | Effectiveness |
|----------|-----------|----------------|---------------|
| **HIGH** | mTLS for Supplier API Calls | Client certificate authentication | Very High |
| **HIGH** | Certificate Pinning | Pin supplier certificates | High |
| **MEDIUM** | Supplier API Response Validation | Validate response signatures | High |
| **MEDIUM** | Price Reasonableness Checks | Alert on >10% price deviation | Medium |

**Residual Risk:** VERY LOW

---

#### Threat S-003: User Account Compromise

**Category:** Spoofing
**Risk Rating:** HIGH
**CVSS Score:** 7.8 (High)

**Description:**
Attacker compromises user credentials (username/password) and gains unauthorized access to FUELCRAFT system.

**Attack Scenario:**
1. Attacker obtains credentials via:
   - Credential stuffing (reused passwords from breaches)
   - Brute force attack (if no rate limiting)
   - Social engineering
2. Attacker authenticates as legitimate user
3. Attacker performs unauthorized actions within user's privilege level

**Impact:**
- **Confidentiality:** Access to fuel pricing, contracts
- **Integrity:** Unauthorized data modifications
- **Compliance:** Audit trail attribution incorrect

**Likelihood:** MEDIUM

**Existing Controls:**
- JWT tokens with 1-hour expiration
- Password complexity requirements
- Account lockout after 5 failed attempts

**Recommended Mitigations:**

| Priority | Mitigation | Implementation | Effectiveness |
|----------|-----------|----------------|---------------|
| **HIGH** | MFA for All Admin Accounts | TOTP or hardware tokens | Very High |
| **HIGH** | Rate Limiting on Login Endpoint | 5 attempts per 5 minutes | High |
| **MEDIUM** | Password Breach Detection | Check against HaveIBeenPwned | Medium |
| **MEDIUM** | Session Monitoring | Alert on suspicious login patterns | Medium |

**Residual Risk:** LOW

---

### 2.2 Tampering Threats

#### Threat T-001: Fuel Pricing Data Manipulation

**Category:** Tampering
**Risk Rating:** CRITICAL
**CVSS Score:** 9.1 (Critical)

**Description:**
Insider threat or attacker with database access modifies fuel pricing data to manipulate procurement costs, causing financial loss or competitive advantage.

**Attack Scenario:**
1. Malicious insider (procurement manager or database admin) accesses database
2. Directly modifies fuel pricing records in database
3. Subsequent procurement decisions based on manipulated pricing
4. Company overpays for fuel (if prices inflated) or violates contracts (if prices deflated)

**Impact:**
- **Financial:** $1M+ annual impact from manipulated pricing
- **Legal:** Contract breach, lawsuits
- **Fraud:** Criminal prosecution potential

**Likelihood:** LOW (requires privileged access, but high impact)

**Existing Controls:**
- Provenance hashing (SHA-256) for all transactions
- Audit logging with immutable storage
- Database encryption at rest
- Role-Based Access Control

**Recommended Mitigations:**

| Priority | Mitigation | Implementation | Cost | Effectiveness |
|----------|-----------|----------------|------|---------------|
| **CRITICAL** | Immutable Audit Logs (Write-Once) | S3 Object Lock, blockchain | Medium | Very High |
| **CRITICAL** | Segregation of Duties | Separate initiate/approve roles | Low | Very High |
| **CRITICAL** | Database Activity Monitoring (DAM) | Monitor direct DB modifications | High | High |
| **HIGH** | Pricing Change Approval Workflow | Require dual approval for price changes | Medium | High |
| **HIGH** | Real-Time Alerting | Alert SOC on price anomalies | Low | High |
| **MEDIUM** | Provenance Hash Chain Verification | Verify hash integrity hourly | Low | High |

**Residual Risk (After Mitigation):** LOW

---

#### Threat T-002: Inventory Data Tampering

**Category:** Tampering
**Risk Rating:** HIGH
**CVSS Score:** 8.2 (High)

**Description:**
Attacker modifies inventory levels to cover fuel theft or create false shortages/surpluses.

**Attack Scenario:**
1. Insider attacker (warehouse operator or DB admin) accesses inventory database
2. Modifies inventory levels:
   - Decrease levels to hide fuel theft
   - Increase levels to delay procurement (causing shortages)
   - Manipulate levels for fraudulent insurance claims
3. Real inventory discrepancies discovered during physical audit

**Impact:**
- **Financial:** Fuel theft ($50K-500K), insurance fraud
- **Operational:** Production downtime from fuel shortages
- **Safety:** Critical fuel shortages impact operations

**Likelihood:** MEDIUM (insider threat)

**Recommended Mitigations:**

| Priority | Mitigation | Implementation | Effectiveness |
|----------|-----------|----------------|---------------|
| **HIGH** | Automated Inventory Reconciliation | Compare DB vs. tank sensors daily | Very High |
| **HIGH** | Tamper-Evident Tank Sensors | Hardened IoT sensors with crypto signatures | High |
| **HIGH** | Blockchain Inventory Ledger | Immutable inventory transaction log | Very High |
| **MEDIUM** | Physical Security (Tank Farms) | Cameras, access controls, security guards | Medium |
| **MEDIUM** | Anomaly Detection | ML-based detection of unusual inventory changes | High |

**Residual Risk:** LOW

---

#### Threat T-003: API Request/Response Tampering (Man-in-the-Middle)

**Category:** Tampering
**Risk Rating:** MEDIUM
**CVSS Score:** 6.8 (Medium)

**Description:**
Attacker intercepts and modifies API requests/responses between FUELCRAFT and external systems (suppliers, ERP).

**Attack Scenario:**
1. Attacker performs MitM attack on network connection
2. Intercepts fuel price quote response from supplier API
3. Modifies price from $3.50/gallon to $2.50/gallon
4. FUELCRAFT creates procurement order based on falsified price
5. Actual delivery invoiced at $3.50/gallon, causing budget overrun

**Impact:**
- **Financial:** Budget overruns, contract disputes
- **Integrity:** Data integrity compromised

**Likelihood:** LOW (requires network compromise)

**Existing Controls:**
- TLS 1.3 encryption for all API calls
- Certificate validation

**Recommended Mitigations:**

| Priority | Mitigation | Implementation | Effectiveness |
|----------|-----------|----------------|---------------|
| **HIGH** | mTLS (Mutual TLS) | Bidirectional certificate authentication | Very High |
| **HIGH** | Certificate Pinning | Pin expected certificates | Very High |
| **MEDIUM** | API Response Signing | Supplier signs responses with private key | Very High |
| **MEDIUM** | Response Validation | Validate response signatures | High |

**Residual Risk:** VERY LOW

---

### 2.3 Repudiation Threats

#### Threat R-001: Procurement Order Repudiation

**Category:** Repudiation
**Risk Rating:** HIGH
**CVSS Score:** 7.5 (High)

**Description:**
User denies creating fraudulent or erroneous fuel procurement order, claiming account was compromised.

**Attack Scenario:**
1. Malicious user creates unauthorized procurement order
2. Order processed and fuel delivered
3. User denies creating order, claims account was hacked
4. Company unable to prove user accountability
5. Financial loss with no recourse

**Impact:**
- **Financial:** $100K-500K loss per order
- **Legal:** Contract disputes, potential lawsuits
- **Operational:** Trust in system eroded

**Likelihood:** LOW (requires malicious intent)

**Existing Controls:**
- Audit logging with timestamps, user IDs, IP addresses
- SHA-256 provenance hashing
- Immutable audit logs (S3 Object Lock)

**Recommended Mitigations:**

| Priority | Mitigation | Implementation | Effectiveness |
|----------|-----------|----------------|---------------|
| **HIGH** | Digital Signatures for High-Value Orders | PKI-based signing (private key) | Very High |
| **HIGH** | Multi-Factor Authentication | Require MFA for orders >$100K | Very High |
| **MEDIUM** | Video Recording of Order Creation | Screen recording for audit | Medium |
| **MEDIUM** | Email Confirmation + Review Period | 1-hour review window before execution | Medium |

**Residual Risk:** LOW

---

#### Threat R-002: Audit Log Deletion/Tampering

**Category:** Repudiation
**Risk Rating:** MEDIUM
**CVSS Score:** 6.5 (Medium)

**Description:**
Attacker with privileged access deletes or modifies audit logs to hide malicious activity.

**Attack Scenario:**
1. Attacker (insider with admin privileges) performs malicious action
2. Accesses audit log storage (S3, database)
3. Deletes or modifies audit log entries to remove evidence
4. Malicious activity undetectable

**Impact:**
- **Forensics:** Unable to investigate incidents
- **Compliance:** Regulatory violations (SOX, SOC 2)
- **Legal:** Spoliation of evidence

**Likelihood:** LOW (requires high-privilege access)

**Existing Controls:**
- Immutable audit logs (S3 Object Lock)
- Write-once storage
- Access controls on log storage

**Recommended Mitigations:**

| Priority | Mitigation | Implementation | Effectiveness |
|----------|-----------|----------------|---------------|
| **HIGH** | S3 Object Lock (Compliance Mode) | Enable compliance mode (cannot override) | Very High |
| **HIGH** | Blockchain Audit Trail | Tamper-proof ledger | Very High |
| **MEDIUM** | Real-Time SIEM Forwarding | Forward logs immediately to SIEM | High |
| **MEDIUM** | Log Integrity Monitoring | Hourly verification of log hashes | High |

**Residual Risk:** VERY LOW

---

### 2.4 Information Disclosure Threats

#### Threat I-001: Fuel Pricing Intelligence Leakage

**Category:** Information Disclosure
**Risk Rating:** CRITICAL
**CVSS Score:** 9.2 (Critical)

**Description:**
Unauthorized disclosure of fuel pricing data, supplier contracts, and procurement strategies to competitors.

**Attack Scenario:**
1. Attacker gains access to FUELCRAFT system via:
   - Compromised credentials
   - SQL injection (if present)
   - Insider threat (malicious employee)
2. Exfiltrates fuel pricing database
3. Sells pricing intelligence to competitors
4. Competitors undercut pricing, company loses contracts

**Impact:**
- **Financial:** $5M-10M annual loss from lost competitive advantage
- **Reputation:** Supplier trust damaged
- **Legal:** Potential lawsuits from suppliers

**Likelihood:** MEDIUM

**Existing Controls:**
- Encryption at rest (AES-256-GCM)
- Encryption in transit (TLS 1.3)
- Access controls (RBAC)
- Audit logging

**Recommended Mitigations:**

| Priority | Mitigation | Implementation | Cost | Effectiveness |
|----------|-----------|----------------|------|---------------|
| **CRITICAL** | Column-Level Encryption | Encrypt sensitive columns (pricing, supplier ID) | Medium | Very High |
| **CRITICAL** | Data Loss Prevention (DLP) | Monitor/block unauthorized data transfers | High | High |
| **HIGH** | Database Activity Monitoring | Alert on bulk exports | Medium | High |
| **HIGH** | Zero Trust Network Access | Verify every request, restrict lateral movement | High | Very High |
| **MEDIUM** | Data Masking for Non-Prod | Mask pricing in dev/test environments | Low | Medium |
| **MEDIUM** | Insider Threat Program | Monitor privileged user activity | High | Medium |

**Residual Risk (After Mitigation):** MEDIUM (due to insider threat risk)

---

#### Threat I-002: API Response Data Leakage

**Category:** Information Disclosure
**Risk Rating:** HIGH
**CVSS Score:** 7.8 (High)

**Description:**
API responses inadvertently expose sensitive data beyond what user is authorized to see.

**Attack Scenario:**
1. Attacker with low-privilege API access (fuel_viewer role)
2. Makes API request: GET /api/v1/fuel/inventory
3. API response includes sensitive fields not authorized for viewer:
   - Actual procurement costs (confidential)
   - Supplier IDs and contract terms (confidential)
   - Fuel quality test results (potentially confidential)
4. Attacker exfiltrates confidential data

**Impact:**
- **Confidentiality:** Exposure of pricing, suppliers, contracts
- **Competitive Advantage:** Intelligence leakage

**Likelihood:** MEDIUM (depends on API design)

**Existing Controls:**
- RBAC authorization checks
- Pydantic response models

**Recommended Mitigations:**

| Priority | Mitigation | Implementation | Effectiveness |
|----------|-----------|----------------|---------------|
| **HIGH** | Role-Based Response Filtering | Filter response fields by user role | Very High |
| **HIGH** | API Response Minimization | Return only required fields | High |
| **MEDIUM** | API Response Validation | Automated testing to detect over-exposure | Medium |
| **MEDIUM** | Data Sensitivity Tagging | Tag fields with sensitivity levels | Medium |

**Implementation Example:**

```python
# Pydantic response model with role-based filtering
class FuelInventoryResponse(BaseModel):
    fuel_type: str
    quantity_gallons: Decimal
    location: str

    # Sensitive fields (only for managers/admins)
    price_per_gallon: Optional[Decimal] = None
    supplier_id: Optional[str] = None
    contract_reference: Optional[str] = None

    @classmethod
    def from_db(cls, db_row, user_role: str):
        """Create response with role-based filtering."""
        data = {
            "fuel_type": db_row.fuel_type,
            "quantity_gallons": db_row.quantity_gallons,
            "location": db_row.location
        }

        # Include sensitive fields only for authorized roles
        if user_role in ["fuel_manager", "fuel_admin"]:
            data["price_per_gallon"] = db_row.price_per_gallon
            data["supplier_id"] = db_row.supplier_id
            data["contract_reference"] = db_row.contract_reference

        return cls(**data)
```

**Residual Risk:** LOW

---

#### Threat I-003: Verbose Error Messages Expose Internal Details

**Category:** Information Disclosure
**Risk Rating:** MEDIUM
**CVSS Score:** 5.3 (Medium)

**Description:**
Error messages expose sensitive internal details (file paths, database schema, software versions) that aid attackers in reconnaissance.

**Attack Scenario:**
1. Attacker sends malformed API request
2. Application returns detailed error with stack trace:
   ```
   {
     "error": "Database connection failed",
     "detail": "File \"/app/fuel_management_orchestrator.py\", line 512, in procure_fuel\n
                postgresql://fuelcraft_user:***@db.internal.greenlang.io:5432/fuelcraft_db"
   }
   ```
3. Attacker learns:
   - Application file structure (/app/fuel_management_orchestrator.py)
   - Database hostname (db.internal.greenlang.io)
   - Database name (fuelcraft_db)
   - Username (fuelcraft_user)
4. Attacker uses information to plan targeted attack

**Impact:**
- **Reconnaissance:** Aids attacker reconnaissance
- **Attack Surface:** Reveals internal architecture

**Likelihood:** HIGH (common misconfiguration)

**Recommended Mitigations:**

| Priority | Mitigation | Implementation | Effectiveness |
|----------|-----------|----------------|---------------|
| **HIGH** | Generic Error Messages (Production) | Return "Internal Server Error" only | Very High |
| **HIGH** | Disable Debug Mode (Production) | FastAPI debug=False | Very High |
| **MEDIUM** | Centralized Error Handling | Custom exception handlers | High |
| **MEDIUM** | Error Logging (Server-Side Only) | Log details server-side, not in response | High |

**Residual Risk:** VERY LOW

---

### 2.5 Denial of Service Threats

#### Threat D-001: API Rate Limit Bypass / Resource Exhaustion

**Category:** Denial of Service
**Risk Rating:** HIGH
**CVSS Score:** 7.5 (High)

**Description:**
Attacker floods API endpoints with requests, exhausting server resources and causing service degradation or outage.

**Attack Scenario:**
1. Attacker identifies API endpoint without rate limiting
2. Launches automated attack:
   - 100,000 requests per minute to /api/v1/fuel/procure
   - Each request triggers expensive database query and calculation
3. Server CPU/memory exhausted
4. Legitimate users unable to access service
5. Fuel operations disrupted

**Impact:**
- **Availability:** Service outage (99.95% SLA violated)
- **Financial:** Lost productivity, SLA penalties
- **Operational:** Fuel procurement delays, potential shortages

**Likelihood:** HIGH (if no rate limiting)

**Current State:** **VULNERABLE** (no rate limiting implemented)

**Recommended Mitigations:**

| Priority | Mitigation | Implementation | Cost | Effectiveness |
|----------|-----------|----------------|------|---------------|
| **CRITICAL** | API Rate Limiting | Implement rate limits per endpoint/user | Low | Very High |
| **HIGH** | Request Throttling | Gradual backoff for excessive requests | Low | High |
| **HIGH** | Resource Limits (K8s) | CPU/memory limits on pods | Low | High |
| **MEDIUM** | WAF Rules | Block malicious traffic patterns | Medium | High |
| **MEDIUM** | CDN/DDoS Protection | Cloudflare Enterprise | High | Very High |

**Recommended Rate Limits:**

| Endpoint | Rate Limit | Justification |
|----------|------------|---------------|
| /api/v1/fuel/procure | 10/minute per API key | High-cost operation |
| /api/v1/fuel/inventory | 60/minute per API key | Moderate-cost query |
| /api/v1/fuel/quality | 30/minute per API key | Database-intensive |
| /api/v1/fuel/optimize | 5/minute per API key | CPU-intensive calculation |
| /health, /metrics | 100/minute per IP | Monitoring endpoints |

**Residual Risk (After Mitigation):** LOW

---

#### Threat D-002: Database Connection Exhaustion

**Category:** Denial of Service
**Risk Rating:** MEDIUM
**CVSS Score:** 6.5 (Medium)

**Description:**
Attacker consumes all database connections, preventing legitimate requests from accessing database.

**Attack Scenario:**
1. Attacker sends requests that trigger long-running database queries
2. Database connection pool (max 100 connections) exhausted
3. Subsequent requests fail with "connection pool exhausted" error
4. Service degraded or unavailable

**Impact:**
- **Availability:** Service degradation
- **User Experience:** Timeout errors for users

**Likelihood:** MEDIUM

**Recommended Mitigations:**

| Priority | Mitigation | Implementation | Effectiveness |
|----------|-----------|----------------|---------------|
| **HIGH** | Connection Pooling with Limits | Configure max connections per app instance | High |
| **HIGH** | Query Timeouts | Timeout slow queries (10 seconds) | High |
| **MEDIUM** | Connection Health Checks | Remove stale connections | Medium |
| **MEDIUM** | Database Query Optimization | Index optimization, query profiling | High |

**Residual Risk:** LOW

---

### 2.6 Elevation of Privilege Threats

#### Threat E-001: RBAC Bypass via Authorization Logic Flaw

**Category:** Elevation of Privilege
**Risk Rating:** HIGH
**CVSS Score:** 8.1 (High)

**Description:**
Attacker with low-privilege role (fuel_viewer) exploits flaw in authorization logic to perform high-privilege actions (fuel procurement, price changes).

**Attack Scenario:**
1. Attacker authenticated as fuel_viewer (read-only role)
2. Discovers authorization check has logic flaw:
   ```python
   # ❌ VULNERABLE CODE
   if user_role == "fuel_admin" or "fuel_manager":  # Bug: always true!
       allow_procurement()
   ```
3. Attacker able to create procurement orders despite viewer role
4. Creates fraudulent orders

**Impact:**
- **Integrity:** Unauthorized data modifications
- **Financial:** Fraudulent procurement ($100K-500K)

**Likelihood:** LOW (requires code defect)

**Recommended Mitigations:**

| Priority | Mitigation | Implementation | Effectiveness |
|----------|-----------|----------------|---------------|
| **HIGH** | Code Review (Security Focus) | Review all RBAC logic | Very High |
| **HIGH** | Automated RBAC Testing | Unit tests for all role/endpoint combinations | Very High |
| **HIGH** | Centralized Authorization | Decorator-based authorization (single source of truth) | Very High |
| **MEDIUM** | Penetration Testing | Annual pen test focusing on authz | High |

**Secure Implementation:**

```python
# ✅ SECURE CODE
from fastapi import Depends, HTTPException

def require_role(allowed_roles: List[str]):
    async def role_checker(user: User = Depends(get_current_user)):
        if user.role not in allowed_roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return role_checker

@app.post("/api/v1/fuel/procure")
async def procure_fuel(
    request: ProcurementRequest,
    user: User = Depends(require_role(["fuel_admin", "fuel_manager"]))
):
    # Only admins and managers can access this endpoint
    ...
```

**Residual Risk:** VERY LOW

---

#### Threat E-002: Container Escape to Host System

**Category:** Elevation of Privilege
**Risk Rating:** CRITICAL
**CVSS Score:** 9.3 (Critical)

**Description:**
Attacker exploits container vulnerability to escape to host system, gaining root access to Kubernetes node.

**Attack Scenario:**
1. Attacker exploits application vulnerability to execute code in FUELCRAFT container
2. Container running as root or with excessive capabilities
3. Exploits kernel vulnerability (e.g., dirty pipe, dirtycow) to escape container
4. Gains root access to Kubernetes worker node
5. Pivots to other pods, accesses secrets, exfiltrates data

**Impact:**
- **Confidentiality:** Access to all secrets, data on node
- **Integrity:** Modify any pod on node
- **Availability:** Crash node, delete pods

**Likelihood:** LOW (requires vulnerability chain)

**Existing Controls:**
- Non-root container (UID 10011)
- Read-only root filesystem
- Capabilities dropped (ALL)
- Seccomp profile (RuntimeDefault)

**Recommended Mitigations:**

| Priority | Mitigation | Implementation | Effectiveness |
|----------|-----------|----------------|---------------|
| **CRITICAL** | Pod Security Standards (Restricted) | Enforce restricted PSS | Very High |
| **HIGH** | Non-Root Container | runAsNonRoot: true | Very High |
| **HIGH** | Read-Only Root Filesystem | readOnlyRootFilesystem: true | High |
| **HIGH** | Drop All Capabilities | capabilities: drop: [ALL] | High |
| **MEDIUM** | Runtime Security (Falco) | Detect escape attempts | High |
| **MEDIUM** | Node Isolation | Dedicated node pools for sensitive workloads | Medium |

**Current State:** ✅ **MITIGATED** (all critical controls in place)

**Residual Risk:** VERY LOW

---

## 3. Attack Tree Analysis

### 3.1 Attack Goal: Financial Fraud via Fuel Procurement Manipulation

```
┌─────────────────────────────────────────────────────────────────┐
│ GOAL: Steal $500K via Fraudulent Fuel Procurement              │
│ (Attacker profit from fake procurement orders)                  │
└────────────────┬────────────────────────────────────────────────┘
                 │
     ┌───────────┴────────────┐
     │                        │
     ▼                        ▼
┌─────────────────┐   ┌──────────────────┐
│ Path 1:         │   │ Path 2:          │
│ Compromise API  │   │ Insider Threat   │
│ Credentials     │   │ (Malicious       │
│                 │   │  Employee)       │
└────────┬────────┘   └────────┬─────────┘
         │                     │
    ┌────┴────┐           ┌────┴────┐
    │         │           │         │
    ▼         ▼           ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Phishing│ │Brute   │ │Bribe   │ │Coercion│
│Manager │ │Force   │ │Employee│ │Employee│
│        │ │Login   │ │        │ │        │
└───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
    │          │          │          │
    └──────┬───┴──────────┴──────────┘
           │
           ▼
    ┌──────────────────┐
    │ Authenticate to  │
    │ FUELCRAFT API    │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Create Fraudulent│
    │ Procurement Order│
    │ ($500K, attacker-│
    │  controlled addr)│
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Bypass Approval  │
    │ (if required)    │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Order Executed   │
    │ Fuel Delivered   │
    │ Payment Sent     │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Attacker Profit: │
    │ $500K            │
    └──────────────────┘
```

**Attack Path Analysis:**

| Step | Attack Technique | Likelihood | Impact | Mitigations |
|------|-----------------|------------|--------|-------------|
| 1. Compromise Credentials | Phishing, brute force | MEDIUM | N/A | MFA, rate limiting, training |
| 2. Authenticate to API | API key reuse | HIGH (if step 1 succeeds) | N/A | API key rotation, IP allowlisting |
| 3. Create Fraudulent Order | Unauthorized procurement | HIGH | $500K loss | RBAC, approval workflow |
| 4. Bypass Approval | Authorization flaw | LOW | N/A | Segregation of duties, dual approval |
| 5. Order Executed | Automatic execution | HIGH | $500K loss | Fraud detection, anomaly detection |

**Overall Attack Likelihood:** LOW-MEDIUM (due to multiple mitigations)

**Recommended Priority Mitigations:**
1. **MFA for procurement managers** (blocks step 1)
2. **Dual approval for orders >$100K** (blocks step 4)
3. **Anomaly detection for unusual orders** (detects step 5)

---

### 3.2 Attack Goal: Data Exfiltration (Fuel Pricing Intelligence)

```
┌─────────────────────────────────────────────────────────────────┐
│ GOAL: Exfiltrate Fuel Pricing Database                         │
│ (Sell to competitors for profit)                                │
└────────────────┬────────────────────────────────────────────────┘
                 │
     ┌───────────┴────────────┐
     │                        │
     ▼                        ▼
┌─────────────────┐   ┌──────────────────┐
│ Path 1:         │   │ Path 2:          │
│ Application     │   │ Database Direct  │
│ Exploit         │   │ Access           │
└────────┬────────┘   └────────┬─────────┘
         │                     │
    ┌────┴────┐           ┌────┴────┐
    │         │           │         │
    ▼         ▼           ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│SQL     │ │API     │ │Stolen  │ │Insider │
│Injection│ │Over-   │ │DB      │ │Threat  │
│        │ │exposure│ │Creds   │ │        │
└───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
    │          │          │          │
    └──────┬───┴──────────┴──────────┘
           │
           ▼
    ┌──────────────────┐
    │ Access Pricing   │
    │ Database         │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Exfiltrate Data  │
    │ (avoid DLP)      │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Sell to          │
    │ Competitors      │
    │ ($100K profit)   │
    └──────────────────┘
```

**Attack Path Analysis:**

| Step | Attack Technique | Likelihood | Impact | Mitigations |
|------|-----------------|------------|--------|-------------|
| 1. Application Exploit | SQL injection, API over-exposure | LOW | N/A | Input validation, RBAC |
| 2. Access Database | Direct DB access | VERY LOW | N/A | DB firewall, access controls |
| 3. Exfiltrate Data | Bulk export, API abuse | MEDIUM | $5M-10M competitive loss | DLP, DAM, rate limiting |

**Overall Attack Likelihood:** LOW (due to defense in depth)

**Recommended Priority Mitigations:**
1. **Data Loss Prevention (DLP)** (blocks step 3)
2. **Database Activity Monitoring (DAM)** (detects step 2/3)
3. **API response filtering** (reduces step 1 success)

---

## 4. Recommended Security Controls (Prioritized)

### 4.1 Critical Priority (Implement Immediately)

| Control ID | Control Name | Threats Addressed | Implementation Cost | ROI |
|------------|-------------|-------------------|-------------------|-----|
| **C-001** | API Rate Limiting | D-001 (DoS) | Low | Very High |
| **C-002** | Segregation of Duties (Dual Approval) | T-001 (Price Manipulation) | Low | Very High |
| **C-003** | Immutable Audit Logs (S3 Object Lock) | R-002 (Log Tampering), T-001 | Medium | Very High |
| **C-004** | Column-Level Encryption (Pricing Data) | I-001 (Pricing Leakage) | Medium | Very High |
| **C-005** | Data Loss Prevention (DLP) - Blocking Mode | I-001 (Data Exfiltration) | High | High |

**Implementation Timeline:** 0-30 days

---

### 4.2 High Priority (Implement within 90 days)

| Control ID | Control Name | Threats Addressed | Implementation Cost | ROI |
|------------|-------------|-------------------|---------------------|-----|
| **C-006** | MFA for High-Value Operations | S-001, S-003, R-001 | Low | High |
| **C-007** | mTLS for Supplier APIs | S-002, T-003 | Medium | High |
| **C-008** | Database Activity Monitoring (DAM) | T-001, T-002, I-001 | High | High |
| **C-009** | Role-Based API Response Filtering | I-002 | Low | High |
| **C-010** | Automated Inventory Reconciliation | T-002 | Medium | High |

**Implementation Timeline:** 30-90 days

---

### 4.3 Medium Priority (Implement within 180 days)

| Control ID | Control Name | Threats Addressed | Implementation Cost | ROI |
|------------|-------------|-------------------|---------------------|-----|
| **C-011** | Anomaly Detection (ML-based) | S-001, T-001, T-002, I-001 | High | Medium |
| **C-012** | Insider Threat Program | T-001, T-002, I-001 | High | Medium |
| **C-013** | API Digital Signatures (High-Value) | R-001 | Medium | Medium |
| **C-014** | Blockchain Inventory Ledger | T-002 | Very High | Medium |
| **C-015** | Zero Trust Network Access (ZTNA) | I-001, E-002 | Very High | Medium |

**Implementation Timeline:** 90-180 days

---

## 5. Residual Risk Assessment

### 5.1 Risk After Mitigation

| Threat Category | Original Risk | Residual Risk | Status |
|----------------|---------------|---------------|--------|
| **Spoofing** | HIGH | LOW | ✅ ACCEPTABLE |
| **Tampering** | CRITICAL | LOW | ✅ ACCEPTABLE |
| **Repudiation** | HIGH | LOW | ✅ ACCEPTABLE |
| **Information Disclosure** | CRITICAL | MEDIUM | ⚠️ MONITOR |
| **Denial of Service** | HIGH | LOW | ✅ ACCEPTABLE |
| **Elevation of Privilege** | HIGH | VERY LOW | ✅ ACCEPTABLE |

**Overall Residual Risk:** **LOW-MEDIUM** (acceptable with continuous monitoring)

### 5.2 Accepted Risks

| Risk ID | Risk Description | Residual Risk | Justification | Monitoring |
|---------|------------------|---------------|---------------|------------|
| **AR-001** | Insider Threat (Malicious Employee) | MEDIUM | Cost of 100% prevention (e.g., background checks, surveillance) exceeds benefit | Audit log monitoring, anomaly detection |
| **AR-002** | Zero-Day Vulnerability Exploitation | LOW | Unpredictable, mitigated by defense in depth | Vulnerability scanning, patch management |
| **AR-003** | Advanced Persistent Threat (APT) | LOW | Nation-state actors beyond scope | Security monitoring, incident response |

---

## 6. Security Monitoring and Detection

### 6.1 Security Monitoring Strategy

| Threat | Detection Method | Alert Threshold | Response |
|--------|------------------|----------------|----------|
| **S-001: API Key Theft** | Failed authentication spike, unusual IP | >5 failed auth from new IP | Lock account, notify user |
| **T-001: Price Manipulation** | Direct DB modification, price anomaly | Any direct DB write, >10% price change | Alert SOC, freeze pricing |
| **T-002: Inventory Tampering** | Inventory discrepancy, sensor vs. DB | >5% discrepancy | Alert ops, physical audit |
| **I-001: Data Exfiltration** | Large data transfers, bulk exports | >1000 records in 1 hour | DLP block, alert SOC |
| **D-001: DoS Attack** | Request rate spike, CPU/memory exhaustion | >100 req/min from single IP | Rate limit, WAF block |
| **E-001: RBAC Bypass** | Low-privilege user performs high-privilege action | Any unauthorized action | Alert SOC, lock account |

### 6.2 SIEM Alert Rules

**Example Splunk Query (Price Manipulation Detection):**

```spl
index=fuelcraft sourcetype=audit_log action="update_fuel_price"
| eval price_change_pct = (new_price - old_price) / old_price * 100
| where abs(price_change_pct) > 10
| table _time, user, fuel_type, old_price, new_price, price_change_pct
| sendmail to="soc@greenlang.io" subject="ALERT: Unusual Fuel Price Change Detected"
```

---

## 7. Conclusion

### 7.1 Threat Model Summary

GL-011 FUELCRAFT faces **57 identified threats** across STRIDE categories:
- **7 Critical** threats (primarily tampering and information disclosure)
- **17 High** threats (spanning all categories)
- **22 Medium** threats
- **11 Low** threats

**Key Findings:**
1. **Financial fraud** is the highest-impact threat (price manipulation, fraudulent procurement)
2. **Data exfiltration** (fuel pricing intelligence) poses significant competitive risk
3. **Denial of Service** is the most likely attack (if rate limiting not implemented)
4. **Insider threats** are difficult to fully mitigate

**Overall Risk Posture:** **MEDIUM** (currently), **LOW** (after priority mitigations)

### 7.2 Next Steps

1. **Immediate (0-30 days):**
   - Implement API rate limiting (C-001)
   - Enable DLP blocking mode (C-005)
   - Implement dual approval for high-value orders (C-002)

2. **Short-Term (30-90 days):**
   - Deploy MFA for high-value operations (C-006)
   - Implement Database Activity Monitoring (C-008)
   - Enable mTLS for supplier APIs (C-007)

3. **Long-Term (90-180 days):**
   - Deploy ML-based anomaly detection (C-011)
   - Establish insider threat program (C-012)
   - Implement Zero Trust Network Access (C-015)

4. **Continuous:**
   - Quarterly threat model updates
   - Annual penetration testing
   - Monthly security metrics review
   - Ongoing security awareness training

---

## Appendix A: Threat Taxonomy

### A.1 STRIDE Threat Categories

| Category | Description | Focus |
|----------|-------------|-------|
| **Spoofing** | Pretending to be something or someone else | Authentication |
| **Tampering** | Modifying data or code | Integrity |
| **Repudiation** | Claiming not to have performed an action | Non-repudiation |
| **Information Disclosure** | Exposing information to unauthorized parties | Confidentiality |
| **Denial of Service** | Making system unavailable | Availability |
| **Elevation of Privilege** | Gaining unauthorized capabilities | Authorization |

### A.2 Risk Rating Matrix

| Likelihood | Impact: LOW | Impact: MEDIUM | Impact: HIGH | Impact: CRITICAL |
|-----------|------------|----------------|--------------|------------------|
| **VERY HIGH** | MEDIUM | HIGH | CRITICAL | CRITICAL |
| **HIGH** | LOW | MEDIUM | HIGH | CRITICAL |
| **MEDIUM** | LOW | MEDIUM | HIGH | HIGH |
| **LOW** | VERY LOW | LOW | MEDIUM | MEDIUM |

---

## Appendix B: References

- **STRIDE Threat Modeling:** Microsoft Security Development Lifecycle
- **PASTA Methodology:** Risk Centric Threat Modeling (VerSprite)
- **ATT&CK Framework:** MITRE ATT&CK for Enterprise
- **NIST Cybersecurity Framework:** NIST CSF v1.1
- **IEC 62443-4-2:** Industrial Automation Security Requirements
- **OWASP Top 10:** Web Application Security Risks (2021)
- **CWE Top 25:** Most Dangerous Software Weaknesses

---

**END OF THREAT MODEL**

*This threat model is a living document and should be reviewed quarterly or after significant system changes.*

*Next Review Date: March 1, 2026*
