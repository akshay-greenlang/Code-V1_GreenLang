# POL-008: Asset Management Policy

## Document Control

| Field | Value |
|-------|-------|
| Policy ID | POL-008 |
| Version | 1.0 |
| Effective Date | 2026-02-06 |
| Last Review | 2026-02-06 |
| Next Review | 2027-02-06 |
| Owner | Chief Technology Officer (CTO) |
| Approver | Chief Financial Officer (CFO) |
| Classification | Internal |

---

## 1. Purpose

This Asset Management Policy establishes requirements for identifying, classifying, tracking, maintaining, and disposing of GreenLang's information technology assets throughout their lifecycle. Effective asset management ensures that:

- All IT assets are inventoried and accounted for
- Asset ownership and accountability are clearly assigned
- Assets are properly secured based on their criticality
- Licensing compliance is maintained to avoid legal and financial risk
- End-of-life assets are disposed of securely
- Accurate records support financial, operational, and compliance needs
- Security risks from unmanaged assets are minimized

---

## 2. Scope

### 2.1 Applicability

This policy applies to:
- All GreenLang employees responsible for IT assets
- IT Operations, Security, Finance, and Procurement teams
- Managers approving asset requests and allocations
- Third parties managing GreenLang assets

### 2.2 Covered Assets

This policy covers the following asset categories:

#### 2.2.1 Hardware Assets
- End-user devices (laptops, desktops, monitors)
- Mobile devices (smartphones, tablets)
- Servers (physical and virtual)
- Network equipment (routers, switches, firewalls, access points)
- Storage devices (NAS, SAN, external drives)
- Peripheral devices (printers, scanners, docking stations)
- Security devices (HSMs, smart cards, tokens)

#### 2.2.2 Software Assets
- Operating systems and licenses
- Commercial software (Microsoft, Adobe, etc.)
- Development tools and IDEs
- Security software (antivirus, EDR, SIEM)
- SaaS subscriptions
- Open source software in use
- Internally developed applications

#### 2.2.3 Cloud Assets
- Cloud compute instances (EC2, ECS, Lambda)
- Cloud storage (S3, EBS, RDS)
- Cloud networking (VPCs, load balancers, CDN)
- Managed services (ElastiCache, OpenSearch, etc.)
- Cloud accounts and subscriptions

#### 2.2.4 Data Assets
- Databases and data stores
- File shares and document repositories
- Backup systems and archives
- Data pipelines and ETL processes
- Machine learning models and datasets

### 2.3 Exclusions

The following are outside the scope of this policy:
- Personal devices not accessing company resources
- Office furniture and non-IT equipment
- Building infrastructure (HVAC, power)

---

## 3. Policy Statement

### 3.1 Asset Inventory Requirements

#### 3.1.1 Comprehensive Inventory

GreenLang shall maintain a comprehensive, accurate inventory of all IT assets including:
- Asset identifier (unique ID or tag number)
- Asset type and category
- Make, model, and specifications
- Serial number and MAC address (hardware)
- License keys and entitlements (software)
- Physical location or cloud region
- Assignment (user, department, service)
- Owner and custodian
- Acquisition date and cost
- Warranty and support status
- Classification (criticality and data sensitivity)
- Lifecycle status (active, maintenance, retired)

#### 3.1.2 Hardware Inventory

Hardware assets must be recorded with:
- Physical asset tag affixed to device
- Serial number verified at receipt
- Firmware/BIOS version documented
- Network configuration (IP, MAC, hostname)
- Security baseline configuration status
- Warranty expiration date
- Assigned user or service

#### 3.1.3 Software Inventory

Software assets must include:
- Software name and publisher
- Version and patch level
- License type (perpetual, subscription, open source)
- License quantity and usage
- Installation locations
- License key or activation details
- Renewal date (subscriptions)
- End-of-support date

#### 3.1.4 Cloud Asset Inventory

Cloud resources must be tracked with:
- Resource ID (ARN, resource name)
- Resource type and size
- Region and availability zone
- Account and project association
- Cost allocation tags
- Auto-discovery integration
- Security group assignments
- Data classification tags

#### 3.1.5 Data Asset Inventory

Data assets must document:
- Data store name and location
- Data classification level
- Data owner and steward
- Retention requirements
- Backup status
- Access control configuration
- Regulatory applicability (GDPR, etc.)

### 3.2 Asset Classification by Criticality

#### 3.2.1 Criticality Levels

| Level | Definition | Recovery Priority | Examples |
|-------|------------|-------------------|----------|
| **Critical** | Essential for core operations; outage causes immediate business impact | RTO <4 hours | Production databases, auth services, API gateways |
| **High** | Important for daily operations; outage significantly impacts productivity | RTO <8 hours | Development servers, CI/CD, internal tools |
| **Medium** | Supports business functions; outage causes moderate inconvenience | RTO <24 hours | Test environments, documentation systems |
| **Low** | Minimal impact if unavailable; nice to have | RTO <72 hours | Archive storage, training systems |

#### 3.2.2 Classification Criteria

Assets are classified based on:
- Business process dependency
- Data sensitivity (per POL-002)
- User population affected
- Revenue impact of unavailability
- Regulatory or compliance requirements
- Replacement difficulty

#### 3.2.3 Classification Review

- Initial classification at asset acquisition
- Review classification annually
- Reclassify when business use changes
- Document classification rationale

### 3.3 Asset Ownership Assignment

#### 3.3.1 Asset Owner

Every asset must have a designated owner responsible for:
- Asset classification decisions
- Access authorization
- Data protection requirements
- Lifecycle decisions (upgrade, retire)
- Budget and cost management
- Compliance with applicable policies

#### 3.3.2 Asset Custodian

Day-to-day management assigned to custodian:
- Physical security of hardware
- Configuration and maintenance
- Patch management
- Performance monitoring
- Issue escalation to owner

#### 3.3.3 Ownership Assignment Rules

| Asset Type | Default Owner | Default Custodian |
|------------|---------------|-------------------|
| End-user device | Assigned user | IT Operations |
| Server/infrastructure | Service owner | Platform Engineering |
| Cloud resources | Service owner | Cloud Operations |
| Software license | Requesting manager | IT Operations |
| Data asset | Data owner (business) | Data Engineering |

### 3.4 Procurement Procedures

#### 3.4.1 Asset Request Process

1. **Request Submission:** User submits request via IT Service Portal
2. **Manager Approval:** Direct manager approves business need
3. **Budget Approval:** Finance approves if >$5,000 or unbudgeted
4. **Security Review:** Security reviews if sensitive data or critical system
5. **Procurement:** IT Procurement sources from approved vendors
6. **Receipt:** Asset received, verified, and recorded in inventory
7. **Configuration:** Security baseline applied before deployment
8. **Deployment:** Asset deployed to user or service
9. **Documentation:** Inventory updated with all details

#### 3.4.2 Approved Vendors

Assets must be procured from approved vendors:
- Hardware: Dell, Apple, Lenovo (pre-negotiated contracts)
- Cloud: AWS (primary), Azure (identity), GCP (ML workloads)
- Software: Through IT for license compliance
- Security products: Approved by Security team

New vendors require:
- Security assessment (POL-004)
- Contract review by Legal
- Finance approval

#### 3.4.3 Procurement Documentation

Maintain for each procurement:
- Purchase order and invoice
- Vendor quotes and contracts
- License agreements
- Warranty documentation
- Delivery confirmation

### 3.5 Asset Labeling and Tracking

#### 3.5.1 Physical Asset Tags

All hardware assets shall have:
- Unique asset tag (barcode/RFID)
- Tag format: GL-[CATEGORY]-[YEAR]-[SEQUENCE]
  - Example: GL-LAP-2026-0142
- Tag affixed in standard location
- Tag linked to inventory record

#### 3.5.2 Asset Categories

| Code | Category |
|------|----------|
| LAP | Laptops |
| DES | Desktops |
| MON | Monitors |
| MOB | Mobile devices |
| SRV | Servers |
| NET | Network equipment |
| PER | Peripherals |
| SEC | Security devices |

#### 3.5.3 Tracking Requirements

- Barcode/RFID scanning at receiving
- Location updates when assets move
- User assignment tracking
- Regular physical inventory reconciliation
- Lost/stolen assets reported within 24 hours

#### 3.5.4 Cloud Resource Tagging

All cloud resources must have tags:
- `Environment`: prod, staging, dev
- `Service`: service name
- `Owner`: team or individual email
- `CostCenter`: budget allocation code
- `DataClassification`: per POL-002
- `CreatedDate`: ISO date
- `ExpirationDate`: if temporary

### 3.6 Maintenance and Update Schedules

#### 3.6.1 Hardware Maintenance

| Activity | Frequency | Responsibility |
|----------|-----------|----------------|
| Firmware updates | Quarterly | IT Operations |
| Hardware diagnostics | Semi-annual | IT Operations |
| Battery replacement | As needed | IT Operations |
| Cleaning/inspection | Annual | IT Operations |
| Warranty verification | Annual | IT Procurement |

#### 3.6.2 Software Maintenance

| Activity | Frequency | Responsibility |
|----------|-----------|----------------|
| Security patches | Per patch policy | IT Security |
| Minor updates | Monthly | IT Operations |
| Major upgrades | Per release schedule | Application teams |
| License renewal | Before expiration | IT Procurement |
| End-of-life planning | 6 months before EOL | IT Operations |

#### 3.6.3 Cloud Resource Maintenance

| Activity | Frequency | Responsibility |
|----------|-----------|----------------|
| Instance patching | Per patch policy | Cloud Operations |
| Cost optimization review | Monthly | FinOps |
| Unused resource cleanup | Weekly | Cloud Operations |
| Security group review | Quarterly | Cloud Security |
| Reserved capacity review | Annual | FinOps |

### 3.7 Secure Configuration Baselines

#### 3.7.1 Baseline Requirements

All assets must be configured to security baselines before deployment:
- Operating system hardening (CIS benchmarks)
- Endpoint protection installed and active
- Disk encryption enabled (BitLocker, FileVault)
- Firewall enabled with minimum required ports
- Automatic updates enabled
- Security logging configured
- Administrative privileges removed from standard users

#### 3.7.2 Baseline Documentation

Configuration baselines must document:
- Required settings and values
- Rationale for each setting
- Exceptions and compensating controls
- Compliance checking procedures
- Baseline version and date

#### 3.7.3 Baseline Compliance

- New assets: Baseline applied before user access
- Existing assets: Scanned monthly for drift
- Non-compliant assets: Remediated within 7 days
- Exceptions: Documented and approved by Security

### 3.8 Software Asset Management

#### 3.8.1 License Tracking

Maintain accurate license records:
- License type (perpetual, subscription, floating, named)
- Quantity purchased vs. deployed
- Deployment locations and users
- Renewal dates and costs
- True-up requirements

#### 3.8.2 License Compliance

- Monitor usage against entitlements
- Automated discovery of installed software
- Quarterly license reconciliation
- Address over-deployment within 30 days
- Audit response procedures documented

#### 3.8.3 Software Metering

Track software usage to:
- Identify unused licenses for reclamation
- Support license negotiation with usage data
- Detect unauthorized software installation
- Plan capacity for future needs

#### 3.8.4 Unauthorized Software

- Prohibited software list maintained by Security
- Automated detection and alerting
- Unauthorized software removed without notice
- Repeat offenders escalated per POL-006

### 3.9 Hardware Lifecycle Management

#### 3.9.1 Acquisition Phase

Activities at acquisition:
- Verify against purchase order
- Inspect for damage
- Record in inventory with all details
- Apply asset tag
- Apply security baseline
- Verify warranty registration
- Store securely until deployment

#### 3.9.2 Deployment Phase

Activities at deployment:
- Assign to user or service
- Update inventory with location and assignment
- Provide user training if needed
- Document accepted condition
- Configure user-specific settings
- Verify security compliance

#### 3.9.3 Maintenance Phase

Ongoing activities:
- Regular patching and updates
- Performance monitoring
- Hardware repairs as needed
- Annual inventory verification
- Reassignment as users change
- Refresh planning based on age

#### 3.9.4 End-of-Life Phase

Asset retirement triggers:
- Age exceeds useful life (laptops: 4 years, servers: 5 years)
- No longer supported by vendor
- Performance no longer meets requirements
- Cost of maintenance exceeds value
- Security vulnerabilities cannot be patched

### 3.10 Disposal and Sanitization

#### 3.10.1 NIST 800-88 Compliance

All storage media must be sanitized per NIST SP 800-88 Rev. 1:

| Data Classification | Minimum Sanitization |
|--------------------|---------------------|
| **Restricted** | Destroy (physical destruction) |
| **Confidential** | Purge (cryptographic erase or overwrite) |
| **Internal** | Clear (overwrite or factory reset) |
| **Public** | Clear (factory reset acceptable) |

#### 3.10.2 Sanitization Methods

| Method | Description | Applicable Media |
|--------|-------------|-----------------|
| **Clear** | Logical overwrite, factory reset | HDDs, SSDs, mobile devices |
| **Purge** | Cryptographic erase, secure erase command | Self-encrypting drives, SSDs |
| **Destroy** | Physical destruction, shredding, degaussing | All media, required for Restricted |

#### 3.10.3 Disposal Process

1. **Retirement Request:** Owner submits retirement request
2. **Data Backup:** Verify required data is backed up elsewhere
3. **Inventory Update:** Mark asset as pending disposal
4. **Data Sanitization:** Apply appropriate sanitization method
5. **Verification:** Verify sanitization complete (certificate)
6. **Physical Disposal:** Recycle, donate, or destroy
7. **Documentation:** Record disposal date and method
8. **Inventory Removal:** Remove from active inventory

#### 3.10.4 Certificate of Destruction

Obtain certificate documenting:
- Asset identifier and serial number
- Sanitization method used
- Date of sanitization/destruction
- Personnel or vendor performing disposal
- Verification method

#### 3.10.5 Third-Party Disposal

If using disposal vendors:
- Vendor must be approved (POL-004 assessment)
- Chain of custody documented
- Certificate of destruction required
- Audit rights included in contract

### 3.11 Asset Audit Requirements

#### 3.11.1 Annual Physical Inventory

- Complete physical inventory audit annually
- Verify asset location and condition
- Reconcile with inventory records
- Investigate and resolve discrepancies
- Report findings to management

#### 3.11.2 Quarterly Sampling

- Random sample 25% of assets quarterly
- Verify existence, location, and condition
- Validate assigned user/service
- Check security baseline compliance

#### 3.11.3 Software License Audit

- Annual license compliance audit
- Reconcile deployed vs. licensed quantities
- Address over-deployment or under-utilization
- Document findings and remediation

#### 3.11.4 Cloud Resource Audit

- Monthly automated resource discovery
- Compare against inventory records
- Identify orphaned or untagged resources
- Review cost allocation accuracy

#### 3.11.5 Audit Documentation

Maintain audit records including:
- Audit date and scope
- Personnel conducting audit
- Findings and discrepancies
- Remediation actions
- Sign-off on completion

---

## 4. Roles and Responsibilities

### 4.1 IT Operations

- Maintain asset inventory system
- Process asset requests and deployments
- Apply security baselines to new assets
- Perform asset maintenance and repairs
- Execute asset disposal procedures
- Conduct physical inventory audits

### 4.2 IT Procurement

- Source assets from approved vendors
- Negotiate contracts and pricing
- Process purchase orders
- Manage warranty registrations
- Track license renewals
- Maintain vendor relationships

### 4.3 Information Security

- Define security baseline requirements
- Review high-risk asset requests
- Monitor baseline compliance
- Approve sanitization methods
- Investigate lost/stolen assets
- Review third-party disposal vendors

### 4.4 Finance

- Approve asset budgets and purchases
- Track asset capitalization and depreciation
- Conduct financial reconciliation
- Support license true-up calculations
- Verify disposal value recovery

### 4.5 Asset Owners

- Classify assets appropriately
- Authorize access to assets
- Make lifecycle decisions
- Ensure policy compliance
- Report issues and changes

### 4.6 All Employees

- Safeguard assigned assets
- Report lost, stolen, or damaged assets
- Return assets upon reassignment or termination
- Use assets in accordance with POL-006
- Cooperate with inventory audits

---

## 5. Procedures

### 5.1 New Asset Request

1. Submit request via IT Service Portal
2. Select asset type and specifications
3. Provide business justification
4. Obtain manager approval
5. Await security review (if applicable)
6. Track request through procurement
7. Receive deployed asset

### 5.2 Lost or Stolen Asset Reporting

1. Report immediately (within 24 hours maximum)
2. Contact IT Service Desk: help@greenlang.io
3. Contact Security if contains sensitive data
4. Provide asset ID, description, and last known location
5. File police report if theft suspected
6. Cooperate with investigation
7. Receive replacement (if approved)

### 5.3 Asset Transfer

1. Current custodian initiates transfer request
2. Identify receiving user or service
3. Obtain receiving manager approval
4. Schedule physical handoff (hardware)
5. Update inventory with new assignment
6. Verify security baseline compliance
7. Document transfer completion

### 5.4 Asset Retirement Request

1. Owner submits retirement request
2. Justify retirement (age, performance, EOL)
3. Verify data backup/migration complete
4. Remove from production use
5. Submit for disposal processing
6. Follow disposal procedures
7. Retain documentation

---

## 6. Exceptions

### 6.1 Exception Criteria

Exceptions to asset management requirements may be granted for:
- Legacy systems with documented compensating controls
- Specialized equipment with unique requirements
- Short-term projects with defined end dates
- Regulatory or contractual requirements

### 6.2 Exception Process

1. Request exception via IT Service Portal
2. Document business justification
3. Identify compensating controls
4. Security review for risk assessment
5. CTO approval for infrastructure exceptions
6. CFO approval for financial/licensing exceptions
7. Annual review of granted exceptions

---

## 7. Enforcement

### 7.1 Compliance Monitoring

Asset management compliance monitored through:
- Automated inventory discovery tools
- License compliance scanning
- Physical inventory audits
- Security baseline scanning
- Cloud resource tagging checks

### 7.2 Non-Compliance Consequences

Failure to comply with this policy may result in:
- Required remediation of non-compliant assets
- Access restriction to non-compliant systems
- Budget impact for lost or damaged assets
- Disciplinary action for willful violations
- Financial penalties for license non-compliance

### 7.3 Metrics and Reporting

Track and report quarterly:
- Inventory accuracy rate (target: >98%)
- Assets without owners (target: 0)
- License compliance rate (target: 100%)
- Baseline compliance rate (target: >95%)
- Average asset age by category
- Disposal backlog volume

---

## 8. Related Documents

| Document | Description |
|----------|-------------|
| POL-001: Information Security Policy | Master security policy |
| POL-002: Data Classification Policy | Data sensitivity classification |
| POL-004: Third-Party Risk Management | Vendor assessment for disposal vendors |
| POL-006: Acceptable Use Policy | Acceptable use of IT assets |
| POL-015: Media Protection Policy | Removable media handling |
| PRD-INFRA-004: S3/Object Storage | Cloud storage infrastructure |
| Procurement Guide | Detailed procurement procedures |
| Disposal Vendor List | Approved disposal vendors |

---

## 9. Definitions

| Term | Definition |
|------|------------|
| **Asset** | Any item of value to the organization (hardware, software, data, cloud resource) |
| **Asset Tag** | Unique identifier affixed to physical assets for tracking |
| **Asset Owner** | Person accountable for asset classification, access, and lifecycle |
| **Asset Custodian** | Person responsible for day-to-day asset management |
| **Baseline** | Documented standard configuration for security and compliance |
| **End-of-Life (EOL)** | Point at which asset no longer receives vendor support |
| **NIST 800-88** | NIST guidelines for media sanitization |
| **Sanitization** | Process of removing data from media before disposal |
| **True-up** | Reconciliation of software licenses with actual usage |
| **CMDB** | Configuration Management Database for asset tracking |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | CTO | Initial policy release |

---

## Appendix A: Asset Lifecycle State Diagram

```
[Requested] -> [Approved] -> [Procured] -> [Received] -> [Configured]
                                                              |
                                                              v
                                                         [Deployed]
                                                              |
                    +------------------+---------------------+
                    |                  |                     |
                    v                  v                     v
               [Assigned]        [In Service]          [In Storage]
                    |                  |                     |
                    +------------------+---------------------+
                                       |
                                       v
                                [Maintenance]
                                       |
                    +------------------+---------------------+
                    |                  |                     |
                    v                  v                     v
              [Repaired]         [Upgraded]            [Transferred]
                    |                  |                     |
                    +------------------+---------------------+
                                       |
                                       v
                               [Pending Retirement]
                                       |
                                       v
                               [Sanitization]
                                       |
                    +------------------+---------------------+
                    |                  |                     |
                    v                  v                     v
               [Recycled]         [Donated]            [Destroyed]
                                       |
                                       v
                                   [Disposed]
```

---

## Appendix B: Useful Life by Asset Type

| Asset Type | Standard Useful Life | Depreciation Method |
|------------|---------------------|---------------------|
| Laptops | 4 years | Straight-line |
| Desktops | 5 years | Straight-line |
| Monitors | 6 years | Straight-line |
| Servers | 5 years | Straight-line |
| Network Equipment | 7 years | Straight-line |
| Mobile Devices | 3 years | Straight-line |
| Security Appliances | 5 years | Straight-line |
| Software (perpetual) | 5 years | Straight-line |
| Software (subscription) | Expense as incurred | N/A |

---

**Document Classification: Internal**
**Policy Owner: Chief Technology Officer**
**Copyright 2026 GreenLang Climate OS. All Rights Reserved.**
