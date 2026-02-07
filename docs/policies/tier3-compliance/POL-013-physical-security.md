# POL-013: Physical Security Policy

**Document Control**

| Attribute | Value |
|-----------|-------|
| Document ID | POL-013 |
| Version | 1.0 |
| Classification | Confidential |
| Policy Tier | Tier 3 - Compliance |
| Owner | Chief Operating Officer (COO) |
| Approved By | Director of Facilities |
| Effective Date | 2026-02-06 |
| Last Review | 2026-02-06 |
| Next Review | 2027-02-06 |

---

## 1. Purpose

This policy establishes requirements for the physical security of GreenLang facilities, equipment, and personnel. The purpose is to protect information assets, prevent unauthorized access, ensure employee safety, and maintain a secure working environment.

Physical security is a foundational layer of the overall security program. Even the most sophisticated technical controls can be bypassed if physical access to systems, data centers, or sensitive areas is not properly controlled. This policy defines access controls, monitoring requirements, and security procedures for all GreenLang physical locations.

This policy supports compliance with SOC 2 Type II (Common Criteria CC6.4 - Physical Access), ISO 27001:2022 (A.7 Physical and Environmental Security), and industry best practices for facility security.

---

## 2. Scope

This policy applies to:

- **Facilities**: All GreenLang offices, co-working spaces, and contractor-accessed locations
- **Data Centers**: AWS data centers are governed by AWS SOC 2 reports; this policy covers GreenLang-managed colocation or on-premises equipment
- **Personnel**: All employees, contractors, visitors, and third-party service providers accessing GreenLang facilities
- **Equipment**: Servers, networking equipment, workstations, mobile devices, and any hardware containing GreenLang data
- **Areas**: All physical spaces including offices, server rooms, network closets, conference rooms, and common areas

### 2.1 Out of Scope

- AWS-managed data centers (covered by AWS SOC 2 Type II report)
- Employee home offices for remote work (covered by POL-014 Mobile Device and Remote Work Policy)
- Third-party vendor facilities (covered by vendor security assessments)

---

## 3. Policy Statement

GreenLang implements physical security controls commensurate with the sensitivity of information processed and stored at each location. Access to facilities is restricted to authorized personnel, and all access is logged and monitored. Visitors must be escorted, and sensitive areas require additional authentication.

### 3.1 Facility Access Control

#### 3.1.1 General Requirements

- All GreenLang facilities must have controlled entry points
- Badge or key card required for entry to all non-public areas
- Access credentials are personal and non-transferable
- Tailgating (following someone through a controlled door) is prohibited
- All access events are logged with timestamp, credential ID, and entry point
- Failed access attempts trigger security alerts after 3 consecutive failures

#### 3.1.2 Access Provisioning

| Personnel Type | Access Level | Approval Required | Duration |
|----------------|--------------|-------------------|----------|
| **Full-time Employees** | Role-based | Manager + HR | Employment duration |
| **Contractors** | Project-specific | Manager + Security | Contract duration |
| **Temporary Staff** | Limited areas | Manager | Assignment duration |
| **Visitors** | Escorted only | Host employee | Single visit |
| **Service Providers** | Specific areas | Facilities + Security | Service agreement |

#### 3.1.3 After-Hours Access

- Building access available 24/7 for authorized employees
- After-hours entry (outside 7:00 AM - 8:00 PM) logged with additional timestamp
- After-hours access to secure areas requires justification in access log
- Security notified of planned after-hours work in secure areas
- Emergency contacts maintained for all after-hours personnel

### 3.2 Visitor Management

#### 3.2.1 Visitor Registration

1. All visitors must be pre-registered by their host employee
2. Visitors sign in at reception with:
   - Full name
   - Company/organization
   - Host employee name
   - Purpose of visit
   - Time in
3. Visitors must present government-issued photo ID
4. Visitor information retained for 90 days

#### 3.2.2 Visitor Badges

| Badge Type | Color | Access Level | Duration |
|------------|-------|--------------|----------|
| **General Visitor** | Red | Escorted only, common areas | Single day |
| **Contractor** | Orange | Project areas, escorted | Multi-day |
| **VIP/Executive** | Purple | Escorted, conference rooms | Single day |
| **Delivery** | Yellow | Loading dock, lobby only | During delivery |

#### 3.2.3 Escort Requirements

- Visitors must be escorted at all times outside of reception/lobby
- Host is responsible for visitor's actions while on premises
- Visitor must be accompanied when entering secure areas
- Host must ensure visitor signs out before leaving
- Unescorted visitors reported to security immediately

#### 3.2.4 Visitor Restrictions

- No photography without written approval
- No access to server rooms or network closets
- No connection to internal network (guest Wi-Fi only)
- No removal of GreenLang property or documents
- NDA required before access to confidential information

### 3.3 Security Zones

GreenLang facilities are divided into security zones with increasing access restrictions:

#### 3.3.1 Zone Definitions

| Zone | Name | Access Control | Monitoring | Examples |
|------|------|----------------|------------|----------|
| **Zone 0** | Public | Open during business hours | CCTV, reception | Lobby, parking lot |
| **Zone 1** | Office | Badge required | CCTV, access logs | Open office, meeting rooms |
| **Zone 2** | Restricted | Badge + PIN or badge + MFA | CCTV, access logs, alerts | Executive offices, finance |
| **Zone 3** | Secure | Escorted + logged + authorized list | CCTV, motion sensors, 24/7 monitoring | Server room, network closet |

#### 3.3.2 Zone Transition Requirements

| From Zone | To Zone | Requirements |
|-----------|---------|--------------|
| Zone 0 | Zone 1 | Valid badge swipe |
| Zone 1 | Zone 2 | Valid badge + PIN/MFA, access list verification |
| Zone 2 | Zone 3 | Escort by authorized personnel, entry logged, access list |
| Any Zone | Zone 0 | Badge out recommended (tracked for safety) |

#### 3.3.3 Zone 3 (Secure Area) Procedures

Access to Zone 3 areas requires:
1. Name on authorized access list (maintained by Security)
2. Business justification documented
3. Escort by Zone 3-authorized personnel
4. Sign-in to physical access log with:
   - Name, company, time in/out
   - Purpose of access
   - Equipment brought in/out
5. Security notification before access
6. No mobile phones with cameras (or cameras disabled)

### 3.4 Badge and Credential Management

#### 3.4.1 Badge Issuance

1. HR submits access request for new employee
2. Security verifies identity and approves access level
3. Badge issued with photo, name, and employee ID
4. Badge programmed with appropriate access rights
5. Employee acknowledges badge responsibility

#### 3.4.2 Badge Requirements

- Badges must be displayed visibly at all times in the facility
- Badges must not be shared, loaned, or used by another person
- Lost or stolen badges must be reported within 1 hour
- Damaged badges must be returned for replacement
- Badges must be returned upon termination (within 24 hours)

#### 3.4.3 Badge Deactivation

| Event | Deactivation Timeline |
|-------|----------------------|
| Voluntary termination | Last day of employment |
| Involuntary termination | Immediately upon notification |
| Lost/stolen badge | Immediately upon report |
| Leave of absence > 30 days | Start of leave |
| Contract completion | Contract end date |

### 3.5 Video Surveillance

#### 3.5.1 Coverage Requirements

| Area Type | Camera Coverage | Recording |
|-----------|-----------------|-----------|
| Building entrances/exits | 100% | 24/7 |
| Parking areas | 100% | 24/7 |
| Lobby/reception | 100% | 24/7 |
| Hallways and corridors | 100% | 24/7 |
| Server rooms | 100%, multiple angles | 24/7 |
| Conference rooms | Entry points only | Business hours |
| Break rooms | Entry points only | 24/7 |

#### 3.5.2 Recording Retention

| Recording Type | Retention Period | Storage |
|----------------|------------------|---------|
| General surveillance | 90 days | On-premises NVR + cloud backup |
| Incident-related | 1 year minimum | Secure evidence storage |
| Legal hold | Until released | Secure evidence storage |
| Server room | 90 days | Encrypted cloud storage |

#### 3.5.3 Footage Access

- Live monitoring by security personnel during business hours
- Recorded footage access restricted to:
  - Security team (investigation purposes)
  - HR (with security escort, employee matters)
  - Legal (litigation, subpoena)
  - Law enforcement (with valid legal process)
- All footage access logged with accessor, timestamp, reason
- No footage shared externally without Legal approval

### 3.6 Environmental Controls

#### 3.6.1 Fire Safety

| Control | Requirement | Inspection |
|---------|-------------|------------|
| Fire detection | Smoke detectors in all rooms | Monthly test |
| Fire suppression | Sprinklers (office), clean agent (server room) | Annual inspection |
| Fire extinguishers | Located per fire code | Monthly visual, annual service |
| Emergency exits | Illuminated, unobstructed | Weekly inspection |
| Evacuation plans | Posted on each floor | Annual review |
| Fire drills | Conducted | Semi-annually |

#### 3.6.2 Water and Flood Protection

| Control | Location | Monitoring |
|---------|----------|------------|
| Water leak detection | Server room, under-floor | 24/7 automated alerts |
| Raised flooring | Server room | Water sensors at floor level |
| Drainage | Server room, basement | Annual inspection |
| Waterproof containers | Backup media storage | Inventory check quarterly |

#### 3.6.3 HVAC and Climate Control

| Area | Temperature | Humidity | Monitoring |
|------|-------------|----------|------------|
| Server room | 64-75 F (18-24 C) | 40-60% | 24/7, alerting |
| Network closets | 64-75 F (18-24 C) | 40-60% | 24/7, alerting |
| Office areas | 68-76 F (20-24 C) | 30-70% | Business hours |
| Media storage | 60-70 F (15-21 C) | 30-40% | Daily check |

#### 3.6.4 Power Protection

| Control | Coverage | Monitoring |
|---------|----------|------------|
| UPS (Uninterruptible Power Supply) | All server room equipment | 24/7, battery status |
| Generator | Full facility backup | Weekly test, monthly load test |
| Surge protection | All IT equipment | Annual inspection |
| Dual power feeds | Server room | Power monitoring system |

### 3.7 Equipment Placement and Protection

#### 3.7.1 Server and Network Equipment

- Servers must be located in Zone 3 (Secure) areas only
- Equipment racks must be lockable
- Cabling must be organized and labeled
- No food, drink, or smoking near equipment
- Equipment inventory maintained with asset tags
- Serial numbers recorded for all hardware

#### 3.7.2 Workstations

- Workstations must be secured with cable locks in open areas
- Laptops must be locked away when unattended overnight
- Screens must lock after 5 minutes of inactivity
- No sensitive data visible when away from desk (clean desk policy)

#### 3.7.3 Mobile and Portable Equipment

- Portable equipment must be secured when not in use
- Laptops taken off-site must be encrypted
- Equipment removal requires authorization for non-company events
- Equipment return verified against asset inventory

### 3.8 Secure Areas

#### 3.8.1 Server Rooms

- Access limited to authorized IT personnel
- All access logged and reviewed weekly
- No windows; solid walls
- Dedicated HVAC with redundancy
- Clean agent fire suppression (FM-200 or equivalent)
- Environmental monitoring (temperature, humidity, water)
- Emergency power off (EPO) button accessible

#### 3.8.2 Network Closets

- Locked at all times
- Access limited to network team
- Temperature monitoring
- Surge protection on all equipment
- Cable management and labeling required

#### 3.8.3 Sensitive Document Storage

- Locked filing cabinets for physical documents
- Access restricted to document owners
- Keys managed by facilities
- Documents classified per POL-004

### 3.9 Delivery and Loading Procedures

#### 3.9.1 General Deliveries

1. Deliveries scheduled with facilities in advance when possible
2. Delivery personnel check in at reception
3. Packages inspected before acceptance
4. Deliveries to secure areas escorted by facilities staff
5. Delivery log maintained with date, carrier, recipient, package description

#### 3.9.2 IT Equipment Deliveries

1. IT equipment deliveries coordinated with IT department
2. Receiving personnel verify packing list against PO
3. Serial numbers recorded before storage
4. Equipment stored in secure cage until deployment
5. Chain of custody documented

#### 3.9.3 Outbound Shipments

1. Shipping request submitted via facilities
2. Sensitive equipment shipments require encryption verification
3. Chain of custody documentation for IT equipment
4. Tracking information retained for 90 days

### 3.10 Physical Security Incident Response

#### 3.10.1 Incident Types

| Incident Type | Response Priority | Notification |
|---------------|-------------------|--------------|
| Unauthorized access attempt | High | Security, Facilities Manager |
| Tailgating observed | Medium | Security |
| Lost/stolen badge | High | Security, HR |
| Unescorted visitor | High | Security, Host employee |
| Theft or vandalism | Critical | Security, Police, Management |
| Environmental alarm | Critical | Facilities, Security, Management |
| Medical emergency | Critical | 911, Security, HR |

#### 3.10.2 Response Procedures

1. **Immediate**: Secure the area, ensure personnel safety
2. **Notify**: Contact Security and appropriate personnel per incident type
3. **Document**: Record incident details, witnesses, evidence
4. **Preserve**: Preserve video footage and access logs
5. **Investigate**: Security conducts investigation within 24 hours
6. **Report**: Incident report completed within 48 hours
7. **Remediate**: Implement corrective actions
8. **Review**: Lessons learned incorporated into procedures

---

## 4. Roles and Responsibilities

| Role | Responsibilities |
|------|------------------|
| **COO** | Policy ownership, facility security budget, executive decisions |
| **Director of Facilities** | Day-to-day security operations, vendor management, maintenance |
| **Security Team** | Access control administration, monitoring, incident response |
| **Reception** | Visitor management, badge issuance, first point of contact |
| **HR** | Access provisioning coordination, termination badge collection |
| **IT** | Technical security systems, server room access management |
| **All Employees** | Badge display, tailgating prevention, visitor escort, incident reporting |

---

## 5. Procedures

### 5.1 Requesting Facility Access

1. Manager submits access request via HR system
2. Security reviews and approves based on role requirements
3. Badge programmed and issued (within 2 business days for employees)
4. Employee completes physical security acknowledgment
5. Access activated in system

### 5.2 Reporting a Security Concern

1. Contact Security immediately (phone, email, or in person)
2. Do not confront suspicious individuals
3. Provide details: location, description, time
4. Security responds and investigates
5. Follow-up communication to reporter within 24 hours

### 5.3 Termination Badge Return

1. HR notifies Security of termination (same day)
2. Badge deactivated in system
3. Manager collects badge during exit process
4. Badge returned to Security and destroyed
5. Access removal confirmed in system

---

## 6. Exceptions

Exceptions to this policy require:

1. Written request with business justification
2. Risk assessment by Security
3. Approval from Director of Facilities and CISO
4. Time-limited exception (maximum 90 days)
5. Compensating controls documented

Emergency exceptions may be granted verbally by COO or CISO with documented follow-up within 24 hours.

---

## 7. Enforcement

Violations of this policy may result in:

- Verbal warning (first minor offense)
- Written warning (second offense or first moderate offense)
- Access suspension pending investigation
- Disciplinary action up to termination
- Legal action for theft, vandalism, or unauthorized disclosure

Employees are expected to report policy violations. Reports may be made anonymously.

---

## 8. Related Documents

| Document ID | Document Name |
|-------------|---------------|
| POL-003 | Access Control Policy |
| POL-006 | Incident Response Policy |
| POL-014 | Mobile Device and Remote Work Policy |
| POL-015 | Media Protection Policy |
| STD-PHYS-001 | Physical Security Standards |
| PRO-EVAC-001 | Emergency Evacuation Procedure |
| PRO-INC-001 | Physical Security Incident Procedure |

---

## 9. Definitions

| Term | Definition |
|------|------------|
| **Badge** | Physical access credential (card, fob) used to authenticate entry |
| **Clean Agent** | Fire suppression agent safe for electronic equipment (e.g., FM-200) |
| **EPO** | Emergency Power Off - button to immediately cut power in emergency |
| **NVR** | Network Video Recorder - system for storing surveillance footage |
| **Tailgating** | Following an authorized person through a controlled access point |
| **Zone** | Physical area with defined access controls and monitoring |
| **MFA** | Multi-Factor Authentication - requiring multiple authentication methods |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | Facilities Team | Initial policy creation |

---

**Document Classification: Confidential**

*This policy is the property of GreenLang Climate OS. Unauthorized distribution, copying, or disclosure is prohibited.*
