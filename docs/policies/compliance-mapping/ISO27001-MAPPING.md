# ISO 27001:2022 Annex A Controls Mapping

**Document Control**

| Attribute | Value |
|-----------|-------|
| Document ID | MAP-ISO27001-001 |
| Version | 1.0 |
| Classification | Internal |
| Owner | Chief Information Security Officer (CISO) |
| Approved By | Compliance Officer |
| Effective Date | 2026-02-06 |
| Last Updated | 2026-02-06 |
| Next Review | 2026-08-06 |

---

## 1. Overview

This document provides a comprehensive mapping of ISO 27001:2022 Annex A controls to GreenLang policies, implementation status, and evidence locations. This serves as the foundation for the Statement of Applicability (SoA) and supports certification audits.

**Scope:** GreenLang Climate OS Platform - Information Security Management System (ISMS)

**Standard Version:** ISO/IEC 27001:2022

**Control Categories (ISO 27001:2022):**
- A.5 Organizational Controls (37 controls)
- A.6 People Controls (8 controls)
- A.7 Physical Controls (14 controls)
- A.8 Technological Controls (34 controls)

**Total Controls:** 93

---

## 2. Statement of Applicability (SoA) Summary

| Category | Total | Applicable | Implemented | Partially Implemented | Planned | Not Applicable |
|----------|-------|------------|-------------|----------------------|---------|----------------|
| A.5 Organizational | 37 | 37 | 35 | 2 | 0 | 0 |
| A.6 People | 8 | 8 | 8 | 0 | 0 | 0 |
| A.7 Physical | 14 | 8 | 8 | 0 | 0 | 6 |
| A.8 Technological | 34 | 34 | 33 | 1 | 0 | 0 |
| **TOTAL** | **93** | **87** | **84** | **3** | **0** | **6** |

---

## 3. A.5 Organizational Controls

### A.5.1 Policies for Information Security

| Attribute | Detail |
|-----------|--------|
| **Control** | Information security policy and topic-specific policies shall be defined, approved by management, published, communicated to and acknowledged by relevant personnel and relevant interested parties, and reviewed at planned intervals and if significant changes occur. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-001 (Information Security Policy), All policies |
| **Implementation Status** | Implemented |
| **Evidence Location** | docs/policies/, Policy acknowledgment records |
| **Last Reviewed** | 2026-02-06 |

### A.5.2 Information Security Roles and Responsibilities

| Attribute | Detail |
|-----------|--------|
| **Control** | Information security roles and responsibilities shall be defined and allocated according to the organization needs. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-001 (Information Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | RACI matrices, Job descriptions, Org chart |
| **Last Reviewed** | 2026-02-06 |

### A.5.3 Segregation of Duties

| Attribute | Detail |
|-----------|--------|
| **Control** | Conflicting duties and conflicting areas of responsibility shall be segregated. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-003 (Access Control Policy), POL-014 (Risk Management Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | SoD matrix, Access control configurations, Approval workflows |
| **Last Reviewed** | 2026-02-06 |

### A.5.4 Management Responsibilities

| Attribute | Detail |
|-----------|--------|
| **Control** | Management shall require all personnel to apply information security in accordance with the established information security policy, topic-specific policies and procedures of the organization. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-001 (Information Security Policy), POL-016 (Security Awareness Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Policy acknowledgments, Training records, Performance reviews |
| **Last Reviewed** | 2026-02-06 |

### A.5.5 Contact with Authorities

| Attribute | Detail |
|-----------|--------|
| **Control** | The organization shall establish and maintain contact with relevant authorities. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-006 (Incident Response Policy), POL-018 (Incident Communication Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Authority contact list, Regulatory notification procedures |
| **Last Reviewed** | 2026-02-06 |

### A.5.6 Contact with Special Interest Groups

| Attribute | Detail |
|-----------|--------|
| **Control** | The organization shall establish and maintain contact with special interest groups or other specialist security forums and professional associations. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-001 (Information Security Policy), POL-016 (Security Awareness Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Industry group memberships, ISAC participation, Conference attendance |
| **Last Reviewed** | 2026-02-06 |

### A.5.7 Threat Intelligence

| Attribute | Detail |
|-----------|--------|
| **Control** | Information relating to information security threats shall be collected and analysed to produce threat intelligence. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-014 (Risk Management Policy), POL-011 (Logging and Monitoring Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Threat intelligence feeds, Security bulletins, Vulnerability advisories |
| **Last Reviewed** | 2026-02-06 |

### A.5.8 Information Security in Project Management

| Attribute | Detail |
|-----------|--------|
| **Control** | Information security shall be integrated into project management. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-010 (SDLC Security Policy), POL-007 (Change Management Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Security gates in SDLC, Project security checklists, Design reviews |
| **Last Reviewed** | 2026-02-06 |

### A.5.9 Inventory of Information and Other Associated Assets

| Attribute | Detail |
|-----------|--------|
| **Control** | An inventory of information and other associated assets, including owners, shall be developed and maintained. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-008 (Asset Management Policy), POL-002 (Data Classification Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Asset inventory, Data inventory, CMDB |
| **Last Reviewed** | 2026-02-06 |

### A.5.10 Acceptable Use of Information and Other Associated Assets

| Attribute | Detail |
|-----------|--------|
| **Control** | Rules for the acceptable use of information and other associated assets shall be identified, documented and implemented. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-006 (Acceptable Use Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Acceptable use policy, Acknowledgments |
| **Last Reviewed** | 2026-02-06 |

### A.5.11 Return of Assets

| Attribute | Detail |
|-----------|--------|
| **Control** | Personnel and other interested parties as appropriate shall return all the organization's assets in their possession upon change or termination of their employment, contract or agreement. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-005 (Personnel Security Policy), POL-008 (Asset Management Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Termination checklist, Asset return records |
| **Last Reviewed** | 2026-02-06 |

### A.5.12 Classification of Information

| Attribute | Detail |
|-----------|--------|
| **Control** | Information shall be classified according to the information security needs of the organization based on confidentiality, integrity, availability and relevant interested party requirements. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-002 (Data Classification Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Classification scheme, Data inventory with classifications |
| **Last Reviewed** | 2026-02-06 |

### A.5.13 Labelling of Information

| Attribute | Detail |
|-----------|--------|
| **Control** | An appropriate set of procedures for information labelling shall be developed and implemented in accordance with the information classification scheme adopted by the organization. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-002 (Data Classification Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Labeling standards, Document templates, Automated labeling |
| **Last Reviewed** | 2026-02-06 |

### A.5.14 Information Transfer

| Attribute | Detail |
|-----------|--------|
| **Control** | Information transfer rules, procedures, or agreements shall be in place for all types of transfer facilities within the organization and between the organization and other parties. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-002 (Data Classification Policy), POL-011 (Encryption Policy), POL-015 (Media Protection Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Transfer procedures, Encryption standards, DPA templates |
| **Last Reviewed** | 2026-02-06 |

### A.5.15 Access Control

| Attribute | Detail |
|-----------|--------|
| **Control** | Rules to control physical and logical access to information and other associated assets shall be established and implemented based on business and information security requirements. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-003 (Access Control Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Access control policy, RBAC configurations, Access matrices |
| **Last Reviewed** | 2026-02-06 |

### A.5.16 Identity Management

| Attribute | Detail |
|-----------|--------|
| **Control** | The full life cycle of identities shall be managed. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-003 (Access Control Policy), POL-005 (Personnel Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | IAM system, Provisioning/deprovisioning records |
| **Last Reviewed** | 2026-02-06 |

### A.5.17 Authentication Information

| Attribute | Detail |
|-----------|--------|
| **Control** | Allocation and management of authentication information shall be controlled by a management process, including advising personnel on appropriate handling of authentication information. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-009 (Password and Authentication Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Password policy, MFA enrollment, Credential management |
| **Last Reviewed** | 2026-02-06 |

### A.5.18 Access Rights

| Attribute | Detail |
|-----------|--------|
| **Control** | Access rights to information and other associated assets shall be provisioned, reviewed, modified and removed in accordance with the organization's topic-specific policy on and rules for access control. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-003 (Access Control Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Access request records, Access reviews, Deprovisioning logs |
| **Last Reviewed** | 2026-02-06 |

### A.5.19 Information Security in Supplier Relationships

| Attribute | Detail |
|-----------|--------|
| **Control** | Processes and procedures shall be defined and implemented to manage the information security risks associated with the use of supplier's products or services. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-004 (Third-Party Risk Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Vendor risk assessments, Contract requirements, DPAs |
| **Last Reviewed** | 2026-02-06 |

### A.5.20 Addressing Information Security within Supplier Agreements

| Attribute | Detail |
|-----------|--------|
| **Control** | Relevant information security requirements shall be established and agreed with each supplier based on the type of supplier relationship. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-004 (Third-Party Risk Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Contract templates, Security addendums, DPA templates |
| **Last Reviewed** | 2026-02-06 |

### A.5.21 Managing Information Security in the ICT Supply Chain

| Attribute | Detail |
|-----------|--------|
| **Control** | Processes and procedures shall be defined and implemented to manage the information security risks associated with the ICT products and services supply chain. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-004 (Third-Party Risk Policy), POL-010 (SDLC Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Supply chain assessments, SCA tools, SBOM generation |
| **Last Reviewed** | 2026-02-06 |

### A.5.22 Monitoring, Review and Change Management of Supplier Services

| Attribute | Detail |
|-----------|--------|
| **Control** | The organization shall regularly monitor, review, evaluate and manage change in supplier information security practices and service delivery. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-004 (Third-Party Risk Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Annual vendor reviews, SOC 2 report reviews, Change notifications |
| **Last Reviewed** | 2026-02-06 |

### A.5.23 Information Security for Use of Cloud Services

| Attribute | Detail |
|-----------|--------|
| **Control** | Processes for acquisition, use, management and exit from cloud services shall be established in accordance with the organization's information security requirements. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-004 (Third-Party Risk Policy), POL-001 (Information Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Cloud security policies, AWS security configurations, Exit procedures |
| **Last Reviewed** | 2026-02-06 |

### A.5.24 Information Security Incident Management Planning and Preparation

| Attribute | Detail |
|-----------|--------|
| **Control** | The organization shall plan and prepare for managing information security incidents by defining, establishing and communicating information security incident management processes, roles and responsibilities. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-006 (Incident Response Policy), POL-018 (Incident Communication Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Incident response plan, IR team roster, Runbooks |
| **Last Reviewed** | 2026-02-06 |

### A.5.25 Assessment and Decision on Information Security Events

| Attribute | Detail |
|-----------|--------|
| **Control** | The organization shall assess information security events and decide if they are to be categorized as information security incidents. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-006 (Incident Response Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Incident classification criteria, Triage procedures, Event logs |
| **Last Reviewed** | 2026-02-06 |

### A.5.26 Response to Information Security Incidents

| Attribute | Detail |
|-----------|--------|
| **Control** | Information security incidents shall be responded to in accordance with the documented procedures. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-006 (Incident Response Policy), POL-018 (Incident Communication Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Incident tickets, Response logs, Communication records |
| **Last Reviewed** | 2026-02-06 |

### A.5.27 Learning from Information Security Incidents

| Attribute | Detail |
|-----------|--------|
| **Control** | Knowledge gained from information security incidents shall be used to strengthen and improve the information security controls. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-006 (Incident Response Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Post-incident reviews, Lessons learned, Improvement tracking |
| **Last Reviewed** | 2026-02-06 |

### A.5.28 Collection of Evidence

| Attribute | Detail |
|-----------|--------|
| **Control** | The organization shall establish and implement procedures for the identification, collection, acquisition and preservation of evidence related to information security events. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-006 (Incident Response Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Evidence handling procedures, Chain of custody forms, Forensic toolkit |
| **Last Reviewed** | 2026-02-06 |

### A.5.29 Information Security During Disruption

| Attribute | Detail |
|-----------|--------|
| **Control** | The organization shall plan how to maintain information security at an appropriate level during disruption. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-007 (Business Continuity Policy), POL-012 (Backup and Recovery Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | BCP documentation, Security during DR procedures |
| **Last Reviewed** | 2026-02-06 |

### A.5.30 ICT Readiness for Business Continuity

| Attribute | Detail |
|-----------|--------|
| **Control** | ICT readiness shall be planned, implemented, maintained and tested based on business continuity objectives and ICT continuity requirements. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-007 (Business Continuity Policy), POL-012 (Backup and Recovery Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | DR architecture, DR test results, RTO/RPO documentation |
| **Last Reviewed** | 2026-02-06 |

### A.5.31 Legal, Statutory, Regulatory and Contractual Requirements

| Attribute | Detail |
|-----------|--------|
| **Control** | Legal, statutory, regulatory and contractual requirements relevant to information security and the organization's approach to meet these requirements shall be identified, documented and kept up to date. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-001 (Information Security Policy), POL-017 (Privacy Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Compliance register, Regulatory mapping documents |
| **Last Reviewed** | 2026-02-06 |

### A.5.32 Intellectual Property Rights

| Attribute | Detail |
|-----------|--------|
| **Control** | The organization shall implement appropriate procedures to protect intellectual property rights. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-006 (Acceptable Use Policy), POL-008 (Asset Management Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Software license inventory, IP agreements, Open source policy |
| **Last Reviewed** | 2026-02-06 |

### A.5.33 Protection of Records

| Attribute | Detail |
|-----------|--------|
| **Control** | Records shall be protected from loss, destruction, falsification, unauthorized access and unauthorized release in accordance with legal, statutory, regulatory and contractual requirements. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-005 (Data Retention Policy), POL-002 (Data Classification Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Retention schedules, Record protection controls, Archive procedures |
| **Last Reviewed** | 2026-02-06 |

### A.5.34 Privacy and Protection of PII

| Attribute | Detail |
|-----------|--------|
| **Control** | The organization shall identify and meet the requirements regarding the preservation of privacy and protection of PII according to applicable laws and regulations and contractual requirements. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-017 (Privacy Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Privacy policy, DPIA records, Data subject request logs |
| **Last Reviewed** | 2026-02-06 |

### A.5.35 Independent Review of Information Security

| Attribute | Detail |
|-----------|--------|
| **Control** | The organization's approach to managing information security and its implementation including people, processes and technologies shall be reviewed independently at planned intervals, or when significant changes occur. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-001 (Information Security Policy), POL-014 (Risk Management Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Internal audit reports, External audit reports, Penetration test reports |
| **Last Reviewed** | 2026-02-06 |

### A.5.36 Compliance with Policies, Rules and Standards for Information Security

| Attribute | Detail |
|-----------|--------|
| **Control** | Compliance with the organization's information security policy, topic-specific policies, rules and standards shall be regularly reviewed. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-001 (Information Security Policy), POL-014 (Risk Management Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Compliance assessments, Policy exception register, Audit findings |
| **Last Reviewed** | 2026-02-06 |

### A.5.37 Documented Operating Procedures

| Attribute | Detail |
|-----------|--------|
| **Control** | Operating procedures for information processing facilities shall be documented and made available to personnel who need them. |
| **Applicable** | Yes |
| **Applicable Policies** | All policies |
| **Implementation Status** | Partially Implemented |
| **Evidence Location** | Runbooks, SOPs, Work instructions |
| **Gap** | Some operational procedures require documentation updates |
| **Remediation** | Update operational runbooks by Q2 2026 |
| **Last Reviewed** | 2026-02-06 |

---

## 4. A.6 People Controls

### A.6.1 Screening

| Attribute | Detail |
|-----------|--------|
| **Control** | Background verification checks on all candidates to become personnel shall be carried out prior to joining the organization and on an ongoing basis taking into consideration applicable laws, regulations and ethics and be proportional to the business requirements, the classification of the information to be accessed and the perceived risks. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-005 (Personnel Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Background check records, Screening criteria documentation |
| **Last Reviewed** | 2026-02-06 |

### A.6.2 Terms and Conditions of Employment

| Attribute | Detail |
|-----------|--------|
| **Control** | The employment contractual agreements shall state the personnel's and the organization's responsibilities for information security. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-005 (Personnel Security Policy), POL-001 (Information Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Employment contracts, NDA templates, Security clauses |
| **Last Reviewed** | 2026-02-06 |

### A.6.3 Information Security Awareness, Education and Training

| Attribute | Detail |
|-----------|--------|
| **Control** | Personnel of the organization and relevant interested parties shall receive appropriate information security awareness, education and training and regular updates of the organization's information security policy, topic-specific policies and procedures, as relevant for their job function. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-016 (Security Awareness Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Training records, LMS completion data, Phishing simulation results |
| **Last Reviewed** | 2026-02-06 |

### A.6.4 Disciplinary Process

| Attribute | Detail |
|-----------|--------|
| **Control** | A disciplinary process shall be formalized and communicated to take actions against personnel and other relevant interested parties who have committed an information security policy violation. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-006 (Acceptable Use Policy), POL-016 (Security Awareness Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Disciplinary procedures, HR policy, Incident documentation |
| **Last Reviewed** | 2026-02-06 |

### A.6.5 Responsibilities After Termination or Change of Employment

| Attribute | Detail |
|-----------|--------|
| **Control** | Information security responsibilities and duties that remain valid after termination or change of employment shall be defined, enforced and communicated to relevant personnel and other interested parties. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-005 (Personnel Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Exit interview checklist, NDA reminders, Offboarding procedures |
| **Last Reviewed** | 2026-02-06 |

### A.6.6 Confidentiality or Non-Disclosure Agreements

| Attribute | Detail |
|-----------|--------|
| **Control** | Confidentiality or non-disclosure agreements reflecting the organization's needs for the protection of information shall be identified, documented, regularly reviewed and signed by personnel and other relevant interested parties. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-005 (Personnel Security Policy), POL-004 (Third-Party Risk Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | NDA templates, Signed NDAs, Annual review records |
| **Last Reviewed** | 2026-02-06 |

### A.6.7 Remote Working

| Attribute | Detail |
|-----------|--------|
| **Control** | Security measures shall be implemented when personnel are working remotely to protect information accessed, processed or stored outside the organization's premises. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-014 (Mobile Device and Remote Work Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Remote work policy, VPN configurations, Endpoint security |
| **Last Reviewed** | 2026-02-06 |

### A.6.8 Information Security Event Reporting

| Attribute | Detail |
|-----------|--------|
| **Control** | The organization shall provide a mechanism for personnel to report observed or suspected information security events through appropriate channels in a timely manner. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-006 (Incident Response Policy), POL-016 (Security Awareness Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Reporting mechanisms, Security channel, Training materials |
| **Last Reviewed** | 2026-02-06 |

---

## 5. A.7 Physical Controls

### A.7.1 Physical Security Perimeters

| Attribute | Detail |
|-----------|--------|
| **Control** | Security perimeters shall be defined and used to protect areas that contain information and other associated assets. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-013 (Physical Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Office security, Data center (AWS) SOC 2 reports |
| **Last Reviewed** | 2026-02-06 |

### A.7.2 Physical Entry

| Attribute | Detail |
|-----------|--------|
| **Control** | Secure areas shall be protected by appropriate entry controls and access points. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-013 (Physical Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Badge access logs, Visitor logs, Entry controls |
| **Last Reviewed** | 2026-02-06 |

### A.7.3 Securing Offices, Rooms and Facilities

| Attribute | Detail |
|-----------|--------|
| **Control** | Physical security for offices, rooms and facilities shall be designed and implemented. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-013 (Physical Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Office security measures, Clean desk policy |
| **Last Reviewed** | 2026-02-06 |

### A.7.4 Physical Security Monitoring

| Attribute | Detail |
|-----------|--------|
| **Control** | Premises shall be continuously monitored for unauthorized physical access. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-013 (Physical Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | CCTV systems, Alarm systems, Monitoring logs |
| **Last Reviewed** | 2026-02-06 |

### A.7.5 Protecting Against Physical and Environmental Threats

| Attribute | Detail |
|-----------|--------|
| **Control** | Protection against physical and environmental threats, such as natural disasters and other intentional or unintentional physical threats to infrastructure shall be designed and implemented. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-013 (Physical Security Policy), POL-007 (Business Continuity Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | AWS data center certifications, Environmental controls |
| **Last Reviewed** | 2026-02-06 |

### A.7.6 Working in Secure Areas

| Attribute | Detail |
|-----------|--------|
| **Control** | Security measures for working in secure areas shall be designed and implemented. |
| **Applicable** | No (Cloud-hosted) |
| **Justification** | GreenLang operates primarily cloud-hosted infrastructure. No on-premises data centers or secure processing areas. |
| **Applicable Policies** | N/A |
| **Implementation Status** | Not Applicable |

### A.7.7 Clear Desk and Clear Screen

| Attribute | Detail |
|-----------|--------|
| **Control** | Clear desk rules for papers and removable storage media and clear screen rules for information processing facilities shall be defined and appropriately enforced. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-006 (Acceptable Use Policy), POL-013 (Physical Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Clear desk policy, Screen lock configurations |
| **Last Reviewed** | 2026-02-06 |

### A.7.8 Equipment Siting and Protection

| Attribute | Detail |
|-----------|--------|
| **Control** | Equipment shall be sited securely and protected. |
| **Applicable** | No (Cloud-hosted) |
| **Justification** | Infrastructure hosted on AWS. Physical equipment protection managed by AWS per their SOC 2. |
| **Applicable Policies** | N/A |
| **Implementation Status** | Not Applicable |

### A.7.9 Security of Assets Off-Premises

| Attribute | Detail |
|-----------|--------|
| **Control** | Off-site assets shall be protected. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-014 (Mobile Device and Remote Work Policy), POL-008 (Asset Management Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Laptop encryption, MDM enrollment, Asset tracking |
| **Last Reviewed** | 2026-02-06 |

### A.7.10 Storage Media

| Attribute | Detail |
|-----------|--------|
| **Control** | Storage media shall be managed through their life cycle of acquisition, use, transportation and disposal in accordance with the organization's classification scheme and handling requirements. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-015 (Media Protection Policy), POL-008 (Asset Management Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Media handling procedures, Disposal records, Encryption requirements |
| **Last Reviewed** | 2026-02-06 |

### A.7.11 Supporting Utilities

| Attribute | Detail |
|-----------|--------|
| **Control** | Information processing facilities shall be protected from power failures and other disruptions caused by failures in supporting utilities. |
| **Applicable** | No (Cloud-hosted) |
| **Justification** | Infrastructure hosted on AWS which provides redundant power, cooling, and utilities per their certifications. |
| **Applicable Policies** | N/A |
| **Implementation Status** | Not Applicable |

### A.7.12 Cabling Security

| Attribute | Detail |
|-----------|--------|
| **Control** | Cables carrying power, data or supporting information services shall be protected from interception, interference or damage. |
| **Applicable** | No (Cloud-hosted) |
| **Justification** | No on-premises data center cabling. AWS manages physical cabling security. |
| **Applicable Policies** | N/A |
| **Implementation Status** | Not Applicable |

### A.7.13 Equipment Maintenance

| Attribute | Detail |
|-----------|--------|
| **Control** | Equipment shall be maintained correctly to ensure availability, integrity and confidentiality of information. |
| **Applicable** | No (Cloud-hosted) |
| **Justification** | Physical infrastructure maintenance managed by AWS. Endpoint maintenance covered under asset management. |
| **Applicable Policies** | POL-008 (Asset Management Policy) |
| **Implementation Status** | Not Applicable (infrastructure), Implemented (endpoints) |

### A.7.14 Secure Disposal or Re-use of Equipment

| Attribute | Detail |
|-----------|--------|
| **Control** | Items of equipment containing storage media shall be verified to ensure that any sensitive data and licensed software has been removed or securely overwritten prior to disposal or re-use. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-008 (Asset Management Policy), POL-015 (Media Protection Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Disposal procedures, Sanitization records, Certificate of destruction |
| **Last Reviewed** | 2026-02-06 |

---

## 6. A.8 Technological Controls

### A.8.1 User Endpoint Devices

| Attribute | Detail |
|-----------|--------|
| **Control** | Information stored on, processed by or accessible via user endpoint devices shall be protected. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-014 (Mobile Device and Remote Work Policy), POL-008 (Asset Management Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | MDM configurations, Encryption policies, EDR deployment |
| **Last Reviewed** | 2026-02-06 |

### A.8.2 Privileged Access Rights

| Attribute | Detail |
|-----------|--------|
| **Control** | The allocation and use of privileged access rights shall be restricted and managed. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-003 (Access Control Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | PAM system, Privileged access logs, JIT access records |
| **Last Reviewed** | 2026-02-06 |

### A.8.3 Information Access Restriction

| Attribute | Detail |
|-----------|--------|
| **Control** | Access to information and other associated assets shall be restricted in accordance with the established topic-specific policy on access control. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-003 (Access Control Policy), POL-002 (Data Classification Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | RBAC configurations, Access control lists, Data access logs |
| **Last Reviewed** | 2026-02-06 |

### A.8.4 Access to Source Code

| Attribute | Detail |
|-----------|--------|
| **Control** | Read and write access to source code, development tools and software libraries shall be appropriately managed. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-003 (Access Control Policy), POL-010 (SDLC Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | GitHub access controls, Branch protection, Code review requirements |
| **Last Reviewed** | 2026-02-06 |

### A.8.5 Secure Authentication

| Attribute | Detail |
|-----------|--------|
| **Control** | Secure authentication technologies and procedures shall be implemented based on information access restrictions and the topic-specific policy on access control. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-009 (Password and Authentication Policy), POL-003 (Access Control Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | MFA configurations, SSO setup, Authentication logs |
| **Last Reviewed** | 2026-02-06 |

### A.8.6 Capacity Management

| Attribute | Detail |
|-----------|--------|
| **Control** | The use of resources shall be monitored and adjusted in line with current and expected capacity requirements. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-007 (Business Continuity Policy), POL-011 (Logging and Monitoring Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Capacity dashboards, Auto-scaling configurations, Resource monitoring |
| **Last Reviewed** | 2026-02-06 |

### A.8.7 Protection Against Malware

| Attribute | Detail |
|-----------|--------|
| **Control** | Protection against malware shall be implemented and supported by appropriate user awareness. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-010 (SDLC Security Policy), POL-018 (Network Security Policy), POL-016 (Security Awareness Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | EDR deployment, Container scanning, Malware awareness training |
| **Last Reviewed** | 2026-02-06 |

### A.8.8 Management of Technical Vulnerabilities

| Attribute | Detail |
|-----------|--------|
| **Control** | Information about technical vulnerabilities of information systems in use shall be obtained, the organization's exposure to such vulnerabilities shall be evaluated and appropriate measures shall be taken. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-010 (SDLC Security Policy), POL-014 (Risk Management Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Vulnerability scans, Patch management, Remediation tracking |
| **Last Reviewed** | 2026-02-06 |

### A.8.9 Configuration Management

| Attribute | Detail |
|-----------|--------|
| **Control** | Configurations, including security configurations, of hardware, software, services and networks shall be established, documented, implemented, monitored and reviewed. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-008 (Asset Management Policy), POL-007 (Change Management Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Configuration baselines, IaC repositories, Configuration drift monitoring |
| **Last Reviewed** | 2026-02-06 |

### A.8.10 Information Deletion

| Attribute | Detail |
|-----------|--------|
| **Control** | Information stored in information systems, devices or in any other storage media shall be deleted when no longer required. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-005 (Data Retention Policy), POL-017 (Privacy Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Retention schedules, Deletion procedures, Deletion logs |
| **Last Reviewed** | 2026-02-06 |

### A.8.11 Data Masking

| Attribute | Detail |
|-----------|--------|
| **Control** | Data masking shall be used in accordance with the organization's topic-specific policy on access control and other related topic-specific policies, and business requirements, taking applicable legislation into consideration. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-002 (Data Classification Policy), POL-017 (Privacy Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Data masking rules, PII redaction in logs, Anonymization procedures |
| **Last Reviewed** | 2026-02-06 |

### A.8.12 Data Leakage Prevention

| Attribute | Detail |
|-----------|--------|
| **Control** | Data leakage prevention measures shall be applied to systems, networks and any other devices that process, store or transmit sensitive information. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-002 (Data Classification Policy), POL-011 (Logging and Monitoring Policy) |
| **Implementation Status** | Partially Implemented |
| **Evidence Location** | DLP configurations (partial), Egress monitoring, Alert rules |
| **Gap** | Full DLP solution deployment in progress |
| **Remediation** | Complete DLP implementation by Q3 2026 |
| **Last Reviewed** | 2026-02-06 |

### A.8.13 Information Backup

| Attribute | Detail |
|-----------|--------|
| **Control** | Backup copies of information, software and systems shall be maintained and regularly tested in accordance with the agreed topic-specific policy on backup. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-012 (Backup and Recovery Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Backup schedules, Backup logs, Restoration test results |
| **Last Reviewed** | 2026-02-06 |

### A.8.14 Redundancy of Information Processing Facilities

| Attribute | Detail |
|-----------|--------|
| **Control** | Information processing facilities shall be implemented with redundancy sufficient to meet availability requirements. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-007 (Business Continuity Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Multi-AZ architecture, DR infrastructure, Failover configurations |
| **Last Reviewed** | 2026-02-06 |

### A.8.15 Logging

| Attribute | Detail |
|-----------|--------|
| **Control** | Logs that record activities, exceptions, faults and other relevant events shall be produced, stored, protected and analysed. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-011 (Logging and Monitoring Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Log aggregation (Loki), Log retention, Log access controls |
| **Last Reviewed** | 2026-02-06 |

### A.8.16 Monitoring Activities

| Attribute | Detail |
|-----------|--------|
| **Control** | Networks, systems and applications shall be monitored for anomalous behaviour and appropriate actions taken to evaluate potential information security incidents. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-011 (Logging and Monitoring Policy), POL-006 (Incident Response Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Monitoring dashboards, Alert configurations, SIEM rules |
| **Last Reviewed** | 2026-02-06 |

### A.8.17 Clock Synchronization

| Attribute | Detail |
|-----------|--------|
| **Control** | The clocks of information processing systems used by the organization shall be synchronized to approved time sources. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-011 (Logging and Monitoring Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | NTP configurations, Time sync monitoring |
| **Last Reviewed** | 2026-02-06 |

### A.8.18 Use of Privileged Utility Programs

| Attribute | Detail |
|-----------|--------|
| **Control** | The use of utility programs that might be capable of overriding system and application controls shall be restricted and tightly controlled. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-003 (Access Control Policy), POL-010 (SDLC Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Utility restrictions, Admin tool logging, Access controls |
| **Last Reviewed** | 2026-02-06 |

### A.8.19 Installation of Software on Operational Systems

| Attribute | Detail |
|-----------|--------|
| **Control** | Procedures and measures shall be implemented to securely manage software installation on operational systems. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-007 (Change Management Policy), POL-008 (Asset Management Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Deployment pipelines, Change records, Software inventory |
| **Last Reviewed** | 2026-02-06 |

### A.8.20 Networks Security

| Attribute | Detail |
|-----------|--------|
| **Control** | Networks and network devices shall be secured, managed and controlled to protect information in systems and applications. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-018 (Network Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Network diagrams, Security groups, Network ACLs, VPC configurations |
| **Last Reviewed** | 2026-02-06 |

### A.8.21 Security of Network Services

| Attribute | Detail |
|-----------|--------|
| **Control** | Security mechanisms, service levels and service requirements of network services shall be identified, implemented and monitored. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-018 (Network Security Policy), POL-004 (Third-Party Risk Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Service agreements, SLA monitoring, Security configurations |
| **Last Reviewed** | 2026-02-06 |

### A.8.22 Segregation of Networks

| Attribute | Detail |
|-----------|--------|
| **Control** | Groups of information services, users and information systems shall be segregated in the organization's networks. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-018 (Network Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | VPC design, Subnet configurations, Security groups, Network policies |
| **Last Reviewed** | 2026-02-06 |

### A.8.23 Web Filtering

| Attribute | Detail |
|-----------|--------|
| **Control** | Access to external websites shall be managed to reduce exposure to malicious content. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-006 (Acceptable Use Policy), POL-018 (Network Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Web filtering configurations, Category blocks, Exception logs |
| **Last Reviewed** | 2026-02-06 |

### A.8.24 Use of Cryptography

| Attribute | Detail |
|-----------|--------|
| **Control** | Rules for the effective use of cryptography, including cryptographic key management, shall be defined and implemented. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-011 (Encryption Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Encryption standards, Key management procedures, KMS configurations |
| **Last Reviewed** | 2026-02-06 |

### A.8.25 Secure Development Life Cycle

| Attribute | Detail |
|-----------|--------|
| **Control** | Rules for the secure development of software and systems shall be established and applied. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-010 (SDLC Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | SDLC documentation, Security gates, Code review records |
| **Last Reviewed** | 2026-02-06 |

### A.8.26 Application Security Requirements

| Attribute | Detail |
|-----------|--------|
| **Control** | Information security requirements shall be identified, specified and approved when developing or acquiring applications. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-010 (SDLC Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Security requirements in stories, Threat models, Design reviews |
| **Last Reviewed** | 2026-02-06 |

### A.8.27 Secure System Architecture and Engineering Principles

| Attribute | Detail |
|-----------|--------|
| **Control** | Principles for engineering secure systems shall be established, documented, maintained and applied to any information system development activities. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-010 (SDLC Security Policy), POL-001 (Information Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Architecture standards, Security patterns, Reference architectures |
| **Last Reviewed** | 2026-02-06 |

### A.8.28 Secure Coding

| Attribute | Detail |
|-----------|--------|
| **Control** | Secure coding principles shall be applied to software development. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-010 (SDLC Security Policy), POL-016 (Security Awareness Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Secure coding standards, SAST results, Code reviews |
| **Last Reviewed** | 2026-02-06 |

### A.8.29 Security Testing in Development and Acceptance

| Attribute | Detail |
|-----------|--------|
| **Control** | Security testing processes shall be defined and implemented in the development life cycle. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-010 (SDLC Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Security test plans, SAST/DAST results, Penetration test reports |
| **Last Reviewed** | 2026-02-06 |

### A.8.30 Outsourced Development

| Attribute | Detail |
|-----------|--------|
| **Control** | The organization shall direct, monitor and review the activities related to outsourced system development. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-004 (Third-Party Risk Policy), POL-010 (SDLC Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Contractor agreements, Code review requirements, Security assessments |
| **Last Reviewed** | 2026-02-06 |

### A.8.31 Separation of Development, Test and Production Environments

| Attribute | Detail |
|-----------|--------|
| **Control** | Development, testing and production environments shall be separated and secured. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-010 (SDLC Security Policy), POL-003 (Access Control Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Environment architecture, Access controls by environment, Deployment pipelines |
| **Last Reviewed** | 2026-02-06 |

### A.8.32 Change Management

| Attribute | Detail |
|-----------|--------|
| **Control** | Changes to information processing facilities and information systems shall be subject to change management procedures. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-007 (Change Management Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Change records, CAB minutes, Deployment logs |
| **Last Reviewed** | 2026-02-06 |

### A.8.33 Test Information

| Attribute | Detail |
|-----------|--------|
| **Control** | Test information shall be appropriately selected, protected and managed. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-002 (Data Classification Policy), POL-010 (SDLC Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Test data procedures, Data anonymization, Test environment access |
| **Last Reviewed** | 2026-02-06 |

### A.8.34 Protection of Information Systems During Audit Testing

| Attribute | Detail |
|-----------|--------|
| **Control** | Audit tests and other assurance activities involving assessment of operational systems shall be planned and agreed between the tester and appropriate management. |
| **Applicable** | Yes |
| **Applicable Policies** | POL-014 (Risk Management Policy), POL-001 (Information Security Policy) |
| **Implementation Status** | Implemented |
| **Evidence Location** | Audit planning documents, Penetration test agreements, Scope documents |
| **Last Reviewed** | 2026-02-06 |

---

## 7. Non-Applicable Controls Justification

| Control | Justification |
|---------|---------------|
| A.7.6 Working in Secure Areas | GreenLang operates cloud-hosted infrastructure on AWS. No on-premises data centers or secure processing facilities requiring working-in-secure-area procedures. |
| A.7.8 Equipment Siting and Protection | All infrastructure hosted on AWS. Physical equipment siting and protection managed by AWS per their SOC 2 Type II certification. |
| A.7.11 Supporting Utilities | No on-premises data centers. AWS provides redundant power, cooling, and utilities per their certifications and SLAs. |
| A.7.12 Cabling Security | No on-premises infrastructure cabling. AWS manages all physical cabling security in their facilities. |
| A.7.13 Equipment Maintenance | Physical infrastructure maintenance managed by AWS. Endpoint maintenance covered under A.8.1 and asset management policy. |

---

## 8. Gap Analysis and Remediation

| Gap ID | Control | Gap Description | Owner | Due Date | Status | Priority |
|--------|---------|-----------------|-------|----------|--------|----------|
| GAP-001 | A.5.37 | Some operational procedures require documentation updates | Operations | Q2 2026 | In Progress | Medium |
| GAP-002 | A.8.12 | Full DLP solution deployment in progress | Security | Q3 2026 | In Progress | Medium |

---

## 9. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | Compliance Team | Initial mapping creation |

---

*This mapping is reviewed annually and updated when controls or policies change.*
