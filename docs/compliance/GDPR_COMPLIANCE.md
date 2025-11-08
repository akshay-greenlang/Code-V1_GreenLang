# GDPR Compliance Documentation for GreenLang

**Document Version:** 1.0
**Last Updated:** 2025-11-08
**Classification:** Internal - Compliance Documentation
**Owner:** Data Protection Officer (DPO)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [GDPR Overview](#gdpr-overview)
3. [Legal Basis for Processing](#legal-basis-for-processing)
4. [Data Subject Rights Implementation](#data-subject-rights-implementation)
5. [Consent Management Procedures](#consent-management-procedures)
6. [Data Breach Notification Process](#data-breach-notification-process)
7. [Data Protection Impact Assessment (DPIA)](#data-protection-impact-assessment-dpia)
8. [Records of Processing Activities](#records-of-processing-activities)
9. [Cross-Border Data Transfer Mechanisms](#cross-border-data-transfer-mechanisms)
10. [Privacy by Design and Default](#privacy-by-design-and-default)
11. [Data Retention and Deletion](#data-retention-and-deletion)
12. [Third-Party Data Processing](#third-party-data-processing)
13. [Appendices](#appendices)

---

## Executive Summary

### Purpose
This document demonstrates GreenLang's compliance with the EU General Data Protection Regulation (GDPR) (Regulation EU 2016/679). It outlines our data protection practices, procedures for honoring data subject rights, and technical and organizational measures implemented to protect personal data.

### Scope
- **Organization:** GreenLang Platform
- **Geographical Scope:** European Economic Area (EEA), United Kingdom, and worldwide customers
- **Data Subjects:** Platform users, employees, contractors, business contacts
- **Personal Data Categories:** Account information, usage data, carbon emissions data, employee data

### GDPR Compliance Status
- **DPO Appointed:** Yes
- **Privacy Policy Published:** Yes
- **Cookie Policy Published:** Yes
- **Data Processing Records:** Maintained
- **Data Subject Rights:** Implemented
- **Technical Measures:** Implemented
- **Organizational Measures:** Implemented

### Key Achievements
- Full data subject rights portal implemented
- Automated data breach detection and notification
- Privacy by design integrated into SDLC
- Comprehensive DPIAs for all high-risk processing
- Standard Contractual Clauses (SCCs) for data transfers
- Cookie consent management system
- Employee privacy training program

---

## GDPR Overview

### What is GDPR?

The General Data Protection Regulation (GDPR) is a comprehensive data protection law that came into effect on May 25, 2018. It applies to organizations that:
- Operate in the EU
- Offer goods/services to individuals in the EU
- Monitor behavior of individuals in the EU

### Key GDPR Principles (Article 5)

**1. Lawfulness, Fairness, and Transparency**
- Process personal data lawfully and fairly
- Be transparent about data processing
- Provide clear privacy information

**2. Purpose Limitation**
- Collect data for specified, explicit, legitimate purposes
- Not process data for incompatible purposes

**3. Data Minimization**
- Collect only data that is necessary
- Limit data collection to what is required

**4. Accuracy**
- Keep personal data accurate and up to date
- Correct or delete inaccurate data

**5. Storage Limitation**
- Retain data only as long as necessary
- Delete or anonymize when no longer needed

**6. Integrity and Confidentiality**
- Protect data with appropriate security measures
- Prevent unauthorized access, loss, or damage

**7. Accountability**
- Demonstrate compliance with GDPR
- Maintain documentation and records

---

## Legal Basis for Processing

### Article 6: Lawful Basis for Processing

GreenLang processes personal data based on one or more of the following legal bases:

#### 1. Consent (Article 6(1)(a))

**When Used:**
- Marketing communications
- Optional analytics cookies
- Newsletter subscriptions
- Beta program participation
- User research and feedback

**Requirements:**
- Freely given
- Specific and informed
- Unambiguous indication
- Easy to withdraw

**Implementation:**
```python
class ConsentManager:
    def __init__(self):
        self.consent_db = ConsentDatabase()

    def request_consent(self, data_subject_id, purpose, description):
        """Request consent from data subject"""
        consent_request = {
            'data_subject_id': data_subject_id,
            'purpose': purpose,
            'description': description,
            'requested_at': datetime.now(),
            'version': self.get_policy_version(purpose),
            'status': 'pending'
        }

        # Present consent request to user
        response = self.present_consent_form(
            data_subject_id=data_subject_id,
            purpose=purpose,
            description=description,
            granular_options=True,  # Granular consent
            easy_decline=True,  # As easy to decline as accept
            clear_language=True  # Plain language
        )

        if response == 'granted':
            consent_request['status'] = 'granted'
            consent_request['granted_at'] = datetime.now()
            consent_request['consent_proof'] = self.generate_consent_proof(response)
        else:
            consent_request['status'] = 'declined'
            consent_request['declined_at'] = datetime.now()

        # Store consent record
        self.consent_db.insert(consent_request)

        # Audit log
        audit_log.log_consent_request(consent_request)

        return consent_request['status']

    def withdraw_consent(self, data_subject_id, purpose):
        """Allow data subject to withdraw consent"""
        # Find active consent
        consent = self.consent_db.get_active_consent(data_subject_id, purpose)

        if not consent:
            return "No active consent found"

        # Withdraw consent
        consent['status'] = 'withdrawn'
        consent['withdrawn_at'] = datetime.now()

        # Update consent record
        self.consent_db.update(consent)

        # Stop processing based on this consent
        self.stop_processing(data_subject_id, purpose)

        # Audit log
        audit_log.log_consent_withdrawn(data_subject_id, purpose)

        # Notify data subject
        notify_consent_withdrawn(data_subject_id, purpose)

        return "Consent withdrawn successfully"

    def verify_consent(self, data_subject_id, purpose):
        """Verify valid consent exists before processing"""
        consent = self.consent_db.get_active_consent(data_subject_id, purpose)

        if not consent:
            return False

        if consent['status'] != 'granted':
            return False

        # Check consent hasn't expired
        if 'expires_at' in consent and consent['expires_at'] < datetime.now():
            return False

        # Check policy hasn't changed
        current_version = self.get_policy_version(purpose)
        if consent['version'] != current_version:
            # Re-request consent for updated policy
            return False

        return True
```

**Consent Records Maintained:**
- Who gave consent
- When consent was given
- What they were told
- How they gave consent
- Whether consent was withdrawn

---

#### 2. Contract (Article 6(1)(b))

**When Used:**
- User account creation
- Service provisioning
- Payment processing
- Customer support

**Justification:**
Processing is necessary to:
- Enter into a contract with the data subject
- Perform the contract (provide the service)

**Example:**
- User email address needed to create account and send login credentials
- Payment information needed to process subscription

---

#### 3. Legal Obligation (Article 6(1)(c))

**When Used:**
- Tax records
- Financial reporting
- Legal compliance
- Regulatory requirements

**Justification:**
Processing is necessary to comply with legal obligations such as:
- Tax laws
- Accounting regulations
- Employment laws
- Anti-money laundering regulations

---

#### 4. Legitimate Interests (Article 6(1)(f))

**When Used:**
- Fraud prevention
- Security monitoring
- System optimization
- Direct marketing to existing customers

**Justification:**
Processing is necessary for legitimate interests pursued by GreenLang or a third party, except where overridden by data subject interests or rights.

**Legitimate Interest Assessment (LIA):**
```python
class LegitimateInterestAssessment:
    def __init__(self):
        self.assessments = []

    def conduct_lia(self, processing_purpose):
        """Conduct Legitimate Interest Assessment"""
        assessment = {
            'purpose': processing_purpose,
            'date': datetime.now()
        }

        # Step 1: Purpose Test
        # Is there a legitimate interest?
        assessment['purpose_test'] = {
            'legitimate_interest': "Fraud prevention and security",
            'lawful': True,
            'sufficiently_clear': True
        }

        # Step 2: Necessity Test
        # Is processing necessary for that interest?
        assessment['necessity_test'] = {
            'necessary': True,
            'less_intrusive_alternatives': self.identify_alternatives(processing_purpose),
            'alternative_effective': False,  # No less intrusive alternative is equally effective
            'justification': "Processing is necessary to detect and prevent fraud"
        }

        # Step 3: Balancing Test
        # Do data subject interests override?
        assessment['balancing_test'] = {
            'data_subject_impact': "Minimal - automated analysis of transaction patterns",
            'reasonable_expectations': True,  # Users expect fraud protection
            'vulnerable_individuals': False,
            'safeguards': [
                "Data minimization",
                "Pseudonymization",
                "Access controls",
                "Retention limits"
            ],
            'balance': "Legitimate interest not overridden"
        }

        # Conclusion
        assessment['conclusion'] = self.determine_conclusion(assessment)

        # Store assessment
        self.assessments.append(assessment)

        # Document assessment
        self.document_lia(assessment)

        return assessment
```

**LIA Records:**
- Purpose of processing
- Legitimate interest identified
- Necessity assessment
- Balancing test
- Safeguards implemented
- Review dates

---

### Article 9: Special Categories of Personal Data

**Special categories** (sensitive data):
- Racial or ethnic origin
- Political opinions
- Religious or philosophical beliefs
- Trade union membership
- Genetic data
- Biometric data (for identification)
- Health data
- Sex life or sexual orientation

**GreenLang Position:**
- GreenLang does **NOT** intentionally collect or process special categories of personal data
- If inadvertently collected (e.g., in free-text fields), it is deleted immediately
- Employees instructed not to request such data
- Technical measures prevent collection (field validation, content filtering)

**If Processing Required:**
Would require explicit consent (Article 9(2)(a)) or other specific legal basis

---

## Data Subject Rights Implementation

### Article 15: Right of Access

**What it is:**
Data subjects have the right to obtain:
- Confirmation of processing
- Copy of personal data
- Information about processing

**Implementation:**

**Self-Service Access Portal:**
```python
class DataSubjectAccessPortal:
    def __init__(self):
        self.data_sources = self.load_data_sources()

    def handle_access_request(self, data_subject_id):
        """Handle Article 15 access request"""
        # Verify identity
        if not self.verify_identity(data_subject_id):
            return "Identity verification failed"

        # Collect data from all sources
        personal_data = self.collect_personal_data(data_subject_id)

        # Prepare data package
        data_package = {
            'data_subject_id': data_subject_id,
            'request_date': datetime.now(),
            'data': personal_data,
            'processing_information': self.get_processing_information(),
            'retention_periods': self.get_retention_periods(),
            'recipients': self.get_data_recipients(),
            'rights_information': self.get_rights_information()
        }

        # Generate human-readable report
        report = self.generate_access_report(data_package)

        # Generate machine-readable export (JSON, CSV)
        export = self.generate_data_export(data_package)

        # Log request
        audit_log.log_data_access_request(data_subject_id)

        # Provide data to subject
        return {
            'report': report,
            'export': export,
            'format': 'PDF and JSON'
        }

    def collect_personal_data(self, data_subject_id):
        """Collect all personal data across systems"""
        data = {}

        for source in self.data_sources:
            source_data = source.get_personal_data(data_subject_id)
            data[source.name] = source_data

        return data

    def get_processing_information(self):
        """Provide information about processing"""
        return {
            'purposes': [
                'Service provisioning',
                'Customer support',
                'Product improvement',
                'Legal compliance'
            ],
            'legal_basis': {
                'account_data': 'Contract (Article 6(1)(b))',
                'usage_analytics': 'Legitimate interest (Article 6(1)(f))',
                'marketing': 'Consent (Article 6(1)(a))'
            },
            'retention_periods': {
                'account_data': 'Duration of account + 1 year',
                'usage_data': '2 years',
                'support_tickets': '3 years'
            },
            'recipients': [
                'AWS (hosting provider)',
                'Stripe (payment processor)',
                'SendGrid (email service)'
            ],
            'transfers': {
                'countries': ['United States', 'EU'],
                'safeguards': 'Standard Contractual Clauses (SCCs)'
            },
            'automated_decision_making': 'None',
            'source': 'Provided by data subject',
            'dpo_contact': 'dpo@greenlang.io'
        }
```

**Response Timeframe:**
- **Without undue delay**
- **Within 1 month** of request
- **Extension:** Additional 2 months if complex (must inform data subject)

**Free of Charge:**
- First request is free
- Subsequent requests may be charged if manifestly unfounded or excessive

**Evidence Maintained:**
- Request receipts
- Identity verification records
- Data provided
- Delivery confirmation

---

### Article 16: Right to Rectification

**What it is:**
Data subjects have the right to:
- Correct inaccurate personal data
- Complete incomplete personal data

**Implementation:**

```python
class RectificationManager:
    def __init__(self):
        self.data_db = PersonalDataDatabase()

    def handle_rectification_request(self, data_subject_id, field, current_value, corrected_value):
        """Handle Article 16 rectification request"""
        # Verify identity
        if not self.verify_identity(data_subject_id):
            return "Identity verification failed"

        # Validate correction
        if not self.validate_correction(field, corrected_value):
            return "Invalid correction"

        # Create rectification request
        request = {
            'data_subject_id': data_subject_id,
            'field': field,
            'current_value': current_value,
            'corrected_value': corrected_value,
            'requested_at': datetime.now(),
            'status': 'pending'
        }

        # Some fields can be self-corrected
        if self.can_self_correct(field):
            self.apply_correction(data_subject_id, field, corrected_value)
            request['status'] = 'completed'
            request['completed_at'] = datetime.now()
        else:
            # Requires verification
            request['status'] = 'verification_required'
            self.request_verification_documents(data_subject_id, field)

        # Store request
        self.data_db.insert_request(request)

        # Audit log
        audit_log.log_rectification_request(request)

        return request

    def apply_correction(self, data_subject_id, field, corrected_value):
        """Apply correction to data"""
        # Update primary database
        self.data_db.update(
            data_subject_id=data_subject_id,
            field=field,
            value=corrected_value,
            updated_at=datetime.now()
        )

        # Notify recipients of correction (Article 19)
        self.notify_recipients_of_correction(data_subject_id, field, corrected_value)

        # Update backups (if technically possible)
        self.update_backups(data_subject_id, field, corrected_value)

        # Log correction
        audit_log.log_data_corrected(data_subject_id, field)
```

**Self-Service Correction:**
Users can correct via account settings:
- Name
- Email (with verification)
- Phone number
- Address
- Company information
- Preferences

**Manual Review Required:**
- Legal documents
- Financial information
- Historical records

---

### Article 17: Right to Erasure (Right to be Forgotten)

**What it is:**
Data subjects have the right to have their personal data erased when:
- Data no longer necessary for original purpose
- Consent is withdrawn (and no other legal basis)
- Data subject objects and no overriding grounds
- Data processed unlawfully
- Legal obligation to erase
- Data collected from children

**Limitations:**
Right to erasure does NOT apply when processing is necessary for:
- Freedom of expression and information
- Legal obligation
- Public interest
- Legal claims

**Implementation:**

```python
class ErasureManager:
    def __init__(self):
        self.data_db = PersonalDataDatabase()
        self.legal_holds = LegalHoldDatabase()

    def handle_erasure_request(self, data_subject_id):
        """Handle Article 17 erasure request"""
        # Verify identity
        if not self.verify_identity(data_subject_id):
            return "Identity verification failed"

        # Check if erasure is permitted
        erasure_check = self.check_erasure_permitted(data_subject_id)

        if not erasure_check['permitted']:
            # Explain why erasure cannot be performed
            return {
                'status': 'denied',
                'reason': erasure_check['reason'],
                'explanation': erasure_check['explanation']
            }

        # Create erasure request
        request = {
            'data_subject_id': data_subject_id,
            'requested_at': datetime.now(),
            'status': 'approved',
            'erasure_scope': self.determine_erasure_scope(data_subject_id)
        }

        # Execute erasure
        erasure_result = self.execute_erasure(data_subject_id)

        request['status'] = 'completed'
        request['completed_at'] = datetime.now()
        request['systems_erased'] = erasure_result['systems']

        # Store request record (anonymized)
        self.data_db.insert_erasure_request(request)

        # Audit log
        audit_log.log_erasure_request(request)

        return request

    def check_erasure_permitted(self, data_subject_id):
        """Check if erasure is legally permitted"""
        # Check for legal holds
        legal_hold = self.legal_holds.check(data_subject_id)
        if legal_hold:
            return {
                'permitted': False,
                'reason': 'legal_obligation',
                'explanation': 'Data retention required for legal claims or compliance'
            }

        # Check for contractual obligations
        active_contract = self.has_active_contract(data_subject_id)
        if active_contract:
            return {
                'permitted': False,
                'reason': 'contract_performance',
                'explanation': 'Data necessary to perform active contract'
            }

        # Check for legitimate interests that override
        overriding_interest = self.has_overriding_legitimate_interest(data_subject_id)
        if overriding_interest:
            return {
                'permitted': False,
                'reason': 'overriding_legitimate_interest',
                'explanation': overriding_interest['explanation']
            }

        # Erasure permitted
        return {'permitted': True}

    def execute_erasure(self, data_subject_id):
        """Execute erasure across all systems"""
        systems_erased = []

        # Primary database
        self.data_db.delete_personal_data(data_subject_id)
        systems_erased.append('primary_database')

        # Backups (mark for deletion on next backup cycle)
        self.mark_for_deletion_in_backups(data_subject_id)
        systems_erased.append('backups')

        # Cloud storage
        self.delete_from_cloud_storage(data_subject_id)
        systems_erased.append('cloud_storage')

        # Third-party systems
        third_parties = self.get_data_recipients(data_subject_id)
        for party in third_parties:
            self.request_third_party_deletion(party, data_subject_id)
            systems_erased.append(f'third_party_{party}')

        # Logs (pseudonymize - cannot delete for security)
        self.pseudonymize_logs(data_subject_id)
        systems_erased.append('logs_pseudonymized')

        # Notify recipients (Article 19)
        self.notify_recipients_of_erasure(data_subject_id)

        return {
            'systems': systems_erased,
            'completed_at': datetime.now()
        }

    def delete_with_retention(self, data_subject_id):
        """Soft delete with retention for legal compliance"""
        # Mark as deleted but retain for legal retention period
        self.data_db.update(
            data_subject_id=data_subject_id,
            status='deleted',
            deleted_at=datetime.now(),
            purge_after=datetime.now() + timedelta(days=self.get_retention_period())
        )

        # Pseudonymize immediately
        self.pseudonymize_record(data_subject_id)
```

**Erasure Process:**
1. Identity verification
2. Legality check
3. Data subject confirmation
4. Erasure execution (30 days)
5. Confirmation to data subject

**What Gets Erased:**
- Account data
- Profile information
- Usage data
- Communications
- Preferences
- Cookies

**What Gets Retained (with justification):**
- Financial records (tax law - 7 years)
- Legal claim data (statute of limitations)
- Aggregated analytics (anonymized)
- Security logs (pseudonymized)

---

### Article 18: Right to Restriction of Processing

**What it is:**
Data subjects can request restriction (instead of erasure) when:
- Accuracy is contested
- Processing is unlawful but data subject opposes erasure
- Data no longer needed but data subject needs it for legal claims
- Objection is pending verification

**During Restriction:**
- Data can be stored but not processed
- Processing only with consent or for legal claims

**Implementation:**

```python
class RestrictionManager:
    def __init__(self):
        self.data_db = PersonalDataDatabase()

    def handle_restriction_request(self, data_subject_id, reason, details):
        """Handle Article 18 restriction request"""
        # Verify identity
        if not self.verify_identity(data_subject_id):
            return "Identity verification failed"

        # Validate reason
        valid_reasons = [
            'accuracy_contested',
            'unlawful_processing',
            'legal_claims',
            'objection_pending'
        ]

        if reason not in valid_reasons:
            return "Invalid reason for restriction"

        # Create restriction
        restriction = {
            'data_subject_id': data_subject_id,
            'reason': reason,
            'details': details,
            'restricted_at': datetime.now(),
            'status': 'active'
        }

        # Apply restriction
        self.apply_restriction(data_subject_id, restriction)

        # Notify recipients (Article 19)
        self.notify_recipients_of_restriction(data_subject_id)

        # Store restriction
        self.data_db.insert_restriction(restriction)

        # Audit log
        audit_log.log_restriction_applied(data_subject_id, reason)

        # Notify data subject
        notify_restriction_applied(data_subject_id)

        return restriction

    def apply_restriction(self, data_subject_id, restriction):
        """Apply processing restriction"""
        # Mark data as restricted
        self.data_db.update(
            data_subject_id=data_subject_id,
            processing_restricted=True,
            restriction_reason=restriction['reason'],
            restricted_at=restriction['restricted_at']
        )

        # Prevent automated processing
        self.disable_automated_processing(data_subject_id)

        # Flag for manual review
        self.flag_for_manual_review(data_subject_id)

    def lift_restriction(self, data_subject_id, reason):
        """Lift processing restriction"""
        # Notify data subject before lifting (Article 18(3))
        self.notify_before_lifting_restriction(data_subject_id)

        # Wait for confirmation or objection
        time.sleep(timedelta(days=7))

        # Lift restriction
        self.data_db.update(
            data_subject_id=data_subject_id,
            processing_restricted=False,
            restriction_lifted_at=datetime.now(),
            lift_reason=reason
        )

        # Re-enable processing
        self.enable_processing(data_subject_id)

        # Audit log
        audit_log.log_restriction_lifted(data_subject_id, reason)
```

---

### Article 20: Right to Data Portability

**What it is:**
Data subjects have the right to:
- Receive personal data in structured, commonly used, machine-readable format
- Transmit data to another controller

**Applies when:**
- Processing based on consent or contract
- Processing is automated

**Implementation:**

```python
class DataPortabilityManager:
    def __init__(self):
        self.data_sources = self.load_data_sources()

    def handle_portability_request(self, data_subject_id, format='json'):
        """Handle Article 20 portability request"""
        # Verify identity
        if not self.verify_identity(data_subject_id):
            return "Identity verification failed"

        # Collect portable data
        portable_data = self.collect_portable_data(data_subject_id)

        # Export in requested format
        if format == 'json':
            export = self.export_as_json(portable_data)
        elif format == 'csv':
            export = self.export_as_csv(portable_data)
        elif format == 'xml':
            export = self.export_as_xml(portable_data)
        else:
            return "Unsupported format"

        # Create download package
        package = {
            'data_subject_id': data_subject_id,
            'export_date': datetime.now(),
            'format': format,
            'data': export,
            'schema': self.get_data_schema()
        }

        # Generate download link
        download_link = self.generate_secure_download_link(package)

        # Log request
        audit_log.log_portability_request(data_subject_id, format)

        # Notify data subject
        notify_portability_ready(data_subject_id, download_link)

        return package

    def collect_portable_data(self, data_subject_id):
        """Collect data subject to portability right"""
        portable_data = {}

        # Account information
        portable_data['account'] = {
            'user_id': data_subject_id,
            'email': self.get_email(data_subject_id),
            'name': self.get_name(data_subject_id),
            'created_at': self.get_account_creation_date(data_subject_id)
        }

        # User-provided data
        portable_data['profile'] = self.get_profile_data(data_subject_id)

        # Usage data (automated processing)
        portable_data['usage'] = self.get_usage_data(data_subject_id)

        # Carbon calculations
        portable_data['emissions'] = self.get_emissions_data(data_subject_id)

        # Preferences
        portable_data['preferences'] = self.get_preferences(data_subject_id)

        # Consent records
        portable_data['consents'] = self.get_consent_records(data_subject_id)

        return portable_data

    def export_as_json(self, data):
        """Export data in JSON format"""
        return json.dumps(data, indent=2, ensure_ascii=False)

    def export_as_csv(self, data):
        """Export data in CSV format"""
        # Flatten nested data
        flat_data = self.flatten_data(data)

        # Convert to CSV
        csv_output = io.StringIO()
        writer = csv.DictWriter(csv_output, fieldnames=flat_data[0].keys())
        writer.writeheader()
        writer.writerows(flat_data)

        return csv_output.getvalue()

    def direct_transfer(self, data_subject_id, target_controller):
        """Transfer data directly to another controller"""
        # Verify target controller
        if not self.verify_controller(target_controller):
            return "Invalid target controller"

        # Collect portable data
        portable_data = self.collect_portable_data(data_subject_id)

        # Encrypt data
        encrypted_data = self.encrypt_for_transfer(portable_data)

        # Transfer via secure API
        transfer_result = self.secure_transfer(
            target=target_controller,
            data=encrypted_data
        )

        # Audit log
        audit_log.log_direct_transfer(data_subject_id, target_controller)

        return transfer_result
```

**Supported Formats:**
- JSON (default)
- CSV
- XML

**Direct Transfer:**
- API available for direct transfer to another controller
- Secure encrypted transfer
- Transfer confirmation

---

### Article 21: Right to Object

**What it is:**
Data subjects can object to processing based on:
- Legitimate interests (Article 6(1)(f))
- Public interest tasks (Article 6(1)(e))
- Direct marketing (absolute right)
- Profiling

**Implementation:**

```python
class ObjectionManager:
    def __init__(self):
        self.processing_db = ProcessingDatabase()

    def handle_objection(self, data_subject_id, processing_purpose, grounds=None):
        """Handle Article 21 objection"""
        # Verify identity
        if not self.verify_identity(data_subject_id):
            return "Identity verification failed"

        # Check objection type
        if processing_purpose == 'direct_marketing':
            # Absolute right - stop immediately
            self.stop_direct_marketing(data_subject_id)
            return {
                'status': 'accepted',
                'reason': 'Direct marketing objection is absolute right'
            }

        # For other purposes, assess objection
        assessment = self.assess_objection(data_subject_id, processing_purpose, grounds)

        if assessment['compelling_grounds']:
            # GreenLang has compelling legitimate grounds
            return {
                'status': 'rejected',
                'reason': assessment['justification'],
                'right_to_complain': self.get_complaint_information()
            }
        else:
            # Stop processing
            self.stop_processing(data_subject_id, processing_purpose)
            return {
                'status': 'accepted',
                'stopped_processing': processing_purpose
            }

    def stop_direct_marketing(self, data_subject_id):
        """Stop all direct marketing"""
        # Unsubscribe from all marketing lists
        self.unsubscribe_all_marketing(data_subject_id)

        # Update marketing preferences
        self.update_preferences(
            data_subject_id=data_subject_id,
            marketing_emails=False,
            marketing_sms=False,
            marketing_phone=False
        )

        # Suppress from future marketing
        self.add_to_suppression_list(data_subject_id)

        # Audit log
        audit_log.log_marketing_objection(data_subject_id)

    def assess_objection(self, data_subject_id, processing_purpose, grounds):
        """Assess whether compelling legitimate grounds exist"""
        # Evaluate data subject's grounds
        ds_grounds_weight = self.evaluate_grounds_weight(grounds)

        # Evaluate GreenLang's legitimate grounds
        gl_grounds = self.get_legitimate_grounds(processing_purpose)
        gl_grounds_weight = self.evaluate_grounds_weight(gl_grounds)

        # Balancing test
        if gl_grounds_weight > ds_grounds_weight:
            return {
                'compelling_grounds': True,
                'justification': gl_grounds['justification']
            }
        else:
            return {
                'compelling_grounds': False
            }
```

**Direct Marketing Opt-Out:**
- Honored immediately
- No justification required
- All marketing channels stopped
- Permanent suppression

---

### Article 22: Automated Decision-Making and Profiling

**What it is:**
Data subjects have the right not to be subject to automated decisions with legal or similarly significant effects, unless:
- Necessary for contract
- Authorized by law
- Based on explicit consent

**GreenLang Position:**
- GreenLang does NOT engage in automated decision-making with legal or significant effects
- Carbon calculations are automated but reviewed by users
- No credit scoring, employment decisions, or similar automated decisions

**If Implemented in Future:**
Would require:
- Explicit consent
- Meaningful information about logic
- Right to human intervention
- Right to contest decision
- Regular accuracy checks

---

## Consent Management Procedures

### Cookie Consent

**Cookie Banner Implementation:**
```javascript
// Cookie consent manager
class CookieConsentManager {
    constructor() {
        this.consentGiven = this.loadConsent();
    }

    showBanner() {
        // Show cookie banner on first visit
        if (!this.consentGiven) {
            this.displayBanner({
                message: "We use cookies to provide essential functionality and improve your experience.",
                categories: [
                    {
                        name: "Essential",
                        description: "Required for site functionality",
                        required: true,
                        cookies: ["session_id", "csrf_token"]
                    },
                    {
                        name: "Analytics",
                        description: "Help us understand how you use the site",
                        required: false,
                        cookies: ["_ga", "_gid"]
                    },
                    {
                        name: "Marketing",
                        description: "Used for targeted advertising",
                        required: false,
                        cookies: ["_fbp", "ads_id"]
                    }
                ],
                granular: true,  // Allow granular control
                easyDecline: true  // "Reject All" button prominent
            });
        }
    }

    acceptAll() {
        this.saveConsent({
            essential: true,
            analytics: true,
            marketing: true,
            timestamp: new Date(),
            version: this.getPolicyVersion()
        });

        this.loadCookies(['essential', 'analytics', 'marketing']);
    }

    acceptSelected(categories) {
        this.saveConsent({
            essential: true,  // Always required
            analytics: categories.includes('analytics'),
            marketing: categories.includes('marketing'),
            timestamp: new Date(),
            version: this.getPolicyVersion()
        });

        this.loadCookies(categories);
    }

    rejectAll() {
        this.saveConsent({
            essential: true,  // Only essential
            analytics: false,
            marketing: false,
            timestamp: new Date(),
            version: this.getPolicyVersion()
        });

        this.loadCookies(['essential']);
    }

    withdrawConsent() {
        // Allow easy withdrawal
        this.deleteAllCookies(['analytics', 'marketing']);
        this.rejectAll();
    }
}
```

**Cookie Policy Requirements:**
- List all cookies used
- Purpose of each cookie
- Duration of each cookie
- Third-party cookies disclosed
- How to manage cookies
- Links to third-party privacy policies

---

### Marketing Consent

**Email Marketing:**
```python
class MarketingConsentManager:
    def __init__(self):
        self.consent_db = ConsentDatabase()
        self.email_service = EmailService()

    def request_marketing_consent(self, email, source):
        """Request marketing consent via double opt-in"""
        # Create consent request
        consent_token = self.generate_token()

        request = {
            'email': email,
            'source': source,
            'token': consent_token,
            'requested_at': datetime.now(),
            'status': 'pending_confirmation'
        }

        # Send confirmation email (double opt-in)
        self.email_service.send(
            to=email,
            subject="Please confirm your subscription",
            template="double_opt_in",
            data={
                'confirmation_link': f"https://greenlang.io/confirm/{consent_token}"
            }
        )

        # Store request
        self.consent_db.insert(request)

        # Audit log
        audit_log.log_marketing_consent_requested(email, source)

    def confirm_marketing_consent(self, token):
        """Confirm marketing consent (double opt-in completion)"""
        # Find consent request
        request = self.consent_db.get_by_token(token)

        if not request:
            return "Invalid token"

        # Check token expiration (24 hours)
        if datetime.now() - request['requested_at'] > timedelta(hours=24):
            return "Token expired"

        # Confirm consent
        request['status'] = 'confirmed'
        request['confirmed_at'] = datetime.now()

        # Update consent record
        self.consent_db.update(request)

        # Add to marketing list
        self.email_service.add_to_list(request['email'])

        # Audit log
        audit_log.log_marketing_consent_confirmed(request['email'])

        # Send welcome email
        self.email_service.send_welcome(request['email'])

        return "Consent confirmed"

    def unsubscribe(self, email, reason=None):
        """Handle unsubscribe request"""
        # Remove from marketing lists
        self.email_service.remove_from_lists(email)

        # Update consent status
        consent = self.consent_db.get_by_email(email)
        consent['status'] = 'withdrawn'
        consent['withdrawn_at'] = datetime.now()
        consent['withdraw_reason'] = reason

        self.consent_db.update(consent)

        # Add to suppression list (permanent)
        self.email_service.add_to_suppression_list(email)

        # Audit log
        audit_log.log_marketing_unsubscribe(email, reason)

        # Send unsubscribe confirmation
        self.email_service.send_unsubscribe_confirmation(email)
```

**Marketing Consent Requirements:**
- Double opt-in for email marketing
- Granular consent (email, SMS, phone separately)
- Easy unsubscribe (one-click)
- Unsubscribe link in every email
- No pre-checked boxes
- Clear and specific language

---

## Data Breach Notification Process

### Article 33: Notification to Supervisory Authority

**Requirements:**
- **Timeframe:** Within 72 hours of becoming aware
- **Threshold:** Breach likely to result in risk to rights and freedoms
- **Information to Include:**
  - Nature of breach
  - Categories and numbers of data subjects
  - Categories and numbers of records
  - Likely consequences
  - Measures taken or proposed
  - DPO contact details

**Breach Detection and Response:**
```python
class DataBreachManager:
    def __init__(self):
        self.breach_db = BreachDatabase()
        self.dpo = DataProtectionOfficer()
        self.supervisory_authority = SupervisoryAuthority()

    def detect_breach(self, event):
        """Detect potential data breach"""
        # Classify event
        if self.is_data_breach(event):
            # Create breach record
            breach = {
                'detected_at': datetime.now(),
                'event': event,
                'status': 'investigating',
                'severity': 'unknown'
            }

            # Immediate notification to DPO
            self.dpo.notify_potential_breach(breach)

            # Start investigation
            investigation = self.initiate_investigation(breach)

            # Activate incident response
            self.activate_incident_response(breach)

            return breach

    def assess_breach(self, breach_id):
        """Assess breach severity and risk"""
        breach = self.breach_db.get(breach_id)

        # Assess risk to data subjects
        risk_assessment = {
            'likelihood_of_harm': self.assess_likelihood_of_harm(breach),
            'severity_of_harm': self.assess_severity_of_harm(breach),
            'data_categories': self.identify_data_categories(breach),
            'data_subjects_affected': self.count_affected_subjects(breach),
            'safeguards': self.identify_safeguards(breach)
        }

        # Determine risk level
        if risk_assessment['likelihood_of_harm'] == 'high' and risk_assessment['severity_of_harm'] == 'high':
            risk_level = 'high'
        elif risk_assessment['likelihood_of_harm'] >= 'medium' or risk_assessment['severity_of_harm'] >= 'medium':
            risk_level = 'medium'
        else:
            risk_level = 'low'

        # Update breach record
        breach['risk_assessment'] = risk_assessment
        breach['risk_level'] = risk_level

        # Determine notification requirements
        breach['notification_required'] = self.determine_notification_requirements(breach)

        self.breach_db.update(breach)

        return breach

    def notify_supervisory_authority(self, breach_id):
        """Notify supervisory authority within 72 hours (Article 33)"""
        breach = self.breach_db.get(breach_id)

        # Check if notification required
        if not breach['notification_required']['supervisory_authority']:
            return "Notification not required"

        # Check 72-hour deadline
        time_since_awareness = datetime.now() - breach['detected_at']

        if time_since_awareness > timedelta(hours=72):
            # Late notification - must explain delay
            delay_justification = self.document_delay_justification(breach)
        else:
            delay_justification = None

        # Prepare notification
        notification = {
            'breach_id': breach_id,
            'nature_of_breach': breach['description'],
            'data_categories': breach['risk_assessment']['data_categories'],
            'data_subjects_affected': breach['risk_assessment']['data_subjects_affected'],
            'likely_consequences': breach['risk_assessment']['likely_consequences'],
            'measures_taken': breach['mitigation_measures'],
            'dpo_contact': self.dpo.get_contact_details(),
            'notification_date': datetime.now(),
            'delay_justification': delay_justification
        }

        # Submit notification
        confirmation = self.supervisory_authority.submit_notification(notification)

        # Update breach record
        breach['supervisory_authority_notified'] = True
        breach['supervisory_authority_notification_date'] = datetime.now()
        breach['supervisory_authority_confirmation'] = confirmation

        self.breach_db.update(breach)

        # Audit log
        audit_log.log_breach_notification_supervisory_authority(breach_id)

        return confirmation

    def assess_likelihood_of_harm(self, breach):
        """Assess likelihood of harm to data subjects"""
        factors = []

        # Nature of data
        if breach['data_type'] in ['financial', 'health', 'biometric']:
            factors.append('high_sensitivity_data')

        # Encryption
        if not breach['data_encrypted']:
            factors.append('unencrypted_data')

        # Threat actor
        if breach['threat_actor'] == 'malicious':
            factors.append('malicious_actor')

        # Data volume
        if breach['records_affected'] > 10000:
            factors.append('large_volume')

        # Determine likelihood
        if len(factors) >= 3:
            return 'high'
        elif len(factors) >= 1:
            return 'medium'
        else:
            return 'low'

    def assess_severity_of_harm(self, breach):
        """Assess severity of potential harm"""
        potential_harms = []

        # Identify potential harms
        if breach['data_type'] == 'financial':
            potential_harms.append({
                'type': 'financial_loss',
                'severity': 'high'
            })

        if breach['data_type'] == 'credentials':
            potential_harms.append({
                'type': 'account_takeover',
                'severity': 'high'
            })

        if breach['data_type'] == 'personal':
            potential_harms.append({
                'type': 'identity_theft',
                'severity': 'medium'
            })

        # Determine overall severity
        if any(h['severity'] == 'high' for h in potential_harms):
            return 'high'
        elif any(h['severity'] == 'medium' for h in potential_harms):
            return 'medium'
        else:
            return 'low'
```

**Breach Notification Timeline:**
```
Hour 0: Breach detected
↓
Hour 1: DPO notified, incident response activated
↓
Hour 4: Initial assessment complete
↓
Hour 24: Risk assessment complete
↓
Hour 48: Mitigation measures implemented
↓
Hour 72: Supervisory authority notification (if required)
↓
Hour 72-96: Data subject notification (if required)
```

---

### Article 34: Notification to Data Subjects

**Requirements:**
- **When:** Breach likely to result in **high risk** to rights and freedoms
- **Timeframe:** Without undue delay
- **Method:** Direct communication to each affected data subject
- **Information:**
  - Nature of breach
  - DPO contact details
  - Likely consequences
  - Measures taken or proposed
  - Recommended actions for data subjects

**Exceptions (no notification required if):**
- Encryption or other safeguards render data unintelligible
- Subsequent measures ensure high risk no longer likely
- Notification would involve disproportionate effort (can use public communication)

**Implementation:**
```python
def notify_data_subjects(self, breach_id):
    """Notify data subjects of breach (Article 34)"""
    breach = self.breach_db.get(breach_id)

    # Check if notification required
    if not breach['notification_required']['data_subjects']:
        return "Notification not required"

    # Check exceptions
    if breach['data_encrypted'] and breach['encryption_key_secure']:
        return "Exception: Data encrypted and unintelligible"

    if breach['risk_mitigated']:
        return "Exception: Risk mitigated by subsequent measures"

    # Get affected data subjects
    affected_subjects = self.get_affected_data_subjects(breach_id)

    # Prepare notification
    for subject in affected_subjects:
        notification = {
            'to': subject.email,
            'subject': 'Important Security Notice',
            'template': 'data_breach_notification',
            'data': {
                'nature_of_breach': breach['description_plain_language'],
                'data_affected': breach['data_categories_affected'],
                'likely_consequences': breach['risk_assessment']['likely_consequences'],
                'measures_taken': breach['mitigation_measures'],
                'recommended_actions': self.get_recommended_actions(breach),
                'dpo_contact': self.dpo.get_contact_details(),
                'support': 'support@greenlang.io'
            }
        }

        # Send notification
        self.email_service.send(notification)

        # Log notification
        audit_log.log_breach_notification_data_subject(subject.id, breach_id)

    # Update breach record
    breach['data_subjects_notified'] = True
    breach['data_subject_notification_date'] = datetime.now()
    breach['data_subjects_notified_count'] = len(affected_subjects)

    self.breach_db.update(breach)

    return f"Notified {len(affected_subjects)} data subjects"
```

**Recommended Actions for Data Subjects:**
- Change passwords immediately
- Enable MFA if not already
- Monitor accounts for suspicious activity
- Review recent account activity
- Contact support with questions
- Consider credit monitoring (if financial data affected)

---

## Data Protection Impact Assessment (DPIA)

### Article 35: DPIA Requirements

**When Required:**
DPIA mandatory when processing is likely to result in high risk, particularly:
- Systematic and extensive automated processing with legal effects
- Large-scale processing of special categories of data
- Systematic monitoring of publicly accessible areas on a large scale

**GreenLang DPIA Triggers:**
- New technology implementation
- Large-scale data processing
- Automated decision-making
- Special categories of data
- Cross-border data transfers
- Vulnerable data subjects (children)
- Innovative use of data

**DPIA Process:**
```python
class DPIAManager:
    def __init__(self):
        self.dpia_db = DPIADatabase()
        self.dpo = DataProtectionOfficer()

    def assess_dpia_required(self, processing_activity):
        """Determine if DPIA is required"""
        triggers = []

        # Check DPIA triggers
        if processing_activity['automated_decision_making']:
            triggers.append('automated_decisions')

        if processing_activity['special_categories']:
            triggers.append('special_categories')

        if processing_activity['large_scale']:
            triggers.append('large_scale')

        if processing_activity['systematic_monitoring']:
            triggers.append('systematic_monitoring')

        if processing_activity['vulnerable_subjects']:
            triggers.append('vulnerable_subjects')

        if processing_activity['new_technology']:
            triggers.append('new_technology')

        # DPIA required if any trigger present
        return {
            'required': len(triggers) > 0,
            'triggers': triggers
        }

    def conduct_dpia(self, processing_activity):
        """Conduct Data Protection Impact Assessment"""
        dpia = {
            'processing_activity': processing_activity,
            'started_at': datetime.now(),
            'status': 'in_progress'
        }

        # Step 1: Describe processing
        dpia['description'] = {
            'nature': processing_activity['description'],
            'scope': processing_activity['scope'],
            'context': processing_activity['context'],
            'purposes': processing_activity['purposes']
        }

        # Step 2: Assess necessity and proportionality
        dpia['necessity'] = self.assess_necessity(processing_activity)

        # Step 3: Assess risks to data subjects
        dpia['risks'] = self.identify_risks(processing_activity)

        # Step 4: Identify measures to address risks
        dpia['measures'] = self.identify_measures(dpia['risks'])

        # Step 5: DPO consultation
        dpia['dpo_opinion'] = self.dpo.review_dpia(dpia)

        # Step 6: Data subject consultation (if appropriate)
        if self.should_consult_data_subjects(processing_activity):
            dpia['data_subject_feedback'] = self.consult_data_subjects(processing_activity)

        # Step 7: Residual risk assessment
        dpia['residual_risks'] = self.assess_residual_risks(dpia['risks'], dpia['measures'])

        # Step 8: Prior consultation with SA if high residual risk
        if any(r['level'] == 'high' for r in dpia['residual_risks']):
            dpia['prior_consultation'] = self.consult_supervisory_authority(dpia)

        # Finalize DPIA
        dpia['status'] = 'completed'
        dpia['completed_at'] = datetime.now()
        dpia['approved_by'] = self.dpo.email

        # Store DPIA
        self.dpia_db.insert(dpia)

        return dpia

    def identify_risks(self, processing_activity):
        """Identify risks to data subjects"""
        risks = []

        # Risk: Unauthorized access
        if processing_activity['data_sensitivity'] >= 'confidential':
            risks.append({
                'risk': 'Unauthorized access to personal data',
                'likelihood': 'medium',
                'severity': 'high',
                'overall': 'high',
                'affected_rights': ['confidentiality', 'security']
            })

        # Risk: Data breach
        if processing_activity['data_volume'] == 'large_scale':
            risks.append({
                'risk': 'Large-scale data breach',
                'likelihood': 'low',
                'severity': 'critical',
                'overall': 'high',
                'affected_rights': ['confidentiality', 'security']
            })

        # Risk: Function creep
        if processing_activity['data_reuse']:
            risks.append({
                'risk': 'Data used for incompatible purposes',
                'likelihood': 'medium',
                'severity': 'medium',
                'overall': 'medium',
                'affected_rights': ['purpose_limitation']
            })

        # Risk: Discrimination
        if processing_activity['automated_decision_making']:
            risks.append({
                'risk': 'Discriminatory outcomes from automated processing',
                'likelihood': 'medium',
                'severity': 'high',
                'overall': 'high',
                'affected_rights': ['fairness', 'non_discrimination']
            })

        return risks

    def identify_measures(self, risks):
        """Identify measures to mitigate risks"""
        measures = []

        for risk in risks:
            if risk['risk'] == 'Unauthorized access to personal data':
                measures.append({
                    'risk': risk['risk'],
                    'measures': [
                        'Implement encryption (AES-256)',
                        'Enforce MFA for all users',
                        'Implement RBAC',
                        'Regular access reviews',
                        'Audit logging'
                    ]
                })

            if risk['risk'] == 'Large-scale data breach':
                measures.append({
                    'risk': risk['risk'],
                    'measures': [
                        'Implement DLP controls',
                        'Network segmentation',
                        'Intrusion detection',
                        'Incident response plan',
                        'Breach notification procedures',
                        'Cyber insurance'
                    ]
                })

            if risk['risk'] == 'Data used for incompatible purposes':
                measures.append({
                    'risk': risk['risk'],
                    'measures': [
                        'Clear purpose specification',
                        'Access controls by purpose',
                        'Regular purpose audits',
                        'Data subject consent for new purposes'
                    ]
                })

            if risk['risk'] == 'Discriminatory outcomes from automated processing':
                measures.append({
                    'risk': risk['risk'],
                    'measures': [
                        'Bias testing of algorithms',
                        'Human review of decisions',
                        'Right to explanation',
                        'Regular fairness audits'
                    ]
                })

        return measures
```

**DPIA Template:**
See Appendix: DPIA Template

---

## Records of Processing Activities

### Article 30: Records of Processing

**Requirements:**
Organizations must maintain records of all processing activities, including:
- Name and contact details of controller/processor
- Purposes of processing
- Categories of data subjects
- Categories of personal data
- Categories of recipients
- Transfers to third countries
- Retention periods
- Security measures

**Implementation:**
```python
class ProcessingRecordsManager:
    def __init__(self):
        self.records_db = ProcessingRecordsDatabase()

    def create_processing_record(self, activity):
        """Create Article 30 processing record"""
        record = {
            'controller': {
                'name': 'GreenLang Inc.',
                'address': '123 Green Street, San Francisco, CA 94105',
                'contact': 'privacy@greenlang.io',
                'dpo': {
                    'name': 'Jane Smith',
                    'email': 'dpo@greenlang.io'
                }
            },
            'processing_activity': {
                'name': activity['name'],
                'purposes': activity['purposes'],
                'legal_basis': activity['legal_basis'],
                'legitimate_interests': activity.get('legitimate_interests')
            },
            'data_subjects': {
                'categories': activity['data_subject_categories'],
                'number': activity.get('data_subject_count')
            },
            'personal_data': {
                'categories': activity['data_categories'],
                'special_categories': activity.get('special_categories', [])
            },
            'recipients': {
                'categories': activity['recipient_categories'],
                'third_parties': activity.get('third_parties', [])
            },
            'transfers': {
                'third_countries': activity.get('third_countries', []),
                'safeguards': activity.get('transfer_safeguards')
            },
            'retention': {
                'periods': activity['retention_periods'],
                'criteria': activity['retention_criteria']
            },
            'security_measures': activity['security_measures'],
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        }

        # Store record
        self.records_db.insert(record)

        return record

    def generate_ropa_report(self):
        """Generate Records of Processing Activities (RoPA) report"""
        # Get all processing activities
        records = self.records_db.get_all()

        # Generate report
        report = {
            'generated_at': datetime.now(),
            'controller': records[0]['controller'],
            'processing_activities': []
        }

        for record in records:
            activity_summary = {
                'name': record['processing_activity']['name'],
                'purposes': record['processing_activity']['purposes'],
                'legal_basis': record['processing_activity']['legal_basis'],
                'data_subjects': record['data_subjects']['categories'],
                'data_categories': record['personal_data']['categories'],
                'recipients': record['recipients']['categories'],
                'transfers': record['transfers']['third_countries'],
                'retention': record['retention']['periods'],
                'security': record['security_measures']
            }

            report['processing_activities'].append(activity_summary)

        return report
```

**GreenLang Processing Activities (Examples):**

**1. User Account Management**
- **Purpose:** Provide platform access
- **Legal Basis:** Contract (Article 6(1)(b))
- **Data Subjects:** Platform users
- **Data Categories:** Name, email, password hash, account settings
- **Recipients:** AWS (hosting), SendGrid (email)
- **Retention:** Duration of account + 1 year
- **Security:** Encryption at rest, MFA, RBAC

**2. Carbon Emissions Calculations**
- **Purpose:** Calculate and track carbon emissions
- **Legal Basis:** Contract (Article 6(1)(b))
- **Data Subjects:** Platform users
- **Data Categories:** Energy usage data, fuel consumption, travel data
- **Recipients:** AWS (hosting)
- **Retention:** Duration of account + 1 year
- **Security:** Encryption at rest, access controls

**3. Customer Support**
- **Purpose:** Provide customer support
- **Legal Basis:** Legitimate interest (Article 6(1)(f))
- **Data Subjects:** Support ticket submitters
- **Data Categories:** Name, email, support inquiry, conversation history
- **Recipients:** Zendesk (ticketing system)
- **Retention:** 3 years
- **Security:** Encryption in transit, access controls

**Complete RoPA Available:** See Appendix B

---

## Cross-Border Data Transfer Mechanisms

### Chapter V: Transfers to Third Countries

**GDPR Requirements:**
- Can only transfer personal data to third countries if adequate protection ensured
- Adequacy decision by EU Commission (Article 45)
- OR Appropriate safeguards (Article 46)

**GreenLang Data Transfers:**

**Primary Storage:**
- **Location:** EU region (AWS eu-west-1, Frankfurt)
- **No transfer:** Data stays in EU

**Backups:**
- **Location:** EU region (AWS eu-west-2, London)
- **No transfer:** Backups stay in EU

**Service Providers:**

**AWS (United States):**
- **Safeguard:** Standard Contractual Clauses (SCCs)
- **Additional Measures:** Encryption, access controls
- **Transfer Impact Assessment:** Conducted

**SendGrid (United States):**
- **Safeguard:** Standard Contractual Clauses (SCCs)
- **Purpose:** Transactional emails only
- **Data Minimized:** Only email address and name

**Stripe (United States):**
- **Safeguard:** Standard Contractual Clauses (SCCs)
- **Purpose:** Payment processing
- **Data Minimized:** Payment information only (tokenized)

**Standard Contractual Clauses (SCCs):**
```python
class DataTransferManager:
    def __init__(self):
        self.scc_db = SCCDatabase()
        self.transfers = []

    def execute_data_transfer(self, data, recipient, purpose):
        """Execute data transfer with appropriate safeguards"""
        # Check if transfer to third country
        if self.is_third_country(recipient.country):
            # Check adequacy decision
            if not self.has_adequacy_decision(recipient.country):
                # Require appropriate safeguards
                safeguards = self.get_transfer_safeguards(recipient)

                if not safeguards:
                    raise ValueError("No appropriate safeguards for transfer")

                # Conduct Transfer Impact Assessment (TIA)
                tia = self.conduct_transfer_impact_assessment(recipient, safeguards)

                if tia['risk_level'] == 'high':
                    # Additional measures required
                    additional_measures = self.identify_additional_measures(tia)
                    self.implement_additional_measures(additional_measures)

        # Execute transfer
        transfer_result = self.transfer_data(data, recipient, purpose)

        # Log transfer
        audit_log.log_data_transfer(recipient, purpose, len(data))

        return transfer_result

    def conduct_transfer_impact_assessment(self, recipient, safeguards):
        """Assess risks of international data transfer (Schrems II)"""
        tia = {
            'recipient': recipient,
            'safeguards': safeguards,
            'assessed_at': datetime.now()
        }

        # Assess legal framework in destination country
        tia['legal_framework'] = self.assess_legal_framework(recipient.country)

        # Assess surveillance laws
        tia['surveillance_risk'] = self.assess_surveillance_risk(recipient.country)

        # Assess recipient practices
        tia['recipient_practices'] = self.assess_recipient_practices(recipient)

        # Assess effectiveness of safeguards
        tia['safeguard_effectiveness'] = self.assess_safeguard_effectiveness(
            safeguards,
            tia['legal_framework'],
            tia['surveillance_risk']
        )

        # Determine risk level
        if tia['surveillance_risk'] == 'high' and tia['safeguard_effectiveness'] == 'low':
            tia['risk_level'] = 'high'
        elif tia['surveillance_risk'] == 'medium' or tia['safeguard_effectiveness'] == 'medium':
            tia['risk_level'] = 'medium'
        else:
            tia['risk_level'] = 'low'

        # Document TIA
        self.document_tia(tia)

        return tia

    def identify_additional_measures(self, tia):
        """Identify supplementary measures to SCCs"""
        measures = []

        # Technical measures
        measures.append({
            'type': 'technical',
            'measure': 'End-to-end encryption',
            'description': 'Encrypt data before transfer, recipient cannot decrypt'
        })

        measures.append({
            'type': 'technical',
            'measure': 'Pseudonymization',
            'description': 'Replace identifiers with pseudonyms'
        })

        # Organizational measures
        measures.append({
            'type': 'organizational',
            'measure': 'Data minimization',
            'description': 'Transfer only necessary data'
        })

        measures.append({
            'type': 'organizational',
            'measure': 'Contractual obligations',
            'description': 'Additional contractual restrictions on use'
        })

        # Transparency measures
        measures.append({
            'type': 'transparency',
            'measure': 'Transfer notification',
            'description': 'Notify data subjects of transfers'
        })

        return measures
```

---

## Privacy by Design and Default

### Article 25: Privacy by Design and Default

**Privacy by Design:**
Integrate data protection into processing activities and business practices from the design stage.

**Privacy by Default:**
Process only personal data necessary for specific purpose, by default.

**Implementation in GreenLang:**

**1. Privacy in Software Development:**
```python
# Privacy design patterns

# 1. Data Minimization
class UserRegistration:
    def collect_user_info(self):
        # Only collect necessary fields
        required_fields = ['email', 'password']  # Minimal
        optional_fields = []  # No unnecessary fields

        # Don't collect: birth date, gender, phone (not needed)

# 2. Purpose Binding
class DataUsage:
    def __init__(self, data, purpose):
        self.data = data
        self.allowed_purpose = purpose

    def use_data(self, requested_purpose):
        if requested_purpose != self.allowed_purpose:
            raise ValueError("Data cannot be used for incompatible purpose")

# 3. Encryption by Default
class DataStorage:
    def store(self, data):
        # Always encrypt before storage
        encrypted_data = self.encrypt(data)
        database.insert(encrypted_data)

# 4. Pseudonymization
class Analytics:
    def track_event(self, user_id, event):
        # Use pseudonymous ID for analytics
        pseudonym = self.generate_pseudonym(user_id)
        analytics_db.insert({'user': pseudonym, 'event': event})

# 5. Access Control by Default
class NewUser:
    def __init__(self):
        self.permissions = ['read_own_data']  # Minimal by default
        # No admin, no access to others' data

# 6. Secure by Default
class APIEndpoint:
    def __init__(self):
        self.require_authentication = True  # Always
        self.require_https = True  # Always
        self.rate_limited = True  # Always
```

**2. Privacy in Architecture:**
- **Separation of data:** PII stored separately from business data
- **Encryption layers:** Multiple layers of encryption
- **Network segmentation:** Data tier isolated
- **Anonymization pipelines:** Auto-anonymize for analytics

**3. Privacy in Operations:**
- **Automated data deletion:** Expired data auto-deleted
- **Consent management:** Built into all data collection
- **Access logging:** All access logged and monitored
- **Regular privacy reviews:** Quarterly privacy audits

---

## Data Retention and Deletion

### Retention Policy

**GreenLang Retention Periods:**

| Data Category | Retention Period | Justification | Deletion Method |
|---------------|------------------|---------------|-----------------|
| Account data | Account lifetime + 1 year | Contract performance, legal claims | Secure deletion |
| Usage logs | 2 years | Security, debugging | Automated deletion |
| Financial records | 7 years | Tax law compliance | Secure deletion after retention |
| Support tickets | 3 years | Service improvement | Automated deletion |
| Marketing data | Until consent withdrawn | Consent | Immediate deletion on withdrawal |
| Backups | 90 days | Disaster recovery | Automatic overwrite |
| Security logs | 1 year | Security monitoring | Automated deletion |
| Employee data | Employment + 7 years | Legal obligations | Secure deletion |

**Automated Deletion:**
```python
class DataRetentionManager:
    def __init__(self):
        self.retention_policies = self.load_retention_policies()

    def schedule_deletion(self):
        """Daily job to delete expired data"""
        # Check each data category
        for policy in self.retention_policies:
            # Find expired data
            expired_data = self.find_expired_data(policy)

            for item in expired_data:
                # Check for legal holds
                if not self.has_legal_hold(item):
                    # Delete data
                    self.delete_data(item)

                    # Log deletion
                    audit_log.log_data_deleted(item, policy.retention_period)

    def delete_data(self, item):
        """Securely delete data"""
        # Delete from primary database
        database.delete(item.id)

        # Delete from backups (mark for deletion)
        backups.mark_for_deletion(item.id)

        # Delete from cloud storage
        cloud_storage.delete(item.path)

        # Notify if required
        if item.notification_required:
            notify_deletion(item.data_subject_id)
```

---

## Third-Party Data Processing

### Article 28: Processor Requirements

**Data Processing Agreements (DPA):**

All third-party processors must sign DPA including:
- Subject matter and duration
- Nature and purpose of processing
- Type of personal data
- Categories of data subjects
- Obligations and rights of controller
- Processor obligations:
  - Process only on documented instructions
  - Ensure confidentiality
  - Implement security measures
  - Engage sub-processors only with authorization
  - Assist with data subject rights
  - Assist with security and breach obligations
  - Delete or return data on termination
  - Make available information for audits

**GreenLang Third-Party Processors:**

| Processor | Purpose | Data Processed | DPA | SCC | Location |
|-----------|---------|----------------|-----|-----|----------|
| AWS | Hosting | All platform data | ✓ | ✓ | EU/US |
| SendGrid | Email | Email addresses | ✓ | ✓ | US |
| Stripe | Payments | Payment info | ✓ | ✓ | US |
| Zendesk | Support | Support tickets | ✓ | ✓ | US |
| Google Analytics | Analytics | Usage data (anonymized) | ✓ | N/A | US |

**Processor Management:**
```python
class ProcessorManager:
    def __init__(self):
        self.processors = ProcessorDatabase()

    def onboard_processor(self, processor):
        """Onboard new data processor"""
        # Due diligence
        due_diligence = self.conduct_due_diligence(processor)

        if not due_diligence['approved']:
            return "Due diligence failed"

        # DPA negotiation
        dpa = self.negotiate_dpa(processor)

        # SCCs if third country
        if self.is_third_country(processor.country):
            scc = self.execute_scc(processor)
        else:
            scc = None

        # Store processor record
        record = {
            'processor': processor,
            'dpa': dpa,
            'scc': scc,
            'onboarded_at': datetime.now(),
            'status': 'active'
        }

        self.processors.insert(record)

        # Annual review scheduled
        self.schedule_annual_review(processor)

        return record

    def conduct_due_diligence(self, processor):
        """Assess processor security and compliance"""
        assessment = {
            'processor': processor.name,
            'assessed_at': datetime.now()
        }

        # Check certifications
        assessment['iso_27001'] = processor.has_certification('ISO 27001')
        assessment['soc2'] = processor.has_certification('SOC 2 Type II')

        # Check security measures
        assessment['encryption'] = self.verify_encryption(processor)
        assessment['access_controls'] = self.verify_access_controls(processor)
        assessment['incident_response'] = self.verify_incident_response(processor)

        # Check subprocessors
        assessment['subprocessors'] = self.review_subprocessors(processor)

        # Approve or reject
        if all([
            assessment['iso_27001'] or assessment['soc2'],
            assessment['encryption'],
            assessment['access_controls']
        ]):
            assessment['approved'] = True
        else:
            assessment['approved'] = False

        return assessment

    def monitor_processor(self, processor_id):
        """Ongoing processor monitoring"""
        processor = self.processors.get(processor_id)

        # Check for security incidents
        incidents = self.check_security_incidents(processor)

        if incidents:
            self.handle_processor_incident(processor, incidents)

        # Check certification status
        certifications = self.check_certifications(processor)

        if not certifications['valid']:
            self.flag_certification_expiry(processor)

        # Annual review
        if self.is_annual_review_due(processor):
            self.conduct_annual_review(processor)
```

---

## Appendices

### Appendix A: GDPR Compliance Checklist

**Accountability:**
- [ ] DPO appointed and contact details published
- [ ] Privacy policy published and up to date
- [ ] Cookie policy published
- [ ] Records of processing activities (RoPA) maintained
- [ ] Data protection policies documented
- [ ] Staff training completed

**Lawful Processing:**
- [ ] Legal basis identified for all processing
- [ ] Consent mechanisms implemented (where applicable)
- [ ] Legitimate interest assessments documented (where applicable)

**Data Subject Rights:**
- [ ] Right of access implemented
- [ ] Right to rectification implemented
- [ ] Right to erasure implemented
- [ ] Right to restriction implemented
- [ ] Right to portability implemented
- [ ] Right to object implemented
- [ ] Rights request process documented
- [ ] Response timeframes monitored (1 month)

**Security:**
- [ ] Encryption at rest implemented
- [ ] Encryption in transit implemented
- [ ] Access controls implemented
- [ ] MFA enforced
- [ ] Security monitoring in place
- [ ] Incident response plan documented
- [ ] Breach notification procedures documented

**Data Protection by Design:**
- [ ] Privacy integrated into product development
- [ ] Data minimization practiced
- [ ] Privacy by default configured
- [ ] DPIAs conducted for high-risk processing

**International Transfers:**
- [ ] Data transfer mechanisms in place (SCCs, adequacy)
- [ ] Transfer impact assessments conducted
- [ ] DPAs with processors include SCCs

**Third Parties:**
- [ ] DPAs signed with all processors
- [ ] Processor due diligence conducted
- [ ] Sub-processor authorization process in place
- [ ] Processor monitoring implemented

### Appendix B: Records of Processing Activities (RoPA)

*Full RoPA document available separately*

### Appendix C: Data Protection Impact Assessment Template

*Full DPIA template available in templates directory*

### Appendix D: Data Subject Rights Request Form

*Form template available in templates directory*

### Appendix E: Consent Records

*Consent management records maintained in compliance database*

---

## Document Control

**Version History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-08 | TEAM 4 | Initial document creation |

**Review Schedule:**
- Next Review: 2026-02-08 (Quarterly)
- Annual Review: 2026-11-08

**Approval:**
- Prepared by: DPO
- Reviewed by: Legal Counsel
- Approved by: CEO

---

**END OF DOCUMENT**
