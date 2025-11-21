# Incident Response Framework

## 1. Incident Response Playbooks

### Master Incident Response Plan

```yaml
# incident-response-plan.yaml
incident_response_framework:
  classification:
    severity_levels:
      P0_critical:
        description: "Complete service outage or data breach"
        response_time: "Immediate"
        escalation: "Automatic to C-level"
        team_size: "Full IR team + executives"
        examples:
          - "Customer data breach"
          - "Ransomware attack"
          - "Complete platform outage"
          - "Authentication system compromise"

      P1_high:
        description: "Major service degradation or security incident"
        response_time: "15 minutes"
        escalation: "Security Lead + Manager"
        team_size: "Core IR team"
        examples:
          - "Partial service outage"
          - "Suspicious privileged access"
          - "DDoS attack"
          - "Critical vulnerability exploitation"

      P2_medium:
        description: "Limited impact security event"
        response_time: "1 hour"
        escalation: "Security Lead"
        team_size: "On-call + specialist"
        examples:
          - "Failed attack attempts"
          - "Malware detection"
          - "Policy violations"
          - "Minor data exposure"

      P3_low:
        description: "Minimal impact security event"
        response_time: "4 hours"
        escalation: "On-call engineer"
        team_size: "Single responder"
        examples:
          - "Spam campaigns"
          - "Low-risk vulnerability"
          - "Failed phishing attempts"

  phases:
    1_detection:
      activities:
        - "Alert triage"
        - "Initial assessment"
        - "Severity classification"
        - "Team activation"

    2_containment:
      activities:
        - "Isolate affected systems"
        - "Preserve evidence"
        - "Stop lateral movement"
        - "Implement temporary fixes"

    3_eradication:
      activities:
        - "Remove malicious artifacts"
        - "Patch vulnerabilities"
        - "Reset compromised credentials"
        - "Clean infected systems"

    4_recovery:
      activities:
        - "Restore services"
        - "Verify system integrity"
        - "Monitor for recurrence"
        - "Implement permanent fixes"

    5_lessons_learned:
      activities:
        - "Post-incident review"
        - "Timeline reconstruction"
        - "Root cause analysis"
        - "Process improvements"

  team_structure:
    incident_commander:
      responsibilities:
        - "Overall incident coordination"
        - "Decision making authority"
        - "External communication"
        - "Resource allocation"

    technical_lead:
      responsibilities:
        - "Technical investigation"
        - "Forensic analysis"
        - "Remediation planning"
        - "Evidence collection"

    communications_lead:
      responsibilities:
        - "Stakeholder updates"
        - "Customer communication"
        - "Media relations"
        - "Internal updates"

    operations_lead:
      responsibilities:
        - "System operations"
        - "Service restoration"
        - "Monitoring coordination"
        - "Infrastructure changes"

    legal_compliance_lead:
      responsibilities:
        - "Legal requirements"
        - "Regulatory notifications"
        - "Evidence preservation"
        - "Law enforcement liaison"
```

### Security Incident Playbooks

```python
# incident_playbooks.py
from typing import Dict, List, Optional
from datetime import datetime
import asyncio

class IncidentPlaybook:
    def __init__(self, incident_type: str):
        self.incident_type = incident_type
        self.steps = []
        self.current_step = 0

    async def execute(self, incident: Dict) -> Dict:
        """Execute playbook for incident"""
        execution_log = []

        for step in self.steps:
            result = await self.execute_step(step, incident)
            execution_log.append({
                "step": step["name"],
                "started": result["started"],
                "completed": result["completed"],
                "status": result["status"],
                "output": result["output"]
            })

            if result["status"] == "failed":
                await self.handle_step_failure(step, result, incident)

        return {
            "playbook": self.incident_type,
            "execution_log": execution_log,
            "status": "completed",
            "recommendations": self.generate_recommendations(execution_log)
        }

class DataBreachPlaybook(IncidentPlaybook):
    def __init__(self):
        super().__init__("data_breach")
        self.steps = [
            {
                "name": "Initial Assessment",
                "actions": [
                    "Identify data types exposed",
                    "Determine number of records affected",
                    "Identify attack vector",
                    "Assess ongoing threat"
                ],
                "tools": ["SIEM", "DLP", "Database Audit Logs"],
                "timeout": 30
            },
            {
                "name": "Containment",
                "actions": [
                    "Isolate affected systems",
                    "Revoke compromised credentials",
                    "Block attacker IP addresses",
                    "Enable enhanced monitoring"
                ],
                "tools": ["Firewall", "IAM", "WAF"],
                "timeout": 15
            },
            {
                "name": "Evidence Collection",
                "actions": [
                    "Capture system images",
                    "Collect network logs",
                    "Preserve database logs",
                    "Document timeline"
                ],
                "tools": ["Forensic Tools", "Log Aggregation"],
                "timeout": 60
            },
            {
                "name": "Impact Analysis",
                "actions": [
                    "Identify affected customers",
                    "Assess regulatory requirements",
                    "Calculate financial impact",
                    "Determine notification requirements"
                ],
                "tools": ["Customer Database", "Compliance Matrix"],
                "timeout": 120
            },
            {
                "name": "Notification",
                "actions": [
                    "Notify executive team",
                    "Prepare customer notifications",
                    "Contact legal counsel",
                    "Notify regulators if required"
                ],
                "tools": ["Communication Platform", "Legal Templates"],
                "timeout": 240
            },
            {
                "name": "Remediation",
                "actions": [
                    "Patch vulnerabilities",
                    "Reset all potentially compromised credentials",
                    "Implement additional security controls",
                    "Update security policies"
                ],
                "tools": ["Patch Management", "IAM", "Security Tools"],
                "timeout": 480
            }
        ]

    async def execute_step(self, step: Dict, incident: Dict) -> Dict:
        """Execute a specific playbook step"""
        result = {
            "started": datetime.now(),
            "status": "in_progress",
            "output": {}
        }

        try:
            for action in step["actions"]:
                if action == "Identify data types exposed":
                    result["output"]["data_types"] = await self.identify_data_types(incident)
                elif action == "Determine number of records affected":
                    result["output"]["record_count"] = await self.count_affected_records(incident)
                elif action == "Isolate affected systems":
                    result["output"]["isolated_systems"] = await self.isolate_systems(incident)
                # ... implement other actions

            result["status"] = "completed"
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)

        result["completed"] = datetime.now()
        return result

class RansomwarePlaybook(IncidentPlaybook):
    def __init__(self):
        super().__init__("ransomware")
        self.steps = [
            {
                "name": "Immediate Containment",
                "priority": "CRITICAL",
                "actions": [
                    "Disconnect affected systems from network",
                    "Disable file sharing protocols",
                    "Isolate backup systems",
                    "Identify patient zero"
                ],
                "automated": True,
                "timeout": 5
            },
            {
                "name": "Assess Encryption Status",
                "actions": [
                    "Identify encrypted files and systems",
                    "Check backup integrity",
                    "Determine ransomware variant",
                    "Search for decryption tools"
                ],
                "timeout": 30
            },
            {
                "name": "Prevent Spread",
                "actions": [
                    "Block C2 communications",
                    "Update endpoint protection",
                    "Reset domain admin credentials",
                    "Implement network segmentation"
                ],
                "timeout": 60
            },
            {
                "name": "Recovery Planning",
                "actions": [
                    "Evaluate recovery options",
                    "Prioritize critical systems",
                    "Prepare clean recovery environment",
                    "Plan restoration sequence"
                ],
                "timeout": 120
            },
            {
                "name": "System Recovery",
                "actions": [
                    "Restore from clean backups",
                    "Rebuild compromised systems",
                    "Verify system integrity",
                    "Restore data and services"
                ],
                "timeout": 1440
            }
        ]

class DDoSPlaybook(IncidentPlaybook):
    def __init__(self):
        super().__init__("ddos_attack")
        self.steps = [
            {
                "name": "Attack Identification",
                "actions": [
                    "Identify attack type and vector",
                    "Determine attack volume",
                    "Identify targeted services",
                    "Assess impact on availability"
                ],
                "timeout": 5
            },
            {
                "name": "Traffic Mitigation",
                "actions": [
                    "Enable DDoS protection",
                    "Implement rate limiting",
                    "Configure geo-blocking if appropriate",
                    "Activate CDN caching"
                ],
                "automated": True,
                "timeout": 10
            },
            {
                "name": "Traffic Analysis",
                "actions": [
                    "Analyze attack patterns",
                    "Identify source IPs/networks",
                    "Detect amplification vectors",
                    "Look for application layer attacks"
                ],
                "timeout": 30
            },
            {
                "name": "Enhanced Protection",
                "actions": [
                    "Tune WAF rules",
                    "Implement CAPTCHA challenges",
                    "Configure advanced scrubbing",
                    "Scale infrastructure if needed"
                ],
                "timeout": 60
            }
        ]

class InsiderThreatPlaybook(IncidentPlaybook):
    def __init__(self):
        super().__init__("insider_threat")
        self.steps = [
            {
                "name": "User Activity Analysis",
                "actions": [
                    "Review user access logs",
                    "Analyze data access patterns",
                    "Check for policy violations",
                    "Identify abnormal behavior"
                ],
                "timeout": 60
            },
            {
                "name": "Access Suspension",
                "actions": [
                    "Suspend user accounts",
                    "Revoke access tokens",
                    "Disable VPN access",
                    "Secure physical access"
                ],
                "requires_approval": True,
                "timeout": 15
            },
            {
                "name": "Forensic Investigation",
                "actions": [
                    "Preserve user workstation",
                    "Collect email communications",
                    "Review file access history",
                    "Interview witnesses"
                ],
                "timeout": 480
            },
            {
                "name": "Data Loss Assessment",
                "actions": [
                    "Identify accessed data",
                    "Check for data exfiltration",
                    "Review external communications",
                    "Assess competitive impact"
                ],
                "timeout": 240
            }
        ]

class PlaybookOrchestrator:
    def __init__(self):
        self.playbooks = {
            "data_breach": DataBreachPlaybook(),
            "ransomware": RansomwarePlaybook(),
            "ddos": DDoSPlaybook(),
            "insider_threat": InsiderThreatPlaybook()
        }

    async def select_playbook(self, incident: Dict) -> IncidentPlaybook:
        """Select appropriate playbook based on incident type"""
        incident_indicators = self.analyze_incident(incident)

        if "data_exposure" in incident_indicators:
            return self.playbooks["data_breach"]
        elif "encryption" in incident_indicators or "ransom" in incident_indicators:
            return self.playbooks["ransomware"]
        elif "availability" in incident_indicators or "traffic_spike" in incident_indicators:
            return self.playbooks["ddos"]
        elif "insider" in incident_indicators or "privileged_abuse" in incident_indicators:
            return self.playbooks["insider_threat"]
        else:
            return self.create_generic_playbook(incident)

    async def execute_playbook(self, incident: Dict) -> Dict:
        """Execute selected playbook for incident"""
        playbook = await self.select_playbook(incident)
        return await playbook.execute(incident)
```

## 2. Escalation Procedures

### Escalation Matrix

```yaml
# escalation-matrix.yaml
escalation_procedures:
  escalation_triggers:
    automatic_escalation:
      - "P0 incident detected"
      - "Customer data breach confirmed"
      - "Ransomware detected"
      - "Critical system compromise"
      - "Regulatory violation"
      - "Media attention"

    time_based_escalation:
      - condition: "P1 unresolved for 2 hours"
        escalate_to: "Director of Security"
      - condition: "P2 unresolved for 8 hours"
        escalate_to: "Security Manager"
      - condition: "Any incident unresolved for 24 hours"
        escalate_to: "CISO"

    impact_based_escalation:
      - condition: "Affects > 100 customers"
        escalate_to: "VP Engineering"
      - condition: "Affects > 1000 customers"
        escalate_to: "C-Suite"
      - condition: "Revenue impact > $100K"
        escalate_to: "CFO"

  escalation_chain:
    level_1:
      role: "On-Call Security Engineer"
      response_time: "5 minutes"
      authority:
        - "Investigate incidents"
        - "Implement containment"
        - "Engage specialists"

    level_2:
      role: "Security Team Lead"
      response_time: "15 minutes"
      authority:
        - "All Level 1 authorities"
        - "Approve system isolation"
        - "Engage external resources"
        - "Initiate customer communication"

    level_3:
      role: "Director of Security"
      response_time: "30 minutes"
      authority:
        - "All Level 2 authorities"
        - "Approve service degradation"
        - "Authorize emergency changes"
        - "Engage law enforcement"

    level_4:
      role: "CISO"
      response_time: "1 hour"
      authority:
        - "All Level 3 authorities"
        - "Approve major service impacts"
        - "Authorize public statements"
        - "Engage board of directors"

    level_5:
      role: "CEO"
      response_time: "2 hours"
      authority:
        - "All authorities"
        - "Crisis management decisions"
        - "Legal action authorization"
        - "Media spokesperson"

  contact_lists:
    on_call_rotation:
      primary:
        - name: "Security Engineer 1"
          phone: "+1-XXX-XXX-XXXX"
          email: "oncall1@greenlang.io"
      secondary:
        - name: "Security Engineer 2"
          phone: "+1-XXX-XXX-XXXX"
          email: "oncall2@greenlang.io"

    management_chain:
      security_lead:
        name: "Security Team Lead"
        phone: "+1-XXX-XXX-XXXX"
        email: "security-lead@greenlang.io"
        backup: "security-lead-backup@greenlang.io"

      director:
        name: "Director of Security"
        phone: "+1-XXX-XXX-XXXX"
        email: "security-director@greenlang.io"

      ciso:
        name: "Chief Information Security Officer"
        phone: "+1-XXX-XXX-XXXX"
        email: "ciso@greenlang.io"

    external_contacts:
      legal_counsel:
        firm: "Legal Firm Name"
        contact: "Attorney Name"
        phone: "+1-XXX-XXX-XXXX"
        email: "legal@lawfirm.com"

      incident_response_firm:
        company: "IR Firm Name"
        hotline: "+1-800-XXX-XXXX"
        email: "incident@irfirm.com"

      law_enforcement:
        fbi_cyber: "+1-XXX-XXX-XXXX"
        local_police: "911"
        interpol: "+XX-XXX-XXX-XXXX"

      regulatory_bodies:
        gdpr_authority: "dpo@dataprotection.eu"
        hipaa_ocr: "ocr@hhs.gov"
        pci_council: "security@pcisecuritystandards.org"
```

### Escalation Automation

```python
# escalation_automation.py
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio

class EscalationManager:
    def __init__(self, notification_service, contact_manager):
        self.notifications = notification_service
        self.contacts = contact_manager
        self.active_escalations = {}

    async def evaluate_escalation(self, incident: Dict) -> bool:
        """Evaluate if incident requires escalation"""
        escalation_needed = False
        reasons = []

        # Check automatic triggers
        if self.check_automatic_triggers(incident):
            escalation_needed = True
            reasons.append("Automatic trigger met")

        # Check time-based triggers
        if self.check_time_triggers(incident):
            escalation_needed = True
            reasons.append("Time threshold exceeded")

        # Check impact triggers
        if self.check_impact_triggers(incident):
            escalation_needed = True
            reasons.append("Impact threshold exceeded")

        if escalation_needed:
            await self.initiate_escalation(incident, reasons)

        return escalation_needed

    async def initiate_escalation(self, incident: Dict, reasons: List[str]):
        """Initiate escalation process"""
        escalation = {
            "incident_id": incident["id"],
            "started_at": datetime.now(),
            "reasons": reasons,
            "current_level": self.determine_escalation_level(incident),
            "notifications_sent": [],
            "acknowledgments": []
        }

        # Get escalation contacts
        contacts = self.get_escalation_contacts(escalation["current_level"])

        # Send notifications
        for contact in contacts:
            notification_result = await self.send_escalation_notification(
                contact,
                incident,
                escalation
            )
            escalation["notifications_sent"].append(notification_result)

        # Start acknowledgment tracking
        asyncio.create_task(self.track_acknowledgment(escalation))

        self.active_escalations[incident["id"]] = escalation

        return escalation

    async def send_escalation_notification(
        self,
        contact: Dict,
        incident: Dict,
        escalation: Dict
    ) -> Dict:
        """Send escalation notification to contact"""
        message = self.format_escalation_message(incident, escalation)

        notification_channels = []

        # Send via multiple channels for critical incidents
        if incident["severity"] == "P0":
            # Phone call
            phone_result = await self.notifications.send_phone_call(
                contact["phone"],
                message["voice_message"]
            )
            notification_channels.append({
                "channel": "phone",
                "status": phone_result["status"]
            })

            # SMS
            sms_result = await self.notifications.send_sms(
                contact["phone"],
                message["sms_message"]
            )
            notification_channels.append({
                "channel": "sms",
                "status": sms_result["status"]
            })

        # Email (always send)
        email_result = await self.notifications.send_email(
            contact["email"],
            message["subject"],
            message["email_body"]
        )
        notification_channels.append({
            "channel": "email",
            "status": email_result["status"]
        })

        # Slack/Teams
        if contact.get("slack_id"):
            slack_result = await self.notifications.send_slack(
                contact["slack_id"],
                message["slack_message"]
            )
            notification_channels.append({
                "channel": "slack",
                "status": slack_result["status"]
            })

        return {
            "contact": contact["name"],
            "timestamp": datetime.now(),
            "channels": notification_channels
        }

    async def track_acknowledgment(self, escalation: Dict):
        """Track acknowledgment of escalation"""
        acknowledgment_deadline = datetime.now() + timedelta(
            minutes=self.get_acknowledgment_deadline(escalation["current_level"])
        )

        while datetime.now() < acknowledgment_deadline:
            # Check for acknowledgment
            if await self.check_acknowledgment(escalation):
                escalation["acknowledged_at"] = datetime.now()
                escalation["acknowledged_by"] = self.get_acknowledger(escalation)
                return

            await asyncio.sleep(30)  # Check every 30 seconds

        # No acknowledgment received - escalate further
        await self.escalate_to_next_level(escalation)

    async def escalate_to_next_level(self, escalation: Dict):
        """Escalate to next level in chain"""
        current_level = escalation["current_level"]
        next_level = self.get_next_level(current_level)

        if next_level:
            escalation["current_level"] = next_level
            escalation["escalated_at"] = datetime.now()

            # Get next level contacts
            contacts = self.get_escalation_contacts(next_level)

            # Send notifications to next level
            for contact in contacts:
                await self.send_escalation_notification(
                    contact,
                    self.get_incident(escalation["incident_id"]),
                    escalation
                )

            # Continue tracking
            asyncio.create_task(self.track_acknowledgment(escalation))
        else:
            # Reached top of escalation chain
            await self.handle_max_escalation(escalation)

    def format_escalation_message(self, incident: Dict, escalation: Dict) -> Dict:
        """Format escalation notification message"""
        base_info = f"""
URGENT: Security Incident Escalation Required

Incident ID: {incident['id']}
Severity: {incident['severity']}
Type: {incident['type']}
Started: {incident['detected_at']}
Current Status: {incident['status']}

Description: {incident['description']}

Impact: {incident.get('impact', 'Under assessment')}

Escalation Reasons: {', '.join(escalation['reasons'])}

Required Action: Please acknowledge and join incident response.
        """

        return {
            "subject": f"[{incident['severity']}] Security Incident {incident['id']} - Escalation Required",
            "email_body": base_info + "\n\nIncident Link: https://incidents.greenlang.io/" + incident['id'],
            "sms_message": f"URGENT: {incident['severity']} incident {incident['id']} requires your attention. Check email for details.",
            "voice_message": f"This is an urgent security escalation for a {incident['severity']} incident. Please check your email immediately.",
            "slack_message": {
                "text": f"🚨 Security Incident Escalation",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{incident['severity']} Incident Escalation"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*ID:* {incident['id']}"},
                            {"type": "mrkdwn", "text": f"*Type:* {incident['type']}"},
                            {"type": "mrkdwn", "text": f"*Status:* {incident['status']}"},
                            {"type": "mrkdwn", "text": f"*Impact:* {incident.get('impact', 'TBD')}"}
                        ]
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "Acknowledge"},
                                "action_id": "acknowledge_escalation",
                                "value": escalation["incident_id"]
                            },
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "Join Response"},
                                "url": f"https://incidents.greenlang.io/{incident['id']}"
                            }
                        ]
                    }
                ]
            }
        }
```

## 3. Communication Templates

### Incident Communication Templates

```yaml
# communication-templates.yaml
communication_templates:
  internal:
    initial_notification:
      subject: "[SEVERITY] Security Incident Detected - [INCIDENT_ID]"
      body: |
        Team,

        We have detected a security incident that requires immediate attention.

        Incident Details:
        - ID: [INCIDENT_ID]
        - Severity: [SEVERITY]
        - Type: [INCIDENT_TYPE]
        - Detection Time: [DETECTION_TIME]
        - Status: [CURRENT_STATUS]

        Initial Assessment:
        [INITIAL_ASSESSMENT]

        Immediate Actions Taken:
        [ACTIONS_TAKEN]

        Next Steps:
        [NEXT_STEPS]

        Incident Commander: [COMMANDER_NAME]
        Incident Channel: [COMMUNICATION_CHANNEL]

        Please join the incident response if you are on-call or have relevant expertise.

    status_update:
      subject: "Update: [INCIDENT_ID] - [CURRENT_STATUS]"
      body: |
        Incident Update

        Current Status: [CURRENT_STATUS]
        Time Since Detection: [ELAPSED_TIME]

        Progress Summary:
        [PROGRESS_SUMMARY]

        Recent Actions:
        [RECENT_ACTIONS]

        Upcoming Actions:
        [PLANNED_ACTIONS]

        Estimated Resolution: [ETA]

    resolution_notification:
      subject: "Resolved: [INCIDENT_ID] - Incident Closed"
      body: |
        The security incident [INCIDENT_ID] has been resolved.

        Resolution Summary:
        [RESOLUTION_SUMMARY]

        Root Cause:
        [ROOT_CAUSE]

        Remediation Actions:
        [REMEDIATION_ACTIONS]

        Lessons Learned:
        [LESSONS_LEARNED]

        Post-Incident Review scheduled for: [PIR_DATE]

  customer:
    breach_notification:
      subject: "Important Security Update from GreenLang"
      body: |
        Dear [CUSTOMER_NAME],

        We are writing to inform you of a security incident that may have affected your account.

        What Happened:
        [INCIDENT_DESCRIPTION]

        Information Involved:
        [AFFECTED_DATA]

        When It Happened:
        [TIMELINE]

        What We Are Doing:
        [OUR_ACTIONS]

        What You Should Do:
        [CUSTOMER_ACTIONS]

        For More Information:
        [CONTACT_INFORMATION]

        We sincerely apologize for any inconvenience and are committed to protecting your information.

        Sincerely,
        [EXECUTIVE_NAME]
        [TITLE]

    service_disruption:
      subject: "Service Disruption Notice - [SERVICE_NAME]"
      body: |
        Dear Customer,

        We are currently experiencing a service disruption affecting [SERVICE_NAME].

        Impact:
        [IMPACT_DESCRIPTION]

        Current Status:
        [CURRENT_STATUS]

        Estimated Resolution:
        [ETA]

        Workaround (if available):
        [WORKAROUND]

        We will provide updates every [UPDATE_FREQUENCY] at [UPDATE_CHANNEL].

        We apologize for the inconvenience.

    all_clear:
      subject: "Service Restored - [SERVICE_NAME]"
      body: |
        Dear Customer,

        We are pleased to inform you that the issue affecting [SERVICE_NAME] has been resolved.

        Service Restoration Time: [RESTORATION_TIME]
        Total Duration: [TOTAL_DURATION]

        Summary:
        [INCIDENT_SUMMARY]

        Prevention Measures:
        [PREVENTION_MEASURES]

        If you continue to experience issues, please contact support at [SUPPORT_CONTACT].

        Thank you for your patience.

  regulatory:
    gdpr_breach_notification:
      recipient: "Data Protection Authority"
      timeline: "Within 72 hours"
      content: |
        Data Breach Notification under Article 33 GDPR

        1. Controller Details:
           Organization: GreenLang Inc.
           DPO Contact: [DPO_CONTACT]

        2. Breach Description:
           Date and Time: [BREACH_DATETIME]
           Nature of Breach: [BREACH_TYPE]
           Categories of Data: [DATA_CATEGORIES]
           Approximate Number of Affected Individuals: [AFFECTED_COUNT]

        3. Likely Consequences:
           [CONSEQUENCE_ASSESSMENT]

        4. Measures Taken:
           [MITIGATION_MEASURES]

        5. Additional Information:
           [ADDITIONAL_INFO]

        Contact for Further Information:
        [CONTACT_DETAILS]

    hipaa_breach_notification:
      recipient: "HHS Office for Civil Rights"
      timeline: "Within 60 days"
      content: |
        HIPAA Breach Notification

        Covered Entity: GreenLang Healthcare Services
        Date of Breach: [BREACH_DATE]
        Date of Discovery: [DISCOVERY_DATE]

        Description of Breach:
        [BREACH_DESCRIPTION]

        Types of PHI Involved:
        [PHI_TYPES]

        Number of Individuals Affected:
        [AFFECTED_COUNT]

        Mitigation Actions:
        [MITIGATION_ACTIONS]

        Individual Notification:
        [NOTIFICATION_STATUS]

        Media Notification (if required):
        [MEDIA_NOTIFICATION_STATUS]

  media:
    press_release:
      embargo_until: "[EMBARGO_TIME]"
      content: |
        FOR IMMEDIATE RELEASE

        GreenLang Addresses Recent Security Incident

        [CITY, Date] - GreenLang today announced [BRIEF_DESCRIPTION].

        "Quote from CEO," said [CEO_NAME], CEO of GreenLang.

        Key Facts:
        - [FACT_1]
        - [FACT_2]
        - [FACT_3]

        Customer Impact:
        [CUSTOMER_IMPACT]

        Company Response:
        [COMPANY_RESPONSE]

        About GreenLang:
        [COMPANY_BOILERPLATE]

        Contact:
        [PR_CONTACT]
```

### Communication Automation

```python
# communication_automation.py
from typing import Dict, List, Optional
from datetime import datetime
import jinja2

class CommunicationManager:
    def __init__(self, template_engine, distribution_service):
        self.templates = template_engine
        self.distribution = distribution_service
        self.communication_log = []

    async def send_incident_communication(
        self,
        incident: Dict,
        communication_type: str,
        audience: str
    ) -> Dict:
        """Send incident communication to specified audience"""
        # Select template
        template = self.select_template(communication_type, audience)

        # Prepare data
        template_data = self.prepare_template_data(incident)

        # Render message
        message = self.render_template(template, template_data)

        # Get distribution list
        recipients = self.get_recipients(audience, incident)

        # Send communication
        result = await self.distribute_communication(
            message,
            recipients,
            incident["severity"]
        )

        # Log communication
        self.log_communication(incident, communication_type, audience, result)

        return result

    def select_template(self, communication_type: str, audience: str) -> str:
        """Select appropriate template"""
        template_map = {
            ("initial", "internal"): "internal.initial_notification",
            ("update", "internal"): "internal.status_update",
            ("resolved", "internal"): "internal.resolution_notification",
            ("breach", "customer"): "customer.breach_notification",
            ("disruption", "customer"): "customer.service_disruption",
            ("restored", "customer"): "customer.all_clear",
            ("breach", "regulatory"): "regulatory.gdpr_breach_notification",
            ("breach", "media"): "media.press_release"
        }

        return template_map.get((communication_type, audience))

    def prepare_template_data(self, incident: Dict) -> Dict:
        """Prepare data for template rendering"""
        return {
            "INCIDENT_ID": incident["id"],
            "SEVERITY": incident["severity"],
            "INCIDENT_TYPE": incident["type"],
            "DETECTION_TIME": incident["detected_at"],
            "CURRENT_STATUS": incident["status"],
            "INITIAL_ASSESSMENT": incident.get("assessment", "Under investigation"),
            "ACTIONS_TAKEN": self.format_actions(incident.get("actions", [])),
            "NEXT_STEPS": self.format_next_steps(incident.get("next_steps", [])),
            "COMMANDER_NAME": incident.get("commander", "On-call Lead"),
            "COMMUNICATION_CHANNEL": self.get_incident_channel(incident),
            "ELAPSED_TIME": self.calculate_elapsed_time(incident),
            "ETA": incident.get("eta", "To be determined"),
            "AFFECTED_DATA": self.describe_affected_data(incident),
            "CUSTOMER_ACTIONS": self.recommend_customer_actions(incident)
        }

    async def distribute_communication(
        self,
        message: Dict,
        recipients: List[str],
        severity: str
    ) -> Dict:
        """Distribute communication to recipients"""
        distribution_results = {
            "total_recipients": len(recipients),
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "delivery_details": []
        }

        # Determine delivery method based on severity
        if severity in ["P0", "P1"]:
            # Multi-channel delivery for critical incidents
            for recipient in recipients:
                email_result = await self.distribution.send_email(
                    recipient,
                    message["subject"],
                    message["body"]
                )

                if self.has_phone(recipient):
                    sms_result = await self.distribution.send_sms(
                        recipient,
                        message["sms_summary"]
                    )

                distribution_results["delivery_details"].append({
                    "recipient": recipient,
                    "channels": ["email", "sms"],
                    "status": "delivered"
                })
                distribution_results["successful_deliveries"] += 1
        else:
            # Email only for lower severity
            for recipient in recipients:
                result = await self.distribution.send_email(
                    recipient,
                    message["subject"],
                    message["body"]
                )

                if result["status"] == "success":
                    distribution_results["successful_deliveries"] += 1
                else:
                    distribution_results["failed_deliveries"] += 1

                distribution_results["delivery_details"].append(result)

        return distribution_results

    def get_recipients(self, audience: str, incident: Dict) -> List[str]:
        """Get recipient list for communication"""
        recipient_lists = {
            "internal": self.get_internal_recipients(incident),
            "customer": self.get_affected_customers(incident),
            "regulatory": self.get_regulatory_contacts(incident),
            "media": self.get_media_contacts(incident),
            "executive": self.get_executive_contacts(incident)
        }

        return recipient_lists.get(audience, [])

    def should_notify_customers(self, incident: Dict) -> bool:
        """Determine if customer notification is required"""
        notification_triggers = [
            incident.get("customer_impact", False),
            incident.get("data_breach", False),
            incident.get("service_disruption", False),
            incident["severity"] in ["P0", "P1"],
            self.is_regulatory_requirement(incident)
        ]

        return any(notification_triggers)

    def get_notification_timeline(self, incident: Dict) -> Dict:
        """Get required notification timeline"""
        timelines = {}

        # GDPR requirement
        if incident.get("gdpr_applicable"):
            timelines["gdpr"] = {
                "authority": "72 hours",
                "individuals": "without undue delay"
            }

        # HIPAA requirement
        if incident.get("phi_involved"):
            timelines["hipaa"] = {
                "individuals": "60 days",
                "hhs": "60 days",
                "media": "60 days if >500 affected"
            }

        # State laws (e.g., California)
        if incident.get("california_residents"):
            timelines["ccpa"] = {
                "individuals": "without unreasonable delay",
                "attorney_general": "if >500 residents"
            }

        return timelines
```

## 4. Post-Incident Reviews

### Post-Incident Review Process

```yaml
# post-incident-review.yaml
post_incident_review:
  review_triggers:
    mandatory_review:
      - "All P0 incidents"
      - "All P1 incidents"
      - "Customer data breaches"
      - "Service outages > 1 hour"
      - "Security breaches"
      - "Regulatory violations"

    optional_review:
      - "P2 incidents with lessons learned"
      - "Near-miss incidents"
      - "Successful attack prevention"
      - "Process failures"

  review_timeline:
    scheduling:
      p0_incidents: "Within 48 hours of resolution"
      p1_incidents: "Within 1 week of resolution"
      p2_incidents: "Within 2 weeks of resolution"

    preparation:
      - "Incident timeline reconstruction"
      - "Evidence collection"
      - "Stakeholder identification"
      - "Data analysis"
      - "Draft report preparation"

  review_participants:
    required:
      - "Incident Commander"
      - "Technical Lead"
      - "Service Owner"
      - "Security Team Representative"

    optional:
      - "Customer Success (if customer impact)"
      - "Legal (if regulatory implications)"
      - "Executive sponsor (if P0)"
      - "External parties (if involved)"

  review_agenda:
    1_incident_overview:
      duration: "10 minutes"
      content:
        - "Incident summary"
        - "Timeline of events"
        - "Impact assessment"
        - "Resolution summary"

    2_what_happened:
      duration: "20 minutes"
      content:
        - "Detection and alerting"
        - "Initial response"
        - "Investigation process"
        - "Root cause identification"

    3_what_went_well:
      duration: "15 minutes"
      content:
        - "Effective responses"
        - "Good decisions"
        - "Helpful tools/processes"
        - "Team collaboration"

    4_what_could_improve:
      duration: "20 minutes"
      content:
        - "Detection gaps"
        - "Response delays"
        - "Communication issues"
        - "Process failures"
        - "Tool limitations"

    5_action_items:
      duration: "20 minutes"
      content:
        - "Immediate fixes"
        - "Short-term improvements"
        - "Long-term changes"
        - "Process updates"
        - "Training needs"

    6_metrics_review:
      duration: "10 minutes"
      content:
        - "MTTD analysis"
        - "MTTR analysis"
        - "SLA compliance"
        - "Communication effectiveness"

  documentation:
    report_sections:
      - executive_summary
      - incident_timeline
      - root_cause_analysis
      - impact_assessment
      - response_evaluation
      - lessons_learned
      - action_items
      - prevention_measures

    distribution:
      - security_team
      - engineering_leadership
      - affected_teams
      - executive_briefing
      - knowledge_base
```

### Post-Incident Review Automation

```python
# post_incident_review.py
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class PostIncidentReview:
    def __init__(self, incident_data, analytics_service):
        self.incident = incident_data
        self.analytics = analytics_service
        self.review_data = {}

    async def conduct_review(self) -> Dict:
        """Conduct comprehensive post-incident review"""
        review = {
            "incident_id": self.incident["id"],
            "review_date": datetime.now(),
            "participants": self.get_participants(),
            "timeline": await self.reconstruct_timeline(),
            "root_cause": await self.perform_root_cause_analysis(),
            "impact_analysis": await self.analyze_impact(),
            "response_evaluation": await self.evaluate_response(),
            "lessons_learned": await self.identify_lessons(),
            "action_items": await self.generate_action_items(),
            "prevention_measures": await self.recommend_prevention(),
            "metrics": await self.calculate_metrics()
        }

        return review

    async def reconstruct_timeline(self) -> List[Dict]:
        """Reconstruct detailed incident timeline"""
        timeline = []

        # Get all events related to incident
        events = await self.get_incident_events()

        for event in events:
            timeline_entry = {
                "timestamp": event["timestamp"],
                "event": event["description"],
                "actor": event.get("actor", "System"),
                "action": event["action"],
                "result": event.get("result"),
                "duration": self.calculate_duration(event),
                "critical_path": self.is_critical_path(event)
            }
            timeline.append(timeline_entry)

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        # Add phase markers
        timeline = self.add_phase_markers(timeline)

        return timeline

    async def perform_root_cause_analysis(self) -> Dict:
        """Perform root cause analysis using 5 Whys and Fishbone"""
        rca = {
            "method": "5 Whys + Fishbone Diagram",
            "direct_cause": self.incident.get("direct_cause"),
            "contributing_factors": [],
            "root_causes": [],
            "analysis": {}
        }

        # 5 Whys Analysis
        whys = []
        current_why = "Why did the incident occur?"
        current_answer = self.incident.get("direct_cause")

        for i in range(5):
            whys.append({
                "question": current_why,
                "answer": current_answer
            })

            next_why = f"Why did {current_answer}?"
            next_answer = await self.analyze_cause(current_answer)

            if not next_answer or next_answer == current_answer:
                break

            current_why = next_why
            current_answer = next_answer

        rca["analysis"]["five_whys"] = whys
        rca["root_causes"].append(current_answer)

        # Fishbone Analysis
        fishbone = {
            "people": await self.analyze_people_factors(),
            "process": await self.analyze_process_factors(),
            "technology": await self.analyze_technology_factors(),
            "environment": await self.analyze_environment_factors()
        }

        rca["analysis"]["fishbone"] = fishbone

        # Identify contributing factors
        for category, factors in fishbone.items():
            rca["contributing_factors"].extend(factors)

        return rca

    async def evaluate_response(self) -> Dict:
        """Evaluate incident response effectiveness"""
        evaluation = {
            "detection": {
                "method": self.incident.get("detection_method"),
                "time_to_detect": self.calculate_detection_time(),
                "effectiveness": self.rate_detection_effectiveness(),
                "gaps": self.identify_detection_gaps()
            },
            "response": {
                "initial_response_time": self.calculate_initial_response_time(),
                "escalation_effectiveness": self.evaluate_escalation(),
                "communication": self.evaluate_communication(),
                "decision_making": self.evaluate_decisions()
            },
            "containment": {
                "time_to_contain": self.calculate_containment_time(),
                "effectiveness": self.rate_containment_effectiveness(),
                "collateral_damage": self.assess_collateral_damage()
            },
            "recovery": {
                "time_to_recover": self.calculate_recovery_time(),
                "completeness": self.assess_recovery_completeness(),
                "verification": self.evaluate_verification()
            }
        }

        # Calculate overall score
        evaluation["overall_score"] = self.calculate_response_score(evaluation)

        return evaluation

    async def identify_lessons(self) -> Dict:
        """Identify lessons learned from incident"""
        lessons = {
            "what_went_well": [],
            "what_went_wrong": [],
            "lucky_breaks": [],
            "improvement_opportunities": []
        }

        # Analyze what went well
        if self.incident.get("detection_time") < 300:  # <5 minutes
            lessons["what_went_well"].append({
                "area": "Detection",
                "description": "Rapid detection through monitoring",
                "preserve": "Maintain current alerting thresholds"
            })

        # Analyze what went wrong
        if self.incident.get("customer_impact"):
            lessons["what_went_wrong"].append({
                "area": "Customer Impact",
                "description": "Customers experienced service disruption",
                "fix": "Implement better circuit breakers"
            })

        # Identify lucky breaks
        if self.incident.get("contained_by_chance"):
            lessons["lucky_breaks"].append({
                "description": "Rate limiting prevented wider impact",
                "action": "Ensure this is intentional, not luck"
            })

        # Improvement opportunities
        improvements = await self.identify_improvements()
        lessons["improvement_opportunities"] = improvements

        return lessons

    async def generate_action_items(self) -> List[Dict]:
        """Generate prioritized action items"""
        action_items = []

        # Immediate fixes
        if self.incident.get("vulnerability_exploited"):
            action_items.append({
                "priority": "P0",
                "category": "Immediate Fix",
                "title": "Patch exploited vulnerability",
                "description": f"Apply security patch for {self.incident['vulnerability']}",
                "owner": "Security Team",
                "due_date": datetime.now() + timedelta(days=1),
                "status": "pending"
            })

        # Process improvements
        if self.incident.get("process_failure"):
            action_items.append({
                "priority": "P1",
                "category": "Process",
                "title": "Update incident response process",
                "description": "Address gaps in current IR process",
                "owner": "Security Lead",
                "due_date": datetime.now() + timedelta(days=7),
                "status": "pending"
            })

        # Tool improvements
        if self.incident.get("tool_gap"):
            action_items.append({
                "priority": "P2",
                "category": "Tooling",
                "title": "Implement additional monitoring",
                "description": "Deploy monitoring for identified blind spots",
                "owner": "DevOps Team",
                "due_date": datetime.now() + timedelta(days=30),
                "status": "pending"
            })

        # Training needs
        if self.incident.get("knowledge_gap"):
            action_items.append({
                "priority": "P2",
                "category": "Training",
                "title": "Conduct security training",
                "description": "Train team on identified knowledge gaps",
                "owner": "Training Team",
                "due_date": datetime.now() + timedelta(days=14),
                "status": "pending"
            })

        return sorted(action_items, key=lambda x: x["priority"])

    async def recommend_prevention(self) -> Dict:
        """Recommend prevention measures"""
        prevention = {
            "technical_controls": [],
            "process_improvements": [],
            "training_requirements": [],
            "tool_enhancements": []
        }

        # Technical controls
        if self.incident["type"] == "data_breach":
            prevention["technical_controls"].append({
                "control": "Data Loss Prevention",
                "implementation": "Deploy DLP on all egress points",
                "effectiveness": "High",
                "cost": "Medium"
            })

        # Process improvements
        prevention["process_improvements"].append({
            "process": "Change Management",
            "improvement": "Add security review gate",
            "impact": "Prevent configuration errors"
        })

        # Training requirements
        prevention["training_requirements"].append({
            "audience": "Development Team",
            "topic": "Secure Coding Practices",
            "frequency": "Quarterly"
        })

        # Tool enhancements
        prevention["tool_enhancements"].append({
            "tool": "SIEM",
            "enhancement": "Add correlation rules for attack pattern",
            "priority": "High"
        })

        return prevention

    async def calculate_metrics(self) -> Dict:
        """Calculate incident metrics"""
        return {
            "mttd": self.calculate_mttd(),
            "mttr": self.calculate_mttr(),
            "mtta": self.calculate_mtta(),
            "mttc": self.calculate_mttc(),
            "total_downtime": self.calculate_downtime(),
            "affected_users": self.count_affected_users(),
            "data_exposed": self.calculate_data_exposure(),
            "financial_impact": self.estimate_financial_impact(),
            "reputation_impact": self.assess_reputation_impact(),
            "compliance_impact": self.assess_compliance_impact()
        }

    def generate_report(self, review_data: Dict) -> str:
        """Generate PIR report document"""
        report = f"""
# Post-Incident Review Report

**Incident ID:** {review_data['incident_id']}
**Review Date:** {review_data['review_date']}

## Executive Summary
{self.generate_executive_summary(review_data)}

## Incident Timeline
{self.format_timeline(review_data['timeline'])}

## Root Cause Analysis
{self.format_rca(review_data['root_cause'])}

## Impact Assessment
{self.format_impact(review_data['impact_analysis'])}

## Response Evaluation
{self.format_response_evaluation(review_data['response_evaluation'])}

## Lessons Learned
{self.format_lessons(review_data['lessons_learned'])}

## Action Items
{self.format_action_items(review_data['action_items'])}

## Prevention Measures
{self.format_prevention(review_data['prevention_measures'])}

## Metrics
{self.format_metrics(review_data['metrics'])}

## Appendices
- Full incident logs
- Communication records
- Technical analysis
        """

        return report
```

## 5. Tabletop Exercise Schedule

### Exercise Planning Framework

```yaml
# tabletop-exercises.yaml
tabletop_exercise_program:
  annual_schedule:
    q1_exercise:
      month: "February"
      scenario: "Ransomware Attack"
      participants:
        - "Security Team"
        - "IT Operations"
        - "Executive Team"
        - "Legal"
        - "PR/Communications"
      duration: "4 hours"
      objectives:
        - "Test containment procedures"
        - "Evaluate backup recovery"
        - "Practice crisis communication"
        - "Test escalation procedures"

    q2_exercise:
      month: "May"
      scenario: "Data Breach"
      participants:
        - "Security Team"
        - "Privacy Team"
        - "Legal"
        - "Customer Success"
        - "Executive Team"
      duration: "4 hours"
      objectives:
        - "Test breach notification process"
        - "Practice regulatory reporting"
        - "Evaluate customer communication"
        - "Test evidence preservation"

    q3_exercise:
      month: "August"
      scenario: "Supply Chain Attack"
      participants:
        - "Security Team"
        - "Vendor Management"
        - "DevOps"
        - "Product Team"
      duration: "3 hours"
      objectives:
        - "Test third-party incident response"
        - "Evaluate dependency management"
        - "Practice vendor communication"
        - "Test isolation procedures"

    q4_exercise:
      month: "November"
      scenario: "Insider Threat"
      participants:
        - "Security Team"
        - "HR"
        - "Legal"
        - "IT Operations"
      duration: "3 hours"
      objectives:
        - "Test insider threat detection"
        - "Practice evidence collection"
        - "Evaluate access revocation"
        - "Test legal procedures"

  exercise_types:
    discussion_based:
      tabletop:
        frequency: "Quarterly"
        duration: "2-4 hours"
        format: "Facilitated discussion"
        deliverables:
          - "Exercise report"
          - "Lessons learned"
          - "Action items"

    operations_based:
      functional:
        frequency: "Semi-annually"
        duration: "4-8 hours"
        format: "Simulated operations"
        deliverables:
          - "Performance metrics"
          - "Gap analysis"
          - "Improvement plan"

      full_scale:
        frequency: "Annually"
        duration: "1-2 days"
        format: "Live simulation"
        deliverables:
          - "Comprehensive assessment"
          - "Capability validation"
          - "Strategic recommendations"

  scenario_library:
    technical_scenarios:
      - "Zero-day exploitation"
      - "APT campaign"
      - "Cryptojacking"
      - "DNS hijacking"
      - "Certificate compromise"

    business_scenarios:
      - "Major vendor breach"
      - "Regulatory investigation"
      - "Reputation crisis"
      - "Business email compromise"
      - "Physical security breach"

    combined_scenarios:
      - "Hybrid attack (cyber + physical)"
      - "Multi-vector attack"
      - "Cascading failures"
      - "Nation-state attack"

  evaluation_criteria:
    technical_capabilities:
      - detection_speed
      - containment_effectiveness
      - evidence_collection
      - system_recovery

    process_effectiveness:
      - escalation_timeliness
      - decision_quality
      - communication_clarity
      - coordination_efficiency

    team_performance:
      - role_clarity
      - stress_management
      - problem_solving
      - collaboration

  improvement_tracking:
    metrics:
      - "Time to detection"
      - "Time to containment"
      - "Decision accuracy"
      - "Communication effectiveness"
      - "Process adherence"

    maturity_assessment:
      level_1_initial:
        - "Ad hoc response"
        - "Undefined processes"
        - "Limited coordination"

      level_2_developing:
        - "Basic processes defined"
        - "Some coordination"
        - "Documented procedures"

      level_3_defined:
        - "Standardized processes"
        - "Clear roles"
        - "Regular exercises"

      level_4_managed:
        - "Measured performance"
        - "Continuous improvement"
        - "Integrated response"

      level_5_optimized:
        - "Predictive capabilities"
        - "Automated response"
        - "Advanced coordination"
```