# -*- coding: utf-8 -*-
"""
Control Test Cases - SEC-009 Phase 4

Concrete test implementations for all SOC 2 Trust Services Criteria.
Provides 48+ test cases covering CC6 (Logical Access), CC7 (System Operations),
CC8 (Change Management), A1 (Availability), and C1 (Confidentiality).

Each test class provides a set of control tests with:
    - Unique test IDs mapped to SOC 2 criteria
    - Test procedures and expected results
    - Evidence collection specifications
    - Pass/fail evaluation logic

Example:
    >>> cc6_tests = CC6Tests()
    >>> tests = cc6_tests.get_all_tests()
    >>> for test in tests:
    ...     framework.register_test(test)
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.soc2_preparation.control_testing.test_framework import (
    ControlTest,
    Evidence,
    Severity,
    TestResult,
    TestStatus,
    TestType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Control Test
# ---------------------------------------------------------------------------


class BaseControlTest(abc.ABC):
    """Abstract base class for SOC 2 control test suites.

    Provides a template for implementing criterion-specific test suites.
    Each subclass should implement get_all_tests() to return its test definitions.

    Attributes:
        criterion_prefix: The SOC 2 criterion prefix (e.g., "CC6").
        description: Description of what this criterion covers.
    """

    criterion_prefix: str = ""
    description: str = ""

    @abc.abstractmethod
    def get_all_tests(self) -> List[ControlTest]:
        """Return all control tests for this criterion.

        Returns:
            List of ControlTest definitions.
        """
        pass

    def create_test(
        self,
        test_num: str,
        description: str,
        procedure: str,
        expected_result: str,
        test_type: TestType = TestType.AUTOMATED,
        frequency: str = "quarterly",
        owner: str = "",
        tags: Optional[List[str]] = None,
    ) -> ControlTest:
        """Factory method to create a ControlTest with consistent formatting.

        Args:
            test_num: Test number within the criterion (e.g., "1.1").
            description: What is being tested.
            procedure: Step-by-step test procedure.
            expected_result: Expected outcome for passing.
            test_type: Execution method.
            frequency: How often to run.
            owner: Responsible party.
            tags: Labels for grouping.

        Returns:
            Configured ControlTest instance.
        """
        criterion_id = f"{self.criterion_prefix}.{test_num.split('.')[0]}"
        test_id = f"{self.criterion_prefix}.{test_num}"

        return ControlTest(
            test_id=test_id,
            criterion_id=criterion_id,
            test_type=test_type,
            description=description,
            procedure=procedure,
            expected_result=expected_result,
            frequency=frequency,
            owner=owner,
            tags=tags or [self.criterion_prefix.lower()],
        )


# ---------------------------------------------------------------------------
# CC6: Logical and Physical Access Controls
# ---------------------------------------------------------------------------


class CC6Tests(BaseControlTest):
    """Test cases for CC6: Logical and Physical Access Controls.

    Covers controls related to authentication, authorization,
    access provisioning, and access revocation.
    """

    criterion_prefix = "CC6"
    description = "Logical and Physical Access Controls"

    def get_all_tests(self) -> List[ControlTest]:
        """Return all CC6 control tests."""
        return [
            # CC6.1 - Logical Access Security Software
            self.test_mfa_enforcement(),
            self.test_password_policy(),
            self.test_session_timeout(),
            self.test_account_lockout(),
            self.test_single_sign_on(),
            # CC6.2 - Access Provisioning
            self.test_access_provisioning_sequence(),
            self.test_access_request_approval(),
            self.test_role_based_access(),
            self.test_least_privilege(),
            # CC6.3 - Access Revocation
            self.test_termination_access_removal(),
            self.test_role_change_access_review(),
            self.test_access_recertification(),
            # CC6.4 - Physical Access
            self.test_physical_access_controls(),
            self.test_visitor_management(),
            # CC6.5 - Logical Access Transmission
            self.test_encryption_in_transit(),
            self.test_vpn_access(),
            # CC6.6 - Logical Access Modification
            self.test_privileged_access_management(),
            self.test_service_account_management(),
            # CC6.7 - Logical Access Restrictions
            self.test_network_segmentation(),
            self.test_firewall_rules(),
        ]

    def test_mfa_enforcement(self) -> ControlTest:
        """Test CC6.1.1: MFA is enforced for all user accounts."""
        return self.create_test(
            test_num="1.1",
            description="Verify Multi-Factor Authentication (MFA) is enforced for all user accounts",
            procedure=(
                "1. Query authentication service for all active users\n"
                "2. Check MFA enrollment status for each user\n"
                "3. Verify MFA is required for login\n"
                "4. Test MFA bypass prevention"
            ),
            expected_result="100% of active users have MFA enabled and enforced",
            test_type=TestType.AUTOMATED,
            frequency="daily",
            owner="security-team",
            tags=["cc6", "mfa", "authentication"],
        )

    def test_password_policy(self) -> ControlTest:
        """Test CC6.1.2: Password policy meets security requirements."""
        return self.create_test(
            test_num="1.2",
            description="Verify password policy enforces minimum complexity and rotation requirements",
            procedure=(
                "1. Retrieve current password policy configuration\n"
                "2. Verify minimum length >= 12 characters\n"
                "3. Verify complexity requirements (upper, lower, number, special)\n"
                "4. Verify password history prevents reuse (last 12)\n"
                "5. Verify maximum age <= 90 days"
            ),
            expected_result="Password policy meets or exceeds SOC 2 requirements",
            test_type=TestType.AUTOMATED,
            frequency="weekly",
            owner="security-team",
            tags=["cc6", "password", "authentication"],
        )

    def test_session_timeout(self) -> ControlTest:
        """Test CC6.1.3: Session timeout is configured."""
        return self.create_test(
            test_num="1.3",
            description="Verify session timeout is enforced for inactive sessions",
            procedure=(
                "1. Check session timeout configuration\n"
                "2. Verify timeout <= 30 minutes for standard users\n"
                "3. Verify timeout <= 15 minutes for privileged users\n"
                "4. Test that sessions are invalidated after timeout"
            ),
            expected_result="Session timeout is configured and enforced per policy",
            test_type=TestType.AUTOMATED,
            frequency="weekly",
            owner="platform-team",
            tags=["cc6", "session", "authentication"],
        )

    def test_account_lockout(self) -> ControlTest:
        """Test CC6.1.4: Account lockout after failed attempts."""
        return self.create_test(
            test_num="1.4",
            description="Verify account lockout is triggered after failed login attempts",
            procedure=(
                "1. Check account lockout policy configuration\n"
                "2. Verify lockout threshold <= 5 failed attempts\n"
                "3. Verify lockout duration >= 30 minutes\n"
                "4. Test lockout mechanism with test account"
            ),
            expected_result="Accounts lock after 5 or fewer failed attempts",
            test_type=TestType.AUTOMATED,
            frequency="weekly",
            owner="security-team",
            tags=["cc6", "lockout", "authentication"],
        )

    def test_single_sign_on(self) -> ControlTest:
        """Test CC6.1.5: SSO is configured and enforced."""
        return self.create_test(
            test_num="1.5",
            description="Verify Single Sign-On (SSO) is configured for all applications",
            procedure=(
                "1. List all applications requiring authentication\n"
                "2. Verify SSO integration for each application\n"
                "3. Check that local authentication is disabled\n"
                "4. Verify SSO session management"
            ),
            expected_result="All applications use SSO with local auth disabled",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="monthly",
            owner="platform-team",
            tags=["cc6", "sso", "authentication"],
        )

    def test_access_provisioning_sequence(self) -> ControlTest:
        """Test CC6.2.1: Access provisioning follows approval workflow."""
        return self.create_test(
            test_num="2.1",
            description="Verify access provisioning follows documented approval workflow",
            procedure=(
                "1. Sample 25 recent access requests\n"
                "2. Verify each request has manager approval\n"
                "3. Verify access was provisioned after approval\n"
                "4. Verify access matches requested roles\n"
                "5. Check for unapproved access grants"
            ),
            expected_result="100% of sampled requests follow approval workflow",
            test_type=TestType.AUTOMATED,
            frequency="monthly",
            owner="access-management",
            tags=["cc6", "provisioning", "access"],
        )

    def test_access_request_approval(self) -> ControlTest:
        """Test CC6.2.2: Access requests require appropriate approval."""
        return self.create_test(
            test_num="2.2",
            description="Verify access requests require manager and data owner approval",
            procedure=(
                "1. Review access request workflow configuration\n"
                "2. Verify manager approval is required\n"
                "3. Verify data owner approval for sensitive systems\n"
                "4. Check approval delegation rules\n"
                "5. Test emergency access procedures"
            ),
            expected_result="All access requests require appropriate multi-level approval",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="monthly",
            owner="access-management",
            tags=["cc6", "approval", "access"],
        )

    def test_role_based_access(self) -> ControlTest:
        """Test CC6.2.3: Role-based access control is implemented."""
        return self.create_test(
            test_num="2.3",
            description="Verify role-based access control (RBAC) is implemented and documented",
            procedure=(
                "1. Review role definitions and permissions\n"
                "2. Verify roles align with job functions\n"
                "3. Check for orphaned permissions outside roles\n"
                "4. Verify role assignment process\n"
                "5. Review role documentation currency"
            ),
            expected_result="RBAC is implemented with documented, maintained roles",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="quarterly",
            owner="access-management",
            tags=["cc6", "rbac", "access"],
        )

    def test_least_privilege(self) -> ControlTest:
        """Test CC6.2.4: Least privilege principle is enforced."""
        return self.create_test(
            test_num="2.4",
            description="Verify access is granted based on least privilege principle",
            procedure=(
                "1. Sample user access across systems\n"
                "2. Compare access to job requirements\n"
                "3. Identify excessive permissions\n"
                "4. Review administrator access distribution\n"
                "5. Check for dormant high-privilege accounts"
            ),
            expected_result="No excessive permissions found in sampled accounts",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="quarterly",
            owner="access-management",
            tags=["cc6", "least-privilege", "access"],
        )

    def test_termination_access_removal(self) -> ControlTest:
        """Test CC6.3.1: Terminated user access is removed within 24 hours."""
        return self.create_test(
            test_num="3.1",
            description="Verify terminated user access is removed within 24 hours",
            procedure=(
                "1. Get list of users terminated in last 90 days\n"
                "2. Cross-reference with active accounts in all systems\n"
                "3. Verify termination timestamp vs. access removal\n"
                "4. Check for any residual access\n"
                "5. Review automated offboarding logs"
            ),
            expected_result="100% of terminated users have access removed within 24 hours",
            test_type=TestType.AUTOMATED,
            frequency="daily",
            owner="hr-operations",
            tags=["cc6", "termination", "offboarding"],
        )

    def test_role_change_access_review(self) -> ControlTest:
        """Test CC6.3.2: Role changes trigger access review."""
        return self.create_test(
            test_num="3.2",
            description="Verify job role changes trigger access recertification",
            procedure=(
                "1. Identify users with role changes in last 90 days\n"
                "2. Verify access review was triggered\n"
                "3. Check that old access was removed if not applicable\n"
                "4. Verify new access was properly provisioned"
            ),
            expected_result="All role changes have corresponding access reviews",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="monthly",
            owner="access-management",
            tags=["cc6", "role-change", "access"],
        )

    def test_access_recertification(self) -> ControlTest:
        """Test CC6.3.3: Periodic access recertification is performed."""
        return self.create_test(
            test_num="3.3",
            description="Verify quarterly access recertification is completed",
            procedure=(
                "1. Check last access recertification date\n"
                "2. Verify all managers completed certification\n"
                "3. Review certification completion rate\n"
                "4. Check remediation of identified issues\n"
                "5. Verify recertification evidence retention"
            ),
            expected_result="Quarterly access recertification completed with 100% manager participation",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="quarterly",
            owner="access-management",
            tags=["cc6", "recertification", "access"],
        )

    def test_physical_access_controls(self) -> ControlTest:
        """Test CC6.4.1: Physical access controls are in place."""
        return self.create_test(
            test_num="4.1",
            description="Verify physical access controls are implemented for data centers",
            procedure=(
                "1. Review physical access control configuration\n"
                "2. Verify badge access is required for all entry points\n"
                "3. Check surveillance camera coverage\n"
                "4. Review physical access logs\n"
                "5. Verify emergency exit monitoring"
            ),
            expected_result="Physical access controls meet security requirements",
            test_type=TestType.OBSERVATION,
            frequency="quarterly",
            owner="facilities",
            tags=["cc6", "physical", "data-center"],
        )

    def test_visitor_management(self) -> ControlTest:
        """Test CC6.4.2: Visitor management procedures are followed."""
        return self.create_test(
            test_num="4.2",
            description="Verify visitor management procedures for secure areas",
            procedure=(
                "1. Review visitor log for last 30 days\n"
                "2. Verify all visitors signed in/out\n"
                "3. Check escort requirements compliance\n"
                "4. Verify visitor badge return\n"
                "5. Review visitor access restrictions"
            ),
            expected_result="All visitors properly logged and escorted",
            test_type=TestType.OBSERVATION,
            frequency="monthly",
            owner="facilities",
            tags=["cc6", "visitor", "physical"],
        )

    def test_encryption_in_transit(self) -> ControlTest:
        """Test CC6.5.1: Data is encrypted in transit."""
        return self.create_test(
            test_num="5.1",
            description="Verify all data in transit is encrypted using TLS 1.2 or higher",
            procedure=(
                "1. Scan all public endpoints for TLS configuration\n"
                "2. Verify minimum TLS 1.2 is enforced\n"
                "3. Check certificate validity and chain\n"
                "4. Verify internal service-to-service encryption\n"
                "5. Test for SSL/TLS vulnerabilities"
            ),
            expected_result="All endpoints use TLS 1.2+ with valid certificates",
            test_type=TestType.AUTOMATED,
            frequency="weekly",
            owner="security-team",
            tags=["cc6", "encryption", "tls"],
        )

    def test_vpn_access(self) -> ControlTest:
        """Test CC6.5.2: VPN is required for remote access."""
        return self.create_test(
            test_num="5.2",
            description="Verify VPN is required for remote access to internal systems",
            procedure=(
                "1. Review VPN configuration and policy\n"
                "2. Verify VPN is required for internal access\n"
                "3. Check VPN authentication mechanism\n"
                "4. Review VPN access logs\n"
                "5. Test direct access prevention"
            ),
            expected_result="VPN required for all remote internal access",
            test_type=TestType.AUTOMATED,
            frequency="weekly",
            owner="network-team",
            tags=["cc6", "vpn", "remote-access"],
        )

    def test_privileged_access_management(self) -> ControlTest:
        """Test CC6.6.1: Privileged access is managed."""
        return self.create_test(
            test_num="6.1",
            description="Verify privileged access management (PAM) controls are in place",
            procedure=(
                "1. List all privileged accounts\n"
                "2. Verify PAM solution is in use\n"
                "3. Check privileged session recording\n"
                "4. Verify just-in-time access provisioning\n"
                "5. Review privileged access audit logs"
            ),
            expected_result="All privileged access is managed through PAM",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="monthly",
            owner="security-team",
            tags=["cc6", "pam", "privileged"],
        )

    def test_service_account_management(self) -> ControlTest:
        """Test CC6.6.2: Service accounts are properly managed."""
        return self.create_test(
            test_num="6.2",
            description="Verify service accounts have documented owners and are reviewed",
            procedure=(
                "1. List all service accounts\n"
                "2. Verify each has documented owner\n"
                "3. Check password/key rotation schedule\n"
                "4. Verify least privilege for service accounts\n"
                "5. Review service account activity logs"
            ),
            expected_result="All service accounts have owners and proper management",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="monthly",
            owner="platform-team",
            tags=["cc6", "service-account", "access"],
        )

    def test_network_segmentation(self) -> ControlTest:
        """Test CC6.7.1: Network segmentation is implemented."""
        return self.create_test(
            test_num="7.1",
            description="Verify network segmentation isolates critical systems",
            procedure=(
                "1. Review network architecture diagram\n"
                "2. Verify production/non-production separation\n"
                "3. Check database tier isolation\n"
                "4. Test cross-segment access controls\n"
                "5. Verify jump host requirements"
            ),
            expected_result="Network properly segmented with documented access controls",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="quarterly",
            owner="network-team",
            tags=["cc6", "segmentation", "network"],
        )

    def test_firewall_rules(self) -> ControlTest:
        """Test CC6.7.2: Firewall rules follow least privilege."""
        return self.create_test(
            test_num="7.2",
            description="Verify firewall rules are documented and follow least privilege",
            procedure=(
                "1. Extract current firewall rules\n"
                "2. Compare against documented baseline\n"
                "3. Identify overly permissive rules\n"
                "4. Check for unused rules\n"
                "5. Verify change management for rule changes"
            ),
            expected_result="All firewall rules documented and follow least privilege",
            test_type=TestType.AUTOMATED,
            frequency="monthly",
            owner="network-team",
            tags=["cc6", "firewall", "network"],
        )


# ---------------------------------------------------------------------------
# CC7: System Operations
# ---------------------------------------------------------------------------


class CC7Tests(BaseControlTest):
    """Test cases for CC7: System Operations.

    Covers controls related to vulnerability management, monitoring,
    incident detection, and system maintenance.
    """

    criterion_prefix = "CC7"
    description = "System Operations"

    def get_all_tests(self) -> List[ControlTest]:
        """Return all CC7 control tests."""
        return [
            # CC7.1 - Detection and Monitoring
            self.test_vulnerability_scanning(),
            self.test_monitoring_alerts(),
            self.test_incident_detection(),
            self.test_security_logging(),
            self.test_log_retention(),
            # CC7.2 - Incident Response
            self.test_incident_response_plan(),
            self.test_incident_communication(),
            self.test_incident_documentation(),
            # CC7.3 - Recovery
            self.test_recovery_procedures(),
            self.test_recovery_testing(),
            # CC7.4 - Patch Management
            self.test_patch_management(),
            self.test_critical_patch_timeline(),
        ]

    def test_vulnerability_scanning(self) -> ControlTest:
        """Test CC7.1.1: Regular vulnerability scanning is performed."""
        return self.create_test(
            test_num="1.1",
            description="Verify vulnerability scanning is performed on all systems",
            procedure=(
                "1. Review vulnerability scanning schedule\n"
                "2. Verify all assets are included in scans\n"
                "3. Check most recent scan results\n"
                "4. Verify critical findings are tracked\n"
                "5. Review remediation SLAs and compliance"
            ),
            expected_result="Weekly vulnerability scans with documented remediation",
            test_type=TestType.AUTOMATED,
            frequency="weekly",
            owner="security-team",
            tags=["cc7", "vulnerability", "scanning"],
        )

    def test_monitoring_alerts(self) -> ControlTest:
        """Test CC7.1.2: Security monitoring alerts are configured."""
        return self.create_test(
            test_num="1.2",
            description="Verify security monitoring and alerting is configured",
            procedure=(
                "1. Review monitoring tool configuration\n"
                "2. Verify critical alert rules are active\n"
                "3. Check alert notification channels\n"
                "4. Test sample alert triggering\n"
                "5. Review alert response SLAs"
            ),
            expected_result="Security alerts configured with defined response procedures",
            test_type=TestType.AUTOMATED,
            frequency="daily",
            owner="soc-team",
            tags=["cc7", "monitoring", "alerts"],
        )

    def test_incident_detection(self) -> ControlTest:
        """Test CC7.1.3: Security incident detection capabilities."""
        return self.create_test(
            test_num="1.3",
            description="Verify security incident detection mechanisms are operational",
            procedure=(
                "1. Review SIEM/detection tool configuration\n"
                "2. Verify detection rules for common attacks\n"
                "3. Check correlation rule effectiveness\n"
                "4. Test detection with simulated incidents\n"
                "5. Review detection coverage metrics"
            ),
            expected_result="Incident detection covers major threat categories",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="monthly",
            owner="soc-team",
            tags=["cc7", "detection", "siem"],
        )

    def test_security_logging(self) -> ControlTest:
        """Test CC7.1.4: Security events are logged."""
        return self.create_test(
            test_num="1.4",
            description="Verify security events are logged across all systems",
            procedure=(
                "1. Review log collection configuration\n"
                "2. Verify authentication events are logged\n"
                "3. Verify authorization failures are logged\n"
                "4. Check administrative action logging\n"
                "5. Verify log integrity mechanisms"
            ),
            expected_result="All security events logged with tamper protection",
            test_type=TestType.AUTOMATED,
            frequency="weekly",
            owner="platform-team",
            tags=["cc7", "logging", "audit"],
        )

    def test_log_retention(self) -> ControlTest:
        """Test CC7.1.5: Log retention meets requirements."""
        return self.create_test(
            test_num="1.5",
            description="Verify log retention meets compliance requirements",
            procedure=(
                "1. Review log retention policy\n"
                "2. Verify operational logs retained 90+ days\n"
                "3. Verify security logs retained 1+ year\n"
                "4. Check compliance/audit logs retained 7+ years\n"
                "5. Verify log archive accessibility"
            ),
            expected_result="Log retention meets or exceeds policy requirements",
            test_type=TestType.AUTOMATED,
            frequency="monthly",
            owner="platform-team",
            tags=["cc7", "retention", "logging"],
        )

    def test_incident_response_plan(self) -> ControlTest:
        """Test CC7.2.1: Incident response plan is documented."""
        return self.create_test(
            test_num="2.1",
            description="Verify incident response plan is documented and current",
            procedure=(
                "1. Review incident response plan document\n"
                "2. Verify plan was reviewed within 12 months\n"
                "3. Check roles and responsibilities are defined\n"
                "4. Verify escalation procedures are documented\n"
                "5. Review incident classification criteria"
            ),
            expected_result="Current incident response plan with defined procedures",
            test_type=TestType.MANUAL,
            frequency="quarterly",
            owner="security-team",
            tags=["cc7", "incident-response", "policy"],
        )

    def test_incident_communication(self) -> ControlTest:
        """Test CC7.2.2: Incident communication procedures exist."""
        return self.create_test(
            test_num="2.2",
            description="Verify incident communication procedures are defined",
            procedure=(
                "1. Review communication plan\n"
                "2. Verify internal notification procedures\n"
                "3. Check customer notification criteria\n"
                "4. Verify regulatory notification requirements\n"
                "5. Test communication channels"
            ),
            expected_result="Communication procedures defined for all stakeholders",
            test_type=TestType.MANUAL,
            frequency="quarterly",
            owner="security-team",
            tags=["cc7", "incident-response", "communication"],
        )

    def test_incident_documentation(self) -> ControlTest:
        """Test CC7.2.3: Incidents are properly documented."""
        return self.create_test(
            test_num="2.3",
            description="Verify security incidents are documented and reviewed",
            procedure=(
                "1. Sample 10 recent security incidents\n"
                "2. Verify each has complete documentation\n"
                "3. Check root cause analysis completion\n"
                "4. Verify remediation actions documented\n"
                "5. Review lessons learned documentation"
            ),
            expected_result="All sampled incidents fully documented with RCA",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="monthly",
            owner="security-team",
            tags=["cc7", "incident-response", "documentation"],
        )

    def test_recovery_procedures(self) -> ControlTest:
        """Test CC7.3.1: Recovery procedures are documented."""
        return self.create_test(
            test_num="3.1",
            description="Verify system recovery procedures are documented",
            procedure=(
                "1. Review disaster recovery plan\n"
                "2. Verify recovery procedures for critical systems\n"
                "3. Check RTO/RPO definitions\n"
                "4. Review runbook documentation\n"
                "5. Verify procedure currency"
            ),
            expected_result="Recovery procedures documented with RTO/RPO targets",
            test_type=TestType.MANUAL,
            frequency="quarterly",
            owner="platform-team",
            tags=["cc7", "recovery", "disaster-recovery"],
        )

    def test_recovery_testing(self) -> ControlTest:
        """Test CC7.3.2: Recovery procedures are tested."""
        return self.create_test(
            test_num="3.2",
            description="Verify disaster recovery testing is performed annually",
            procedure=(
                "1. Review last DR test date and results\n"
                "2. Verify all critical systems were tested\n"
                "3. Check RTO/RPO targets were met\n"
                "4. Review issues identified and remediated\n"
                "5. Verify lessons learned documented"
            ),
            expected_result="Annual DR test completed with documented results",
            test_type=TestType.MANUAL,
            frequency="annually",
            owner="platform-team",
            tags=["cc7", "recovery", "dr-testing"],
        )

    def test_patch_management(self) -> ControlTest:
        """Test CC7.4.1: Patch management process is followed."""
        return self.create_test(
            test_num="4.1",
            description="Verify patch management process is implemented",
            procedure=(
                "1. Review patch management policy\n"
                "2. Check patching schedule and cadence\n"
                "3. Verify patch testing procedures\n"
                "4. Review patch deployment evidence\n"
                "5. Check for unpatched systems"
            ),
            expected_result="Patch management process followed with documented deployments",
            test_type=TestType.AUTOMATED,
            frequency="monthly",
            owner="platform-team",
            tags=["cc7", "patch", "vulnerability"],
        )

    def test_critical_patch_timeline(self) -> ControlTest:
        """Test CC7.4.2: Critical patches are applied within SLA."""
        return self.create_test(
            test_num="4.2",
            description="Verify critical patches are applied within defined SLA",
            procedure=(
                "1. Identify critical vulnerabilities in last 90 days\n"
                "2. Check patch deployment dates\n"
                "3. Calculate time from disclosure to patch\n"
                "4. Verify SLA compliance (critical: 7 days)\n"
                "5. Document any exceptions"
            ),
            expected_result="Critical patches applied within 7-day SLA",
            test_type=TestType.AUTOMATED,
            frequency="weekly",
            owner="platform-team",
            tags=["cc7", "patch", "sla"],
        )


# ---------------------------------------------------------------------------
# CC8: Change Management
# ---------------------------------------------------------------------------


class CC8Tests(BaseControlTest):
    """Test cases for CC8: Change Management.

    Covers controls related to change authorization, testing,
    deployment, and documentation.
    """

    criterion_prefix = "CC8"
    description = "Change Management"

    def get_all_tests(self) -> List[ControlTest]:
        """Return all CC8 control tests."""
        return [
            self.test_change_approval_workflow(),
            self.test_change_testing(),
            self.test_change_documentation(),
            self.test_emergency_change_process(),
            self.test_deployment_automation(),
            self.test_rollback_capability(),
            self.test_segregation_of_duties(),
            self.test_change_review(),
        ]

    def test_change_approval_workflow(self) -> ControlTest:
        """Test CC8.1.1: Changes follow approval workflow."""
        return self.create_test(
            test_num="1.1",
            description="Verify all changes follow documented approval workflow",
            procedure=(
                "1. Sample 25 recent production changes\n"
                "2. Verify each has change request ticket\n"
                "3. Check for required approvals\n"
                "4. Verify testing evidence attached\n"
                "5. Check deployment matches approved change"
            ),
            expected_result="100% of sampled changes have proper approvals",
            test_type=TestType.AUTOMATED,
            frequency="monthly",
            owner="change-management",
            tags=["cc8", "change", "approval"],
        )

    def test_change_testing(self) -> ControlTest:
        """Test CC8.1.2: Changes are tested before deployment."""
        return self.create_test(
            test_num="1.2",
            description="Verify changes are tested in non-production before deployment",
            procedure=(
                "1. Sample 25 recent production deployments\n"
                "2. Verify each was deployed to staging first\n"
                "3. Check test execution evidence\n"
                "4. Verify test coverage meets standards\n"
                "5. Check for direct production deployments"
            ),
            expected_result="All changes tested in non-production first",
            test_type=TestType.AUTOMATED,
            frequency="monthly",
            owner="devops-team",
            tags=["cc8", "testing", "deployment"],
        )

    def test_change_documentation(self) -> ControlTest:
        """Test CC8.1.3: Changes are documented."""
        return self.create_test(
            test_num="1.3",
            description="Verify changes are documented with impact and rollback plans",
            procedure=(
                "1. Sample 25 recent change requests\n"
                "2. Verify description of change\n"
                "3. Check impact assessment\n"
                "4. Verify rollback plan documented\n"
                "5. Check post-implementation review"
            ),
            expected_result="All changes have complete documentation",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="monthly",
            owner="change-management",
            tags=["cc8", "documentation", "change"],
        )

    def test_emergency_change_process(self) -> ControlTest:
        """Test CC8.1.4: Emergency changes follow expedited process."""
        return self.create_test(
            test_num="1.4",
            description="Verify emergency changes follow documented expedited process",
            procedure=(
                "1. Sample emergency changes from last 90 days\n"
                "2. Verify emergency justification\n"
                "3. Check for expedited approval\n"
                "4. Verify post-implementation documentation\n"
                "5. Check CAB review of emergency changes"
            ),
            expected_result="Emergency changes follow expedited process with CAB review",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="monthly",
            owner="change-management",
            tags=["cc8", "emergency", "change"],
        )

    def test_deployment_automation(self) -> ControlTest:
        """Test CC8.1.5: Deployments use automated pipelines."""
        return self.create_test(
            test_num="1.5",
            description="Verify deployments use automated CI/CD pipelines",
            procedure=(
                "1. Review CI/CD pipeline configuration\n"
                "2. Verify automated testing gates\n"
                "3. Check deployment automation\n"
                "4. Verify manual deployment is restricted\n"
                "5. Review pipeline audit logs"
            ),
            expected_result="Deployments automated with quality gates",
            test_type=TestType.AUTOMATED,
            frequency="weekly",
            owner="devops-team",
            tags=["cc8", "cicd", "automation"],
        )

    def test_rollback_capability(self) -> ControlTest:
        """Test CC8.1.6: Rollback capability exists for deployments."""
        return self.create_test(
            test_num="1.6",
            description="Verify rollback capability exists and is tested",
            procedure=(
                "1. Review deployment rollback procedures\n"
                "2. Verify rollback automation exists\n"
                "3. Check last rollback test date\n"
                "4. Review actual rollback events\n"
                "5. Verify rollback time targets"
            ),
            expected_result="Rollback capability tested and documented",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="quarterly",
            owner="devops-team",
            tags=["cc8", "rollback", "deployment"],
        )

    def test_segregation_of_duties(self) -> ControlTest:
        """Test CC8.1.7: Segregation of duties in change management."""
        return self.create_test(
            test_num="1.7",
            description="Verify segregation of duties between development and deployment",
            procedure=(
                "1. Review access controls on code repositories\n"
                "2. Verify developer cannot merge to main\n"
                "3. Check approval requirements for merges\n"
                "4. Verify production deployment restrictions\n"
                "5. Review access audit logs"
            ),
            expected_result="SoD enforced between development and deployment",
            test_type=TestType.AUTOMATED,
            frequency="monthly",
            owner="security-team",
            tags=["cc8", "sod", "access"],
        )

    def test_change_review(self) -> ControlTest:
        """Test CC8.1.8: Changes are reviewed post-implementation."""
        return self.create_test(
            test_num="1.8",
            description="Verify changes are reviewed after implementation",
            procedure=(
                "1. Sample 25 recent changes\n"
                "2. Check for post-implementation review\n"
                "3. Verify issues are documented\n"
                "4. Check lessons learned capture\n"
                "5. Verify metrics tracking"
            ),
            expected_result="All changes have post-implementation review",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="monthly",
            owner="change-management",
            tags=["cc8", "review", "change"],
        )


# ---------------------------------------------------------------------------
# A1: Availability
# ---------------------------------------------------------------------------


class A1Tests(BaseControlTest):
    """Test cases for A1: Availability.

    Covers controls related to backup, disaster recovery,
    capacity planning, and system availability.
    """

    criterion_prefix = "A1"
    description = "Availability"

    def get_all_tests(self) -> List[ControlTest]:
        """Return all A1 control tests."""
        return [
            self.test_backup_verification(),
            self.test_backup_encryption(),
            self.test_backup_offsite(),
            self.test_disaster_recovery_testing(),
            self.test_capacity_monitoring(),
            self.test_high_availability(),
            self.test_sla_monitoring(),
            self.test_maintenance_windows(),
        ]

    def test_backup_verification(self) -> ControlTest:
        """Test A1.1.1: Backups are verified and restorable."""
        return self.create_test(
            test_num="1.1",
            description="Verify backups are completed and periodically tested for restoration",
            procedure=(
                "1. Review backup job success rate\n"
                "2. Verify backup completion for all critical systems\n"
                "3. Check last backup restoration test date\n"
                "4. Verify restoration met RTO targets\n"
                "5. Review backup monitoring alerts"
            ),
            expected_result="Backups successful with quarterly restoration tests",
            test_type=TestType.AUTOMATED,
            frequency="daily",
            owner="platform-team",
            tags=["a1", "backup", "restoration"],
        )

    def test_backup_encryption(self) -> ControlTest:
        """Test A1.1.2: Backups are encrypted."""
        return self.create_test(
            test_num="1.2",
            description="Verify backups are encrypted at rest",
            procedure=(
                "1. Review backup encryption configuration\n"
                "2. Verify encryption algorithm (AES-256)\n"
                "3. Check key management procedures\n"
                "4. Verify encryption is enforced\n"
                "5. Review unencrypted backup alerts"
            ),
            expected_result="All backups encrypted with AES-256",
            test_type=TestType.AUTOMATED,
            frequency="weekly",
            owner="platform-team",
            tags=["a1", "backup", "encryption"],
        )

    def test_backup_offsite(self) -> ControlTest:
        """Test A1.1.3: Backups are stored offsite."""
        return self.create_test(
            test_num="1.3",
            description="Verify backups are replicated to offsite location",
            procedure=(
                "1. Review backup replication configuration\n"
                "2. Verify geographic separation\n"
                "3. Check replication success rate\n"
                "4. Verify offsite backup accessibility\n"
                "5. Test cross-region restoration"
            ),
            expected_result="Backups replicated to geographically separate location",
            test_type=TestType.AUTOMATED,
            frequency="weekly",
            owner="platform-team",
            tags=["a1", "backup", "offsite"],
        )

    def test_disaster_recovery_testing(self) -> ControlTest:
        """Test A1.2.1: Disaster recovery is tested annually."""
        return self.create_test(
            test_num="2.1",
            description="Verify disaster recovery procedures are tested at least annually",
            procedure=(
                "1. Review last DR test date\n"
                "2. Verify all critical systems tested\n"
                "3. Check RTO/RPO targets met\n"
                "4. Review test documentation\n"
                "5. Verify remediation of issues"
            ),
            expected_result="Annual DR test completed with documented results",
            test_type=TestType.MANUAL,
            frequency="annually",
            owner="platform-team",
            tags=["a1", "dr", "testing"],
        )

    def test_capacity_monitoring(self) -> ControlTest:
        """Test A1.3.1: Capacity is monitored and planned."""
        return self.create_test(
            test_num="3.1",
            description="Verify capacity monitoring and planning is in place",
            procedure=(
                "1. Review capacity monitoring dashboards\n"
                "2. Verify alerting thresholds (80%)\n"
                "3. Check capacity trending reports\n"
                "4. Review capacity planning process\n"
                "5. Verify scaling procedures"
            ),
            expected_result="Capacity monitored with proactive planning",
            test_type=TestType.AUTOMATED,
            frequency="weekly",
            owner="platform-team",
            tags=["a1", "capacity", "monitoring"],
        )

    def test_high_availability(self) -> ControlTest:
        """Test A1.3.2: High availability is implemented."""
        return self.create_test(
            test_num="3.2",
            description="Verify high availability configuration for critical systems",
            procedure=(
                "1. Review HA architecture documentation\n"
                "2. Verify redundancy for critical components\n"
                "3. Check failover configuration\n"
                "4. Test automatic failover\n"
                "5. Review HA test results"
            ),
            expected_result="HA implemented with tested automatic failover",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="quarterly",
            owner="platform-team",
            tags=["a1", "ha", "redundancy"],
        )

    def test_sla_monitoring(self) -> ControlTest:
        """Test A1.3.3: SLA is monitored and reported."""
        return self.create_test(
            test_num="3.3",
            description="Verify system availability SLA is monitored and reported",
            procedure=(
                "1. Review SLA definitions\n"
                "2. Check uptime monitoring configuration\n"
                "3. Verify SLA reporting frequency\n"
                "4. Review last 12 months SLA performance\n"
                "5. Check SLA breach handling"
            ),
            expected_result="99.9% uptime SLA monitored and maintained",
            test_type=TestType.AUTOMATED,
            frequency="monthly",
            owner="platform-team",
            tags=["a1", "sla", "monitoring"],
        )

    def test_maintenance_windows(self) -> ControlTest:
        """Test A1.3.4: Maintenance windows are scheduled and communicated."""
        return self.create_test(
            test_num="3.4",
            description="Verify maintenance windows are scheduled and communicated",
            procedure=(
                "1. Review maintenance window policy\n"
                "2. Check scheduled maintenance calendar\n"
                "3. Verify customer notification process\n"
                "4. Review maintenance execution logs\n"
                "5. Check unplanned downtime incidents"
            ),
            expected_result="Maintenance windows scheduled with advance notice",
            test_type=TestType.MANUAL,
            frequency="monthly",
            owner="platform-team",
            tags=["a1", "maintenance", "communication"],
        )


# ---------------------------------------------------------------------------
# C1: Confidentiality
# ---------------------------------------------------------------------------


class C1Tests(BaseControlTest):
    """Test cases for C1: Confidentiality.

    Covers controls related to data classification, encryption,
    access restrictions, and data handling.
    """

    criterion_prefix = "C1"
    description = "Confidentiality"

    def get_all_tests(self) -> List[ControlTest]:
        """Return all C1 control tests."""
        return [
            self.test_encryption_at_rest(),
            self.test_data_classification(),
            self.test_pii_protection(),
            self.test_data_masking(),
            self.test_key_management(),
            self.test_data_retention(),
            self.test_secure_deletion(),
            self.test_dlp_controls(),
        ]

    def test_encryption_at_rest(self) -> ControlTest:
        """Test C1.1.1: Data is encrypted at rest."""
        return self.create_test(
            test_num="1.1",
            description="Verify all sensitive data is encrypted at rest",
            procedure=(
                "1. Review encryption configuration for databases\n"
                "2. Verify object storage encryption\n"
                "3. Check encryption for file systems\n"
                "4. Verify encryption algorithm (AES-256)\n"
                "5. Review encryption audit logs"
            ),
            expected_result="All sensitive data encrypted at rest with AES-256",
            test_type=TestType.AUTOMATED,
            frequency="weekly",
            owner="security-team",
            tags=["c1", "encryption", "data-at-rest"],
        )

    def test_data_classification(self) -> ControlTest:
        """Test C1.1.2: Data classification is implemented."""
        return self.create_test(
            test_num="1.2",
            description="Verify data classification scheme is implemented and followed",
            procedure=(
                "1. Review data classification policy\n"
                "2. Verify classification labels exist\n"
                "3. Check data is tagged appropriately\n"
                "4. Review classification accuracy\n"
                "5. Verify handling procedures by level"
            ),
            expected_result="Data classification implemented with tagged data",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="quarterly",
            owner="data-governance",
            tags=["c1", "classification", "data"],
        )

    def test_pii_protection(self) -> ControlTest:
        """Test C1.1.3: PII is protected."""
        return self.create_test(
            test_num="1.3",
            description="Verify personally identifiable information (PII) is protected",
            procedure=(
                "1. Review PII handling procedures\n"
                "2. Verify PII is identified and inventoried\n"
                "3. Check access controls on PII\n"
                "4. Verify PII encryption\n"
                "5. Review PII access logs"
            ),
            expected_result="PII identified, encrypted, and access controlled",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="monthly",
            owner="data-governance",
            tags=["c1", "pii", "privacy"],
        )

    def test_data_masking(self) -> ControlTest:
        """Test C1.1.4: Data masking in non-production."""
        return self.create_test(
            test_num="1.4",
            description="Verify sensitive data is masked in non-production environments",
            procedure=(
                "1. Review data masking policy\n"
                "2. Check non-production data sources\n"
                "3. Verify PII is masked/tokenized\n"
                "4. Test for unmasked sensitive data\n"
                "5. Review masking automation"
            ),
            expected_result="Sensitive data masked in all non-production environments",
            test_type=TestType.AUTOMATED,
            frequency="monthly",
            owner="data-governance",
            tags=["c1", "masking", "non-production"],
        )

    def test_key_management(self) -> ControlTest:
        """Test C1.2.1: Encryption keys are properly managed."""
        return self.create_test(
            test_num="2.1",
            description="Verify encryption key management follows best practices",
            procedure=(
                "1. Review key management procedures\n"
                "2. Verify keys stored in HSM/KMS\n"
                "3. Check key rotation schedule\n"
                "4. Verify key access controls\n"
                "5. Review key audit logs"
            ),
            expected_result="Keys managed in HSM/KMS with annual rotation",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="quarterly",
            owner="security-team",
            tags=["c1", "key-management", "encryption"],
        )

    def test_data_retention(self) -> ControlTest:
        """Test C1.3.1: Data retention policy is enforced."""
        return self.create_test(
            test_num="3.1",
            description="Verify data retention policy is implemented and enforced",
            procedure=(
                "1. Review data retention policy\n"
                "2. Verify retention periods by data type\n"
                "3. Check automated deletion jobs\n"
                "4. Verify no data beyond retention period\n"
                "5. Review retention compliance reports"
            ),
            expected_result="Data retention policy enforced with automated deletion",
            test_type=TestType.AUTOMATED,
            frequency="monthly",
            owner="data-governance",
            tags=["c1", "retention", "data"],
        )

    def test_secure_deletion(self) -> ControlTest:
        """Test C1.3.2: Secure data deletion is performed."""
        return self.create_test(
            test_num="3.2",
            description="Verify secure deletion procedures are followed",
            procedure=(
                "1. Review secure deletion policy\n"
                "2. Verify deletion method meets standards\n"
                "3. Check deletion certificates for hardware\n"
                "4. Verify backup deletion includes old data\n"
                "5. Review deletion audit logs"
            ),
            expected_result="Secure deletion verified with certificates",
            test_type=TestType.MANUAL,
            frequency="quarterly",
            owner="it-operations",
            tags=["c1", "deletion", "secure"],
        )

    def test_dlp_controls(self) -> ControlTest:
        """Test C1.4.1: Data loss prevention controls are in place."""
        return self.create_test(
            test_num="4.1",
            description="Verify data loss prevention (DLP) controls are operational",
            procedure=(
                "1. Review DLP policy configuration\n"
                "2. Verify email DLP rules\n"
                "3. Check endpoint DLP controls\n"
                "4. Review DLP incident reports\n"
                "5. Test DLP effectiveness"
            ),
            expected_result="DLP controls active and monitoring data movement",
            test_type=TestType.SEMI_AUTOMATED,
            frequency="monthly",
            owner="security-team",
            tags=["c1", "dlp", "data-protection"],
        )


__all__ = [
    "BaseControlTest",
    "CC6Tests",
    "CC7Tests",
    "CC8Tests",
    "A1Tests",
    "C1Tests",
]
