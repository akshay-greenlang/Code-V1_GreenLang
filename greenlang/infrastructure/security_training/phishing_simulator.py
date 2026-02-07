# -*- coding: utf-8 -*-
"""
Security Training Phishing Simulator - SEC-010

Automated phishing simulation campaign management for security awareness testing.
Supports multiple attack templates, email generation, interaction tracking,
automatic training enrollment for clickers, and comprehensive reporting.

Classes:
    - PhishingSimulator: Main class for phishing campaign management

Features:
    - 6 phishing template types (credential harvest, CEO fraud, etc.)
    - Personalized email generation
    - Tracking pixels for open detection
    - Link click and credential entry tracking
    - Phishing report tracking (positive behavior)
    - Auto-enrollment in training for vulnerable users
    - Campaign metrics and PDF report generation

Example:
    >>> from greenlang.infrastructure.security_training.phishing_simulator import (
    ...     PhishingSimulator,
    ... )
    >>> simulator = PhishingSimulator()
    >>> campaign = await simulator.create_campaign(
    ...     name="Q1 Awareness Test",
    ...     template_type=TemplateType.CREDENTIAL_HARVEST,
    ...     target_users=["user-1", "user-2"],
    ... )
    >>> await simulator.send_phishing_emails(campaign.id)
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.security_training.models import (
    CampaignMetrics,
    CampaignStatus,
    PhishingCampaign,
    PhishingResult,
    TemplateType,
)
from greenlang.infrastructure.security_training.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phishing Templates
# ---------------------------------------------------------------------------


PHISHING_TEMPLATES: Dict[TemplateType, Dict[str, Any]] = {
    TemplateType.CREDENTIAL_HARVEST: {
        "subject": "Action Required: Verify Your Account",
        "sender_name": "IT Security Team",
        "sender_prefix": "security-noreply",
        "body_html": """
<html>
<body style="font-family: Arial, sans-serif;">
<p>Dear {user_name},</p>

<p>We've detected unusual activity on your account. To protect your data,
please verify your identity by clicking the link below:</p>

<p><a href="{tracking_url}" style="background-color: #0066cc; color: white;
padding: 10px 20px; text-decoration: none; border-radius: 4px;">
Verify Account Now</a></p>

<p>If you don't verify within 24 hours, your account may be suspended.</p>

<p>Best regards,<br>IT Security Team</p>

<img src="{tracking_pixel}" width="1" height="1" />
</body>
</html>
""",
        "indicators": [
            "Urgency/deadline pressure",
            "Generic greeting",
            "Suspicious sender address",
            "Request for credentials",
        ],
    },
    TemplateType.MALICIOUS_ATTACHMENT: {
        "subject": "Invoice #INV-{invoice_number} - Payment Due",
        "sender_name": "Accounts Payable",
        "sender_prefix": "ap-invoices",
        "body_html": """
<html>
<body style="font-family: Arial, sans-serif;">
<p>Hi {user_name},</p>

<p>Please find attached the invoice for services rendered. Payment is due
within 7 days.</p>

<p>If you have questions, click here to view the invoice details:
<a href="{tracking_url}">View Invoice (INV-{invoice_number}.pdf)</a></p>

<p>Thank you for your prompt attention.</p>

<p>Accounts Payable Department</p>

<img src="{tracking_pixel}" width="1" height="1" />
</body>
</html>
""",
        "indicators": [
            "Unexpected invoice",
            "Unknown sender",
            "Request to open attachment",
            "Payment pressure",
        ],
    },
    TemplateType.FAKE_INVOICE: {
        "subject": "Overdue Invoice - Immediate Action Required",
        "sender_name": "Vendor Payments",
        "sender_prefix": "vendor-payments",
        "body_html": """
<html>
<body style="font-family: Arial, sans-serif;">
<p>Dear {user_name},</p>

<p>Our records show an overdue payment of ${amount} for invoice #{invoice_number}.</p>

<p>To avoid service interruption, please submit payment immediately:</p>

<p><a href="{tracking_url}" style="background-color: #cc0000; color: white;
padding: 10px 20px; text-decoration: none; border-radius: 4px;">
Pay Now</a></p>

<p>Contact us if you believe this is an error.</p>

<p>Vendor Payments Team</p>

<img src="{tracking_pixel}" width="1" height="1" />
</body>
</html>
""",
        "indicators": [
            "Payment urgency",
            "Unknown vendor",
            "No previous correspondence",
            "Direct payment link",
        ],
    },
    TemplateType.URGENT_ACTION: {
        "subject": "[URGENT] Your Account Will Be Suspended",
        "sender_name": "System Administrator",
        "sender_prefix": "admin-alerts",
        "body_html": """
<html>
<body style="font-family: Arial, sans-serif;">
<p style="color: red; font-weight: bold;">URGENT: Action Required</p>

<p>Dear {user_name},</p>

<p>Your account is scheduled for suspension due to a policy violation.
To prevent this, you must verify your account immediately:</p>

<p><a href="{tracking_url}" style="background-color: #ff6600; color: white;
padding: 10px 20px; text-decoration: none; border-radius: 4px;">
Verify Now to Prevent Suspension</a></p>

<p>You have 2 hours to complete this verification.</p>

<p>IT Department</p>

<img src="{tracking_pixel}" width="1" height="1" />
</body>
</html>
""",
        "indicators": [
            "Extreme urgency",
            "Threat of consequences",
            "Unusual request",
            "Suspicious timing",
        ],
    },
    TemplateType.CEO_FRAUD: {
        "subject": "Quick favor needed",
        "sender_name": "{ceo_name}",
        "sender_prefix": "{ceo_email_prefix}",
        "body_html": """
<html>
<body style="font-family: Arial, sans-serif;">
<p>{user_name},</p>

<p>I'm in a meeting and need your help with something urgent and confidential.</p>

<p>Can you process a wire transfer for me? I'll explain the details when
I'm out of this meeting. Click here to review the request:</p>

<p><a href="{tracking_url}">Review Transfer Request</a></p>

<p>Please keep this confidential for now.</p>

<p>Thanks,<br>{ceo_name}</p>

<p style="font-size: 11px; color: #666;">Sent from my iPhone</p>

<img src="{tracking_pixel}" width="1" height="1" />
</body>
</html>
""",
        "indicators": [
            "Unusual request from executive",
            "Request for confidentiality",
            "Urgency with no details",
            "Sent from mobile",
        ],
    },
    TemplateType.IT_SUPPORT: {
        "subject": "Password Expiration Notice - Action Required",
        "sender_name": "IT Help Desk",
        "sender_prefix": "helpdesk",
        "body_html": """
<html>
<body style="font-family: Arial, sans-serif;">
<p>Hello {user_name},</p>

<p>Your network password will expire in 24 hours. To avoid being locked out,
please update your password now:</p>

<p><a href="{tracking_url}" style="background-color: #0066cc; color: white;
padding: 10px 20px; text-decoration: none; border-radius: 4px;">
Update Password</a></p>

<p>If you have questions, contact the Help Desk at ext. 1234.</p>

<p>IT Help Desk<br>
<em>Helping you stay productive</em></p>

<img src="{tracking_pixel}" width="1" height="1" />
</body>
</html>
""",
        "indicators": [
            "Password change request via email",
            "Direct link to 'update' password",
            "Time pressure",
            "Generic IT branding",
        ],
    },
}


# ---------------------------------------------------------------------------
# Phishing Simulator Class
# ---------------------------------------------------------------------------


class PhishingSimulator:
    """Phishing simulation campaign manager.

    Handles the complete lifecycle of phishing simulation campaigns including
    campaign creation, email generation, sending, tracking, and reporting.

    Attributes:
        _campaigns: Cache of campaigns by ID.
        _results: Cache of phishing results by campaign_id.
        _tracking_tokens: Map of tracking tokens to (campaign_id, user_id).

    Example:
        >>> simulator = PhishingSimulator()
        >>> campaign = await simulator.create_campaign(
        ...     name="Q1 Test",
        ...     template_type=TemplateType.CREDENTIAL_HARVEST,
        ...     target_users=["u-1", "u-2"],
        ... )
        >>> await simulator.send_phishing_emails(campaign.id)
        >>> metrics = await simulator.get_campaign_metrics(campaign.id)
    """

    def __init__(self) -> None:
        """Initialize the phishing simulator."""
        self._config = get_config()

        # Caches (in production, backed by database)
        self._campaigns: Dict[str, PhishingCampaign] = {}
        self._results: Dict[str, List[PhishingResult]] = {}
        self._tracking_tokens: Dict[str, tuple[str, str]] = {}

        logger.info(
            "PhishingSimulator initialized (enabled=%s, cooldown=%dd)",
            self._config.phishing_enabled,
            self._config.phishing_cooldown_days,
        )

    @property
    def templates(self) -> Dict[TemplateType, Dict[str, Any]]:
        """Get available phishing templates."""
        return PHISHING_TEMPLATES

    async def create_campaign(
        self,
        name: str,
        template_type: TemplateType,
        target_users: List[str],
        target_roles: Optional[List[str]] = None,
        scheduled_at: Optional[datetime] = None,
        created_by: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PhishingCampaign:
        """Create a new phishing simulation campaign.

        Args:
            name: Campaign name for identification.
            template_type: Type of phishing template to use.
            target_users: List of target user IDs.
            target_roles: Optional list of roles to target.
            scheduled_at: Optional future send time.
            created_by: User ID of campaign creator.
            metadata: Optional additional metadata.

        Returns:
            Created PhishingCampaign.
        """
        if not self._config.phishing_enabled:
            logger.warning("Phishing campaigns are disabled in configuration")

        campaign = PhishingCampaign(
            name=name,
            template_type=template_type,
            status=CampaignStatus.SCHEDULED if scheduled_at else CampaignStatus.DRAFT,
            scheduled_at=scheduled_at,
            target_count=len(target_users),
            target_user_ids=target_users,
            target_roles=target_roles or [],
            created_by=created_by,
            metadata=metadata or {},
        )

        self._campaigns[campaign.id] = campaign
        self._results[campaign.id] = []

        logger.info(
            "Created phishing campaign %s (%s) targeting %d users",
            campaign.id,
            name,
            len(target_users),
        )

        return campaign

    async def generate_emails(
        self,
        campaign_id: str,
    ) -> List[Dict[str, Any]]:
        """Generate personalized phishing emails for a campaign.

        Args:
            campaign_id: Campaign identifier.

        Returns:
            List of email data dicts with recipient, subject, body, tracking info.

        Raises:
            ValueError: If campaign not found.
        """
        campaign = self._campaigns.get(campaign_id)
        if campaign is None:
            raise ValueError(f"Campaign {campaign_id} not found")

        template = PHISHING_TEMPLATES.get(campaign.template_type)
        if template is None:
            raise ValueError(f"Template {campaign.template_type} not found")

        emails: List[Dict[str, Any]] = []
        sender_domain = self._config.phishing_sender_domain

        for user_id in campaign.target_user_ids:
            # Generate tracking tokens
            click_token = self._generate_tracking_token(campaign_id, user_id, "click")
            pixel_token = self._generate_tracking_token(campaign_id, user_id, "pixel")

            # Build URLs
            tracking_url = f"https://{sender_domain}/t/{click_token}"
            tracking_pixel = f"https://{sender_domain}/p/{pixel_token}"

            # Personalize template
            body = template["body_html"].format(
                user_name=f"User {user_id[-4:]}",  # In prod, lookup actual name
                tracking_url=tracking_url,
                tracking_pixel=tracking_pixel,
                invoice_number=f"{random_int(10000, 99999)}",
                amount=f"{random_int(500, 5000):,.00f}",
                ceo_name="John Smith",
                ceo_email_prefix="jsmith",
            )

            sender = f"{template['sender_prefix']}@{sender_domain}"
            subject = template["subject"].format(
                invoice_number=f"{random_int(10000, 99999)}"
            )

            emails.append({
                "recipient_id": user_id,
                "sender": sender,
                "sender_name": template["sender_name"],
                "subject": subject,
                "body_html": body,
                "click_token": click_token,
                "pixel_token": pixel_token,
                "tracking_url": tracking_url,
                "tracking_pixel": tracking_pixel,
            })

        logger.info(
            "Generated %d emails for campaign %s",
            len(emails),
            campaign_id,
        )

        return emails

    async def send_phishing_emails(
        self,
        campaign_id: str,
    ) -> int:
        """Send phishing simulation emails via SES.

        In production, this would integrate with AWS SES or similar.
        For now, it simulates sending and creates result records.

        Args:
            campaign_id: Campaign identifier.

        Returns:
            Number of emails sent.

        Raises:
            ValueError: If campaign not found or not in valid state.
        """
        campaign = self._campaigns.get(campaign_id)
        if campaign is None:
            raise ValueError(f"Campaign {campaign_id} not found")

        if campaign.status not in (CampaignStatus.DRAFT, CampaignStatus.SCHEDULED):
            raise ValueError(
                f"Campaign {campaign_id} is in {campaign.status} state, cannot send"
            )

        if not self._config.phishing_enabled:
            raise ValueError("Phishing campaigns are disabled in configuration")

        # Generate emails
        emails = await self.generate_emails(campaign_id)

        # Create result records for each recipient
        sent_at = datetime.now(timezone.utc)
        for email in emails:
            result = PhishingResult(
                campaign_id=campaign_id,
                user_id=email["recipient_id"],
                sent_at=sent_at,
            )
            self._results[campaign_id].append(result)

        # Update campaign status
        campaign.status = CampaignStatus.RUNNING
        campaign.sent_at = sent_at

        logger.info(
            "Sent %d phishing emails for campaign %s",
            len(emails),
            campaign_id,
        )

        return len(emails)

    async def track_open(
        self,
        campaign_id: str,
        user_id: str,
    ) -> bool:
        """Track email open via tracking pixel.

        Args:
            campaign_id: Campaign identifier.
            user_id: User who opened the email.

        Returns:
            True if tracked successfully.
        """
        results = self._results.get(campaign_id, [])
        for result in results:
            if result.user_id == user_id and result.opened_at is None:
                result.opened_at = datetime.now(timezone.utc)
                logger.info(
                    "Tracked email open: campaign=%s user=%s",
                    campaign_id,
                    user_id,
                )
                return True
        return False

    async def track_click(
        self,
        campaign_id: str,
        user_id: str,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> bool:
        """Track link click.

        Args:
            campaign_id: Campaign identifier.
            user_id: User who clicked.
            user_agent: Browser user agent.
            ip_address: Client IP address.

        Returns:
            True if tracked successfully.
        """
        results = self._results.get(campaign_id, [])
        for result in results:
            if result.user_id == user_id and result.clicked_at is None:
                result.clicked_at = datetime.now(timezone.utc)
                result.user_agent = user_agent
                result.ip_address = ip_address

                logger.warning(
                    "User clicked phishing link: campaign=%s user=%s",
                    campaign_id,
                    user_id,
                )

                # Auto-enroll in training if configured
                if self._config.phishing_auto_enroll_on_click:
                    await self.trigger_training(user_id)

                return True
        return False

    async def track_credential_entry(
        self,
        campaign_id: str,
        user_id: str,
    ) -> bool:
        """Track credential submission (most severe failure).

        Args:
            campaign_id: Campaign identifier.
            user_id: User who entered credentials.

        Returns:
            True if tracked successfully.
        """
        results = self._results.get(campaign_id, [])
        for result in results:
            if result.user_id == user_id:
                result.credentials_entered = True

                logger.error(
                    "User submitted credentials on phishing page: campaign=%s user=%s",
                    campaign_id,
                    user_id,
                )

                # Always trigger training for credential submission
                await self.trigger_training(user_id)
                return True
        return False

    async def track_report(
        self,
        campaign_id: str,
        user_id: str,
    ) -> bool:
        """Track user reporting email as phishing (positive behavior).

        Args:
            campaign_id: Campaign identifier.
            user_id: User who reported the phishing.

        Returns:
            True if tracked successfully.
        """
        results = self._results.get(campaign_id, [])
        for result in results:
            if result.user_id == user_id and result.reported_at is None:
                result.reported_at = datetime.now(timezone.utc)

                logger.info(
                    "User correctly reported phishing: campaign=%s user=%s",
                    campaign_id,
                    user_id,
                )
                return True
        return False

    async def get_campaign_metrics(
        self,
        campaign_id: str,
    ) -> CampaignMetrics:
        """Get aggregated metrics for a campaign.

        Args:
            campaign_id: Campaign identifier.

        Returns:
            CampaignMetrics with all statistics.

        Raises:
            ValueError: If campaign not found.
        """
        campaign = self._campaigns.get(campaign_id)
        if campaign is None:
            raise ValueError(f"Campaign {campaign_id} not found")

        results = self._results.get(campaign_id, [])

        total_sent = len(results)
        total_opened = sum(1 for r in results if r.opened_at is not None)
        total_clicked = sum(1 for r in results if r.clicked_at is not None)
        total_credentials = sum(1 for r in results if r.credentials_entered)
        total_reported = sum(1 for r in results if r.reported_at is not None)

        return CampaignMetrics(
            campaign_id=campaign_id,
            total_sent=total_sent,
            total_opened=total_opened,
            total_clicked=total_clicked,
            total_credentials=total_credentials,
            total_reported=total_reported,
            open_rate=total_opened / total_sent if total_sent > 0 else 0.0,
            click_rate=total_clicked / total_sent if total_sent > 0 else 0.0,
            credential_rate=total_credentials / total_sent if total_sent > 0 else 0.0,
            report_rate=total_reported / total_sent if total_sent > 0 else 0.0,
        )

    async def trigger_training(
        self,
        user_id: str,
    ) -> bool:
        """Auto-enroll user who failed phishing test in training.

        Args:
            user_id: User to enroll.

        Returns:
            True if enrollment triggered.
        """
        # In production, this would create a training enrollment
        logger.info(
            "Auto-enrolling user %s in phishing_recognition course",
            user_id,
        )
        return True

    async def generate_campaign_report(
        self,
        campaign_id: str,
    ) -> Dict[str, Any]:
        """Generate comprehensive campaign report.

        Args:
            campaign_id: Campaign identifier.

        Returns:
            Report data dictionary suitable for PDF generation.

        Raises:
            ValueError: If campaign not found.
        """
        campaign = self._campaigns.get(campaign_id)
        if campaign is None:
            raise ValueError(f"Campaign {campaign_id} not found")

        metrics = await self.get_campaign_metrics(campaign_id)
        results = self._results.get(campaign_id, [])
        template = PHISHING_TEMPLATES.get(campaign.template_type, {})

        # Build detailed results
        user_results = []
        for result in results:
            user_results.append({
                "user_id": result.user_id,
                "opened": result.opened_at is not None,
                "clicked": result.clicked_at is not None,
                "credentials_entered": result.credentials_entered,
                "reported": result.reported_at is not None,
                "outcome": self._determine_outcome(result),
            })

        return {
            "report_id": f"RPT-{campaign_id[:8]}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "campaign": {
                "id": campaign.id,
                "name": campaign.name,
                "template_type": campaign.template_type.value,
                "status": campaign.status.value,
                "created_at": campaign.created_at.isoformat(),
                "sent_at": campaign.sent_at.isoformat() if campaign.sent_at else None,
            },
            "metrics": {
                "total_sent": metrics.total_sent,
                "total_opened": metrics.total_opened,
                "total_clicked": metrics.total_clicked,
                "total_credentials": metrics.total_credentials,
                "total_reported": metrics.total_reported,
                "open_rate_pct": round(metrics.open_rate * 100, 1),
                "click_rate_pct": round(metrics.click_rate * 100, 1),
                "credential_rate_pct": round(metrics.credential_rate * 100, 1),
                "report_rate_pct": round(metrics.report_rate * 100, 1),
            },
            "target_click_rate": self._config.phishing_click_threshold * 100,
            "click_rate_status": (
                "PASS" if metrics.click_rate <= self._config.phishing_click_threshold
                else "FAIL"
            ),
            "template_indicators": template.get("indicators", []),
            "user_results": user_results,
            "recommendations": self._generate_recommendations(metrics),
        }

    async def get_campaign(
        self,
        campaign_id: str,
    ) -> Optional[PhishingCampaign]:
        """Get a campaign by ID.

        Args:
            campaign_id: Campaign identifier.

        Returns:
            PhishingCampaign if found, None otherwise.
        """
        return self._campaigns.get(campaign_id)

    async def list_campaigns(
        self,
        status_filter: Optional[CampaignStatus] = None,
        limit: int = 50,
    ) -> List[PhishingCampaign]:
        """List all campaigns with optional status filter.

        Args:
            status_filter: Optional status to filter by.
            limit: Maximum campaigns to return.

        Returns:
            List of PhishingCampaign objects.
        """
        campaigns = list(self._campaigns.values())

        if status_filter:
            campaigns = [c for c in campaigns if c.status == status_filter]

        # Sort by created_at descending
        campaigns.sort(key=lambda c: c.created_at, reverse=True)

        return campaigns[:limit]

    async def update_campaign(
        self,
        campaign_id: str,
        updates: Dict[str, Any],
    ) -> Optional[PhishingCampaign]:
        """Update a campaign.

        Args:
            campaign_id: Campaign identifier.
            updates: Fields to update.

        Returns:
            Updated PhishingCampaign or None if not found.
        """
        campaign = self._campaigns.get(campaign_id)
        if campaign is None:
            return None

        # Only allow updates if campaign is in draft state
        if campaign.status != CampaignStatus.DRAFT:
            raise ValueError(
                f"Cannot update campaign in {campaign.status} state"
            )

        for key, value in updates.items():
            if hasattr(campaign, key):
                setattr(campaign, key, value)

        return campaign

    async def complete_campaign(
        self,
        campaign_id: str,
    ) -> Optional[PhishingCampaign]:
        """Mark a campaign as completed.

        Args:
            campaign_id: Campaign identifier.

        Returns:
            Updated campaign or None if not found.
        """
        campaign = self._campaigns.get(campaign_id)
        if campaign is None:
            return None

        campaign.status = CampaignStatus.COMPLETED
        campaign.completed_at = datetime.now(timezone.utc)

        logger.info("Completed phishing campaign %s", campaign_id)
        return campaign

    def _generate_tracking_token(
        self,
        campaign_id: str,
        user_id: str,
        token_type: str,
    ) -> str:
        """Generate a unique tracking token.

        Args:
            campaign_id: Campaign identifier.
            user_id: User identifier.
            token_type: Type of token (click, pixel).

        Returns:
            Tracking token string.
        """
        unique_string = f"{campaign_id}:{user_id}:{token_type}:{uuid.uuid4()}"
        token = hashlib.sha256(unique_string.encode()).hexdigest()[:24]

        # Store mapping
        self._tracking_tokens[token] = (campaign_id, user_id)

        return token

    def _determine_outcome(self, result: PhishingResult) -> str:
        """Determine outcome classification for a result.

        Args:
            result: Phishing result record.

        Returns:
            Outcome classification string.
        """
        if result.reported_at is not None:
            return "REPORTED_PHISHING"  # Best outcome
        if result.credentials_entered:
            return "CREDENTIALS_SUBMITTED"  # Worst outcome
        if result.clicked_at is not None:
            return "CLICKED_LINK"  # Bad outcome
        if result.opened_at is not None:
            return "OPENED_EMAIL"  # Neutral
        return "NO_INTERACTION"

    def _generate_recommendations(
        self,
        metrics: CampaignMetrics,
    ) -> List[str]:
        """Generate training recommendations based on metrics.

        Args:
            metrics: Campaign metrics.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if metrics.click_rate > 0.1:
            recommendations.append(
                "High click rate detected. Conduct mandatory phishing awareness training."
            )
        if metrics.credential_rate > 0.05:
            recommendations.append(
                "Critical: Users submitted credentials. Implement immediate security briefing."
            )
        if metrics.report_rate < 0.2:
            recommendations.append(
                "Low phishing report rate. Train users on how and when to report suspicious emails."
            )
        if metrics.open_rate > 0.8:
            recommendations.append(
                "High open rate indicates template was effective. Consider testing with similar templates."
            )

        if not recommendations:
            recommendations.append(
                "Results within acceptable range. Continue regular awareness training."
            )

        return recommendations


def random_int(min_val: int, max_val: int) -> int:
    """Generate a random integer (helper for template personalization)."""
    import random
    return random.randint(min_val, max_val)


__all__ = [
    "PHISHING_TEMPLATES",
    "PhishingSimulator",
]
