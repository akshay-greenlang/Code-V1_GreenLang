"""
SupplierEngagementAgent - Consent-aware supplier engagement and data collection.

Main agent orchestrating consent management, campaigns, portal, and email delivery.

Version: 2.0.0 - Enhanced with GreenLang SDK
Phase: 5 (Agent Architecture Compliance)
Date: 2025-11-09
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# GreenLang SDK Integration
from greenlang.sdk.base import Agent, Metadata, Result
from greenlang.cache import CacheManager, get_cache_manager
from greenlang.telemetry import (
    MetricsCollector,
    get_logger,
    track_execution,
    create_span,
)

from .models import (
    Campaign,
    CampaignAnalytics,
    ConsentRecord,
    ConsentStatus,
    EmailSequence,
    EmailTemplate,
    EmailResult,
    ValidationResult,
    SupplierProgress,
    Leaderboard
)
from .consent import ConsentRegistry, OptOutHandler
from .campaigns import CampaignManager, EmailScheduler, CampaignAnalytics as CampaignAnalyticsEngine
from .portal import PortalAuthenticator, UploadHandler, LiveValidator, GamificationEngine
from .templates import get_template, render_template
from .integrations import SendGridService, MailgunService, AWSSESService
from .config import (
    EMAIL_SERVICE_CONFIG,
    DEFAULT_EMAIL_SEQUENCE,
    get_email_service_config
)
from .exceptions import (
    ConsentNotGrantedError,
    CampaignNotFoundError,
    SupplierNotFoundError
)


logger = get_logger(__name__)


class SupplierEngagementAgent(Agent[Dict[str, Any], Dict[str, Any]]):
    """
    Consent-aware supplier engagement and data collection agent.

    Features:
    - GDPR/CCPA/CAN-SPAM compliant consent management
    - Multi-touch email campaigns
    - Supplier portal (upload, validation, progress)
    - Gamification (leaderboard, badges)
    - Campaign analytics

    This is the main orchestrator for Phase 3 supplier engagement.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Supplier Engagement Agent.

        Args:
            config: Optional configuration dictionary
        """
        # Initialize base Agent with metadata
        metadata = Metadata(
            id="supplier_engagement_agent",
            name="SupplierEngagementAgent",
            version="2.0.0",
            description="Consent-aware supplier engagement and data collection agent",
            tags=["engagement", "supplier", "consent", "gdpr", "campaigns"],
        )
        super().__init__(metadata)

        self.config = config or {}

        # Initialize GreenLang infrastructure
        self.cache_manager = get_cache_manager()
        self.metrics = MetricsCollector(namespace="vcci.engagement")

        # Initialize core components
        self.consent_registry = ConsentRegistry(
            storage_path=self.config.get("consent_storage", "data/consent_registry.json")
        )
        self.opt_out_handler = OptOutHandler(self.consent_registry)

        self.campaign_manager = CampaignManager(
            storage_path=self.config.get("campaign_storage", "data/campaigns.json")
        )
        self.email_scheduler = EmailScheduler(self.consent_registry)
        self.analytics_engine = CampaignAnalyticsEngine()

        self.portal_auth = PortalAuthenticator(
            session_duration_hours=self.config.get("session_duration_hours", 24)
        )
        self.upload_handler = UploadHandler(
            max_file_size_mb=self.config.get("max_file_size_mb", 50)
        )
        self.live_validator = LiveValidator()
        self.gamification = GamificationEngine()

        # Initialize email service
        email_provider = self.config.get("email_provider", EMAIL_SERVICE_CONFIG["default_provider"])
        self.email_service = self._init_email_service(email_provider)

        logger.info("SupplierEngagementAgent v2.0 initialized")

    def validate(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data for engagement operations.

        Args:
            input_data: Input data containing operation and parameters

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, dict):
            logger.error("Input data must be a dictionary")
            return False

        if "operation" not in input_data:
            logger.error("Input data must contain 'operation' field")
            return False

        operation = input_data.get("operation")
        valid_operations = ["create_campaign", "send_email", "validate_upload", "get_analytics"]

        if operation not in valid_operations:
            logger.error(f"Invalid operation: {operation}. Must be one of {valid_operations}")
            return False

        return True

    @track_execution(metric_name="engagement_process")
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process engagement operation.

        Args:
            input_data: Dictionary with operation and parameters

        Returns:
            Dictionary with operation results
        """
        operation = input_data["operation"]
        params = input_data.get("params", {})

        with create_span(name="engagement_operation", attributes={"operation": operation}):
            if operation == "create_campaign":
                campaign = self.create_campaign(**params)
                result = {"campaign_id": campaign.campaign_id, "status": "created"}
            elif operation == "send_email":
                email_result = self.send_email(**params)
                result = {"success": email_result.success, "message_id": email_result.message_id}
            elif operation == "validate_upload":
                validation = self.validate_upload(**params)
                result = {"valid": validation.is_valid, "errors": validation.errors}
            elif operation == "get_analytics":
                analytics = self.get_campaign_analytics(**params)
                result = analytics.__dict__
            else:
                raise ValueError(f"Unknown operation: {operation}")

        # Record metrics
        if self.metrics:
            self.metrics.record_metric(
                f"engagement.{operation}",
                1,
                unit="count"
            )

        return result

    def _init_email_service(self, provider: str):
        """Initialize email service provider."""
        if provider == "sendgrid":
            return SendGridService()
        elif provider == "mailgun":
            return MailgunService()
        elif provider == "aws_ses":
            return AWSSESService()
        else:
            logger.warning(f"Unknown email provider {provider}, defaulting to SendGrid")
            return SendGridService()

    # ========== Consent Management ==========

    def register_supplier(
        self,
        supplier_id: str,
        email_address: str,
        country: str,
        auto_opt_in: bool = False
    ) -> ConsentRecord:
        """
        Register supplier in consent registry.

        Args:
            supplier_id: Unique supplier identifier
            email_address: Contact email
            country: Supplier country (ISO 3166-1)
            auto_opt_in: Auto opt-in for existing business relationships

        Returns:
            Consent record
        """
        return self.consent_registry.register_supplier(
            supplier_id=supplier_id,
            email_address=email_address,
            country=country,
            auto_opt_in=auto_opt_in
        )

    def check_consent(self, supplier_id: str) -> bool:
        """
        Check if supplier has valid consent for contact.

        Args:
            supplier_id: Supplier identifier

        Returns:
            True if contact is allowed, False otherwise
        """
        return self.consent_registry.check_consent(supplier_id)

    def register_opt_out(
        self,
        supplier_id: str,
        reason: Optional[str] = None
    ):
        """
        Register supplier opt-out (GDPR/CCPA/CAN-SPAM compliance).

        Args:
            supplier_id: Supplier identifier
            reason: Reason for opt-out
        """
        self.opt_out_handler.process_opt_out(supplier_id, reason)

    # ========== Campaign Management ==========

    def create_campaign(
        self,
        name: str,
        target_suppliers: List[str],
        email_sequence: Optional[EmailSequence] = None,
        start_date: Optional[datetime] = None,
        duration_days: int = 90,
        response_rate_target: float = 0.50
    ) -> Campaign:
        """
        Create and launch supplier engagement campaign.

        Args:
            name: Campaign name
            target_suppliers: List of supplier IDs to target
            email_sequence: Email sequence (default: 4-touch sequence)
            start_date: Campaign start date (default: now)
            duration_days: Campaign duration
            response_rate_target: Target response rate (0-1)

        Returns:
            Created campaign
        """
        # Use default sequence if not provided
        if not email_sequence:
            email_sequence = EmailSequence(**DEFAULT_EMAIL_SEQUENCE)

        campaign = self.campaign_manager.create_campaign(
            name=name,
            target_suppliers=target_suppliers,
            email_sequence=email_sequence,
            start_date=start_date,
            duration_days=duration_days,
            response_rate_target=response_rate_target
        )

        logger.info(f"Created campaign {campaign.campaign_id}: {name}")

        return campaign

    def start_campaign(
        self,
        campaign_id: str,
        personalization_base: Optional[Dict[str, str]] = None
    ) -> Campaign:
        """
        Start campaign and schedule emails.

        Args:
            campaign_id: Campaign identifier
            personalization_base: Base personalization data

        Returns:
            Started campaign
        """
        # Start campaign
        campaign = self.campaign_manager.start_campaign(campaign_id)

        # Schedule emails
        personalization_base = personalization_base or {
            "company_name": "Your Company",
            "sender_name": "Sustainability Team",
            "support_email": "sustainability@company.com",
            "portal_url": "https://portal.company.com",
            "privacy_policy_url": "https://company.com/privacy",
            "company_address": "123 Main St, City, Country"
        }

        scheduled_messages = self.email_scheduler.schedule_campaign_emails(
            campaign=campaign,
            personalization_base=personalization_base
        )

        logger.info(
            f"Started campaign {campaign_id} with {len(scheduled_messages)} scheduled emails"
        )

        return campaign

    def send_email(
        self,
        supplier_id: str,
        template: EmailTemplate,
        personalization_data: Dict[str, str]
    ) -> EmailResult:
        """
        Send email to supplier (consent-aware).

        Args:
            supplier_id: Supplier identifier
            template: Email template
            personalization_data: Personalization data

        Returns:
            Email result

        Raises:
            ConsentNotGrantedError: If consent not granted
        """
        # Enforce consent
        self.consent_registry.enforce_consent(supplier_id)

        # Get supplier details
        consent_record = self.consent_registry.get_record(supplier_id)
        if not consent_record:
            raise SupplierNotFoundError(supplier_id)

        # Generate unsubscribe URL
        unsubscribe_url = self.opt_out_handler.generate_unsubscribe_url(
            supplier_id=supplier_id,
            campaign_id="manual",
            base_url=personalization_data.get("portal_url", "https://portal.company.com")
        )
        personalization_data["unsubscribe_url"] = unsubscribe_url

        # Render template
        rendered = render_template(template, personalization_data)

        # Create email message (simplified for manual send)
        from .models import EmailMessage, EmailStatus
        import uuid

        message = EmailMessage(
            message_id=f"msg_{uuid.uuid4().hex[:16]}",
            campaign_id="manual",
            supplier_id=supplier_id,
            to_email=consent_record.email_address,
            subject=rendered["subject"],
            body_html=rendered["body_html"],
            body_text=rendered["body_text"],
            status=EmailStatus.PENDING,
            scheduled_send=datetime.utcnow(),
            unsubscribe_url=unsubscribe_url
        )

        # Send via email service
        result = self.email_service.send_email(message)

        # Record contact
        if result.success:
            self.consent_registry.record_contact(supplier_id)

        return result

    def get_campaign_analytics(self, campaign_id: str) -> CampaignAnalytics:
        """
        Get campaign performance metrics.

        Args:
            campaign_id: Campaign identifier

        Returns:
            Campaign analytics

        Raises:
            CampaignNotFoundError: If campaign not found
        """
        campaign = self.campaign_manager.get_campaign(campaign_id)
        messages = self.email_scheduler.get_campaign_messages(campaign_id)

        analytics = self.analytics_engine.generate_analytics(
            campaign=campaign,
            messages=messages
        )

        return analytics

    # ========== Portal & Validation ==========

    def validate_upload(
        self,
        supplier_id: str,
        data: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate supplier data upload in real-time.

        Args:
            supplier_id: Supplier identifier
            data: Data to validate

        Returns:
            Validation result
        """
        return self.live_validator.validate_record(data)

    def generate_magic_link(
        self,
        supplier_id: str,
        email: str
    ) -> str:
        """
        Generate magic link for passwordless portal login.

        Args:
            supplier_id: Supplier identifier
            email: Supplier email

        Returns:
            Magic link URL
        """
        return self.portal_auth.generate_magic_link(supplier_id, email)

    # ========== Gamification ==========

    def get_leaderboard(
        self,
        campaign_id: str,
        top_n: int = 10
    ) -> Leaderboard:
        """
        Get supplier leaderboard for gamification.

        Args:
            campaign_id: Campaign identifier
            top_n: Number of top suppliers

        Returns:
            Leaderboard
        """
        return self.gamification.generate_leaderboard(campaign_id, top_n)

    def track_supplier_progress(
        self,
        supplier_id: str,
        campaign_id: str,
        completion_percentage: float,
        data_quality_score: Optional[float] = None
    ) -> SupplierProgress:
        """
        Track supplier progress and award badges.

        Args:
            supplier_id: Supplier identifier
            campaign_id: Campaign identifier
            completion_percentage: Progress percentage
            data_quality_score: DQI score

        Returns:
            Supplier progress
        """
        progress = self.gamification.track_progress(
            supplier_id=supplier_id,
            campaign_id=campaign_id,
            completion_percentage=completion_percentage,
            data_quality_score=data_quality_score
        )

        # Check and award badges
        self.gamification.check_and_award_badges(supplier_id, campaign_id)

        return progress

    # ========== Reporting & Statistics ==========

    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get overall agent statistics.

        Returns:
            Agent statistics
        """
        consent_stats = self.consent_registry.get_statistics()
        opt_out_stats = self.opt_out_handler.get_opt_out_statistics()

        campaigns = self.campaign_manager.list_campaigns()
        active_campaigns = self.campaign_manager.get_active_campaigns()

        return {
            "consent": consent_stats,
            "opt_outs": opt_out_stats,
            "campaigns": {
                "total": len(campaigns),
                "active": len(active_campaigns)
            },
            "agent_version": "1.0",
            "timestamp": datetime.utcnow().isoformat()
        }
