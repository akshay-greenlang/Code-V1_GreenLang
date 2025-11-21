# -*- coding: utf-8 -*-
"""
Email scheduler for campaign automation.

Schedules and manages multi-touch email sequences with consent checking.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid

from ..models import (
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock
    Campaign,
    EmailMessage,
    EmailStatus,
    EmailSequence
)
from ..consent import ConsentRegistry
from ..templates import get_template, render_template
from ..exceptions import ConsentNotGrantedError


logger = logging.getLogger(__name__)


class EmailScheduler:
    """
    Schedules and manages email sends for campaigns.

    Features:
    - Multi-touch sequence scheduling
    - Consent-aware sending
    - Batch processing
    - Retry logic
    - Delivery tracking
    """

    def __init__(self, consent_registry: ConsentRegistry):
        """
        Initialize email scheduler.

        Args:
            consent_registry: Consent registry for compliance checks
        """
        self.consent_registry = consent_registry
        self.scheduled_messages: Dict[str, EmailMessage] = {}
        logger.info("EmailScheduler initialized")

    def schedule_campaign_emails(
        self,
        campaign: Campaign,
        personalization_base: Dict[str, str]
    ) -> List[EmailMessage]:
        """
        Schedule all emails for campaign based on email sequence.

        Args:
            campaign: Campaign to schedule
            personalization_base: Base personalization data (common to all emails)

        Returns:
            List of scheduled email messages
        """
        scheduled = []

        for supplier_id in campaign.target_suppliers:
            # Check consent before scheduling
            try:
                self.consent_registry.enforce_consent(supplier_id)
            except (ConsentNotGrantedError, Exception) as e:
                logger.warning(
                    f"Skipping supplier {supplier_id} in campaign {campaign.campaign_id}: {e}"
                )
                continue

            # Get supplier details
            consent_record = self.consent_registry.get_record(supplier_id)
            if not consent_record:
                continue

            # Schedule each touch in the sequence
            for touch in campaign.email_sequence.touches:
                message = self._schedule_touch(
                    campaign=campaign,
                    supplier_id=supplier_id,
                    email_address=consent_record.email_address,
                    touch=touch,
                    personalization_base=personalization_base
                )
                scheduled.append(message)

        logger.info(
            f"Scheduled {len(scheduled)} emails for campaign {campaign.campaign_id}"
        )

        return scheduled

    def _schedule_touch(
        self,
        campaign: Campaign,
        supplier_id: str,
        email_address: str,
        touch: Dict,
        personalization_base: Dict[str, str]
    ) -> EmailMessage:
        """
        Schedule individual touch email.

        Args:
            campaign: Campaign
            supplier_id: Supplier ID
            email_address: Supplier email
            touch: Touch configuration
            personalization_base: Base personalization data

        Returns:
            Scheduled email message
        """
        # Calculate send time
        send_time = campaign.start_date + timedelta(days=touch['day_offset'])

        # Get template
        template_id = touch['template']
        template = get_template(template_id)

        # Prepare personalization data
        personalization_data = personalization_base.copy()
        personalization_data['supplier_id'] = supplier_id
        personalization_data['email_address'] = email_address

        # Render template
        rendered = render_template(template, personalization_data)

        # Generate message ID
        message_id = f"msg_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:16]}"

        # Create email message
        message = EmailMessage(
            message_id=message_id,
            campaign_id=campaign.campaign_id,
            supplier_id=supplier_id,
            to_email=email_address,
            subject=rendered['subject'],
            body_html=rendered['body_html'],
            body_text=rendered['body_text'],
            status=EmailStatus.PENDING,
            scheduled_send=send_time,
            unsubscribe_url=personalization_data.get(
                'unsubscribe_url',
                f"https://portal.company.com/unsubscribe?supplier={supplier_id}"
            ),
            tracking_metadata={
                "touch_number": touch.get('touch_number', 0),
                "template_id": template_id,
            }
        )

        self.scheduled_messages[message_id] = message

        logger.debug(
            f"Scheduled email {message_id} for supplier {supplier_id} "
            f"at {send_time.isoformat()}"
        )

        return message

    def get_due_messages(
        self,
        before: Optional[datetime] = None
    ) -> List[EmailMessage]:
        """
        Get messages that are due to be sent.

        Args:
            before: Get messages due before this time (default: now)

        Returns:
            List of due messages
        """
        cutoff = before or DeterministicClock.utcnow()

        due_messages = [
            msg for msg in self.scheduled_messages.values()
            if msg.status == EmailStatus.PENDING and msg.scheduled_send <= cutoff
        ]

        return due_messages

    def mark_sent(
        self,
        message_id: str,
        sent_at: Optional[datetime] = None
    ):
        """
        Mark message as sent.

        Args:
            message_id: Message identifier
            sent_at: Timestamp of send (default: now)
        """
        if message_id in self.scheduled_messages:
            message = self.scheduled_messages[message_id]
            message.status = EmailStatus.SENT
            message.sent_at = sent_at or DeterministicClock.utcnow()
            logger.debug(f"Marked message {message_id} as sent")

    def mark_delivered(
        self,
        message_id: str,
        delivered_at: Optional[datetime] = None
    ):
        """
        Mark message as delivered.

        Args:
            message_id: Message identifier
            delivered_at: Timestamp of delivery (default: now)
        """
        if message_id in self.scheduled_messages:
            message = self.scheduled_messages[message_id]
            message.status = EmailStatus.DELIVERED
            message.delivered_at = delivered_at or DeterministicClock.utcnow()
            logger.debug(f"Marked message {message_id} as delivered")

    def mark_opened(
        self,
        message_id: str,
        opened_at: Optional[datetime] = None
    ):
        """
        Mark message as opened.

        Args:
            message_id: Message identifier
            opened_at: Timestamp of open (default: now)
        """
        if message_id in self.scheduled_messages:
            message = self.scheduled_messages[message_id]
            message.status = EmailStatus.OPENED
            message.opened_at = opened_at or DeterministicClock.utcnow()
            logger.debug(f"Marked message {message_id} as opened")

    def mark_clicked(
        self,
        message_id: str,
        clicked_at: Optional[datetime] = None
    ):
        """
        Mark message as clicked.

        Args:
            message_id: Message identifier
            clicked_at: Timestamp of click (default: now)
        """
        if message_id in self.scheduled_messages:
            message = self.scheduled_messages[message_id]
            message.status = EmailStatus.CLICKED
            message.clicked_at = clicked_at or DeterministicClock.utcnow()
            logger.debug(f"Marked message {message_id} as clicked")

    def mark_failed(
        self,
        message_id: str,
        error_message: str
    ):
        """
        Mark message as failed.

        Args:
            message_id: Message identifier
            error_message: Error description
        """
        if message_id in self.scheduled_messages:
            message = self.scheduled_messages[message_id]
            message.status = EmailStatus.FAILED
            message.tracking_metadata['error'] = error_message
            logger.warning(f"Message {message_id} failed: {error_message}")

    def get_campaign_messages(
        self,
        campaign_id: str
    ) -> List[EmailMessage]:
        """
        Get all messages for campaign.

        Args:
            campaign_id: Campaign identifier

        Returns:
            List of messages
        """
        return [
            msg for msg in self.scheduled_messages.values()
            if msg.campaign_id == campaign_id
        ]

    def get_supplier_messages(
        self,
        supplier_id: str
    ) -> List[EmailMessage]:
        """
        Get all messages for supplier.

        Args:
            supplier_id: Supplier identifier

        Returns:
            List of messages
        """
        return [
            msg for msg in self.scheduled_messages.values()
            if msg.supplier_id == supplier_id
        ]
