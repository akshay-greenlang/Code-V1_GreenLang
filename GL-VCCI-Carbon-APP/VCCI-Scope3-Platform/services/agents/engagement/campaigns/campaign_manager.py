# -*- coding: utf-8 -*-
"""
Campaign manager for supplier engagement campaigns.

Handles campaign creation, targeting, lifecycle management, and metrics tracking.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
import uuid

from ..models import (
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock
    Campaign,
    CampaignStatus,
    EmailSequence,
    EmailTemplate,
    CampaignAnalytics
)
from ..exceptions import (
    CampaignNotFoundError,
    CampaignAlreadyActiveError,
    InvalidEmailSequenceError
)
from ..config import CAMPAIGN_CONFIG


logger = logging.getLogger(__name__)


class CampaignManager:
    """
    Manages supplier engagement campaigns with targeting and analytics.

    Features:
    - Campaign creation and configuration
    - Target supplier selection (e.g., top 20% spend)
    - Email sequence management
    - Campaign lifecycle (draft, active, paused, completed)
    - Performance tracking
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize campaign manager.

        Args:
            storage_path: Path to campaign storage
        """
        self.storage_path = storage_path or "data/campaigns.json"
        self.campaigns: Dict[str, Campaign] = {}
        self._load_campaigns()
        logger.info("CampaignManager initialized")

    def _load_campaigns(self):
        """Load campaigns from storage."""
        path = Path(self.storage_path)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    for campaign_id, campaign_data in data.items():
                        # Convert datetime strings
                        campaign_data['start_date'] = datetime.fromisoformat(
                            campaign_data['start_date']
                        )
                        campaign_data['end_date'] = datetime.fromisoformat(
                            campaign_data['end_date']
                        )
                        campaign_data['created_at'] = datetime.fromisoformat(
                            campaign_data['created_at']
                        )
                        campaign_data['updated_at'] = datetime.fromisoformat(
                            campaign_data['updated_at']
                        )

                        self.campaigns[campaign_id] = Campaign(**campaign_data)
                logger.info(f"Loaded {len(self.campaigns)} campaigns")
            except Exception as e:
                logger.error(f"Failed to load campaigns: {e}")
                self.campaigns = {}
        else:
            logger.info("No existing campaigns found")
            path.parent.mkdir(parents=True, exist_ok=True)

    def _save_campaigns(self):
        """Save campaigns to storage."""
        try:
            path = Path(self.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {}
            for campaign_id, campaign in self.campaigns.items():
                campaign_dict = campaign.model_dump()
                # Convert datetime objects to ISO strings
                campaign_dict['start_date'] = campaign_dict['start_date'].isoformat()
                campaign_dict['end_date'] = campaign_dict['end_date'].isoformat()
                campaign_dict['created_at'] = campaign_dict['created_at'].isoformat()
                campaign_dict['updated_at'] = campaign_dict['updated_at'].isoformat()
                data[campaign_id] = campaign_dict

            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.campaigns)} campaigns")
        except Exception as e:
            logger.error(f"Failed to save campaigns: {e}")

    def create_campaign(
        self,
        name: str,
        target_suppliers: List[str],
        email_sequence: EmailSequence,
        start_date: Optional[datetime] = None,
        duration_days: int = 90,
        response_rate_target: float = 0.50
    ) -> Campaign:
        """
        Create new supplier engagement campaign.

        Args:
            name: Campaign name
            target_suppliers: List of supplier IDs to target
            email_sequence: Email sequence configuration
            start_date: Campaign start date (default: now)
            duration_days: Campaign duration in days
            response_rate_target: Target response rate (0.0-1.0)

        Returns:
            Created campaign

        Raises:
            InvalidEmailSequenceError: If email sequence is invalid
        """
        # Validate email sequence
        self._validate_email_sequence(email_sequence)

        # Generate campaign ID
        campaign_id = f"camp_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:12]}"

        # Set dates
        start = start_date or DeterministicClock.utcnow()
        end = start + timedelta(days=duration_days)

        campaign = Campaign(
            campaign_id=campaign_id,
            name=name,
            target_suppliers=target_suppliers,
            email_sequence=email_sequence,
            start_date=start,
            end_date=end,
            status=CampaignStatus.DRAFT,
            response_rate_target=response_rate_target
        )

        self.campaigns[campaign_id] = campaign
        self._save_campaigns()

        logger.info(
            f"Created campaign {campaign_id}: {name} "
            f"targeting {len(target_suppliers)} suppliers"
        )

        return campaign

    def _validate_email_sequence(self, email_sequence: EmailSequence):
        """
        Validate email sequence configuration.

        Args:
            email_sequence: Email sequence to validate

        Raises:
            InvalidEmailSequenceError: If sequence is invalid
        """
        if not email_sequence.touches:
            raise InvalidEmailSequenceError("Email sequence must have at least one touch")

        if len(email_sequence.touches) > CAMPAIGN_CONFIG['max_touches_per_sequence']:
            raise InvalidEmailSequenceError(
                f"Email sequence exceeds maximum {CAMPAIGN_CONFIG['max_touches_per_sequence']} touches"
            )

        # Validate touch intervals
        for i in range(1, len(email_sequence.touches)):
            current_offset = email_sequence.touches[i]['day_offset']
            previous_offset = email_sequence.touches[i-1]['day_offset']

            if current_offset <= previous_offset:
                raise InvalidEmailSequenceError(
                    f"Touch {i+1} day_offset must be greater than previous touch"
                )

            interval = current_offset - previous_offset
            if interval < CAMPAIGN_CONFIG['min_touch_interval_days']:
                raise InvalidEmailSequenceError(
                    f"Touch interval {interval} days is below minimum "
                    f"{CAMPAIGN_CONFIG['min_touch_interval_days']} days"
                )

    def start_campaign(self, campaign_id: str) -> Campaign:
        """
        Start campaign (activate).

        Args:
            campaign_id: Campaign identifier

        Returns:
            Updated campaign

        Raises:
            CampaignNotFoundError: If campaign not found
            CampaignAlreadyActiveError: If campaign already active
        """
        if campaign_id not in self.campaigns:
            raise CampaignNotFoundError(campaign_id)

        campaign = self.campaigns[campaign_id]

        if campaign.status == CampaignStatus.ACTIVE:
            raise CampaignAlreadyActiveError(campaign_id)

        campaign.status = CampaignStatus.ACTIVE
        campaign.updated_at = DeterministicClock.utcnow()

        self._save_campaigns()

        logger.info(f"Started campaign {campaign_id}: {campaign.name}")

        return campaign

    def pause_campaign(self, campaign_id: str) -> Campaign:
        """
        Pause active campaign.

        Args:
            campaign_id: Campaign identifier

        Returns:
            Updated campaign

        Raises:
            CampaignNotFoundError: If campaign not found
        """
        if campaign_id not in self.campaigns:
            raise CampaignNotFoundError(campaign_id)

        campaign = self.campaigns[campaign_id]
        campaign.status = CampaignStatus.PAUSED
        campaign.updated_at = DeterministicClock.utcnow()

        self._save_campaigns()

        logger.info(f"Paused campaign {campaign_id}: {campaign.name}")

        return campaign

    def complete_campaign(self, campaign_id: str) -> Campaign:
        """
        Mark campaign as completed.

        Args:
            campaign_id: Campaign identifier

        Returns:
            Updated campaign

        Raises:
            CampaignNotFoundError: If campaign not found
        """
        if campaign_id not in self.campaigns:
            raise CampaignNotFoundError(campaign_id)

        campaign = self.campaigns[campaign_id]
        campaign.status = CampaignStatus.COMPLETED
        campaign.updated_at = DeterministicClock.utcnow()

        self._save_campaigns()

        logger.info(f"Completed campaign {campaign_id}: {campaign.name}")

        return campaign

    def get_campaign(self, campaign_id: str) -> Campaign:
        """
        Get campaign by ID.

        Args:
            campaign_id: Campaign identifier

        Returns:
            Campaign

        Raises:
            CampaignNotFoundError: If campaign not found
        """
        if campaign_id not in self.campaigns:
            raise CampaignNotFoundError(campaign_id)

        return self.campaigns[campaign_id]

    def update_metrics(
        self,
        campaign_id: str,
        emails_sent: Optional[int] = None,
        emails_delivered: Optional[int] = None,
        emails_opened: Optional[int] = None,
        emails_clicked: Optional[int] = None,
        portal_visits: Optional[int] = None,
        data_submissions: Optional[int] = None
    ) -> Campaign:
        """
        Update campaign metrics.

        Args:
            campaign_id: Campaign identifier
            emails_sent: Number of emails sent
            emails_delivered: Number of emails delivered
            emails_opened: Number of emails opened
            emails_clicked: Number of emails clicked
            portal_visits: Number of portal visits
            data_submissions: Number of data submissions

        Returns:
            Updated campaign

        Raises:
            CampaignNotFoundError: If campaign not found
        """
        if campaign_id not in self.campaigns:
            raise CampaignNotFoundError(campaign_id)

        campaign = self.campaigns[campaign_id]

        if emails_sent is not None:
            campaign.emails_sent += emails_sent
        if emails_delivered is not None:
            campaign.emails_delivered += emails_delivered
        if emails_opened is not None:
            campaign.emails_opened += emails_opened
        if emails_clicked is not None:
            campaign.emails_clicked += emails_clicked
        if portal_visits is not None:
            campaign.portal_visits += portal_visits
        if data_submissions is not None:
            campaign.data_submissions += data_submissions

        campaign.updated_at = DeterministicClock.utcnow()

        self._save_campaigns()

        logger.debug(f"Updated metrics for campaign {campaign_id}")

        return campaign

    def get_active_campaigns(self) -> List[Campaign]:
        """
        Get list of active campaigns.

        Returns:
            List of active campaigns
        """
        return [
            campaign for campaign in self.campaigns.values()
            if campaign.status == CampaignStatus.ACTIVE
        ]

    def get_campaigns_by_status(self, status: CampaignStatus) -> List[Campaign]:
        """
        Get campaigns by status.

        Args:
            status: Campaign status

        Returns:
            List of campaigns
        """
        return [
            campaign for campaign in self.campaigns.values()
            if campaign.status == status
        ]

    def list_campaigns(self) -> List[Campaign]:
        """
        List all campaigns.

        Returns:
            List of campaigns
        """
        return list(self.campaigns.values())
