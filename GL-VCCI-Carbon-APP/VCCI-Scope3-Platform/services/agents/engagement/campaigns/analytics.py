# -*- coding: utf-8 -*-
"""
Campaign analytics and reporting.

Provides performance metrics, insights, and dashboards for campaigns.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime

from ..models import (
    Campaign,
    CampaignAnalytics,
    EmailMessage,
    EmailStatus
)
from ..exceptions import CampaignNotFoundError


logger = logging.getLogger(__name__)


class CampaignAnalytics:
    """
    Analyzes campaign performance and generates insights.

    Features:
    - Email delivery metrics
    - Engagement tracking (opens, clicks)
    - Portal conversion tracking
    - Response rate analysis
    - Time-to-response metrics
    """

    def __init__(self):
        """Initialize campaign analytics."""
        logger.info("CampaignAnalytics initialized")

    def generate_analytics(
        self,
        campaign: Campaign,
        messages: List[EmailMessage],
        data_submissions: Optional[Dict[str, datetime]] = None
    ) -> CampaignAnalytics:
        """
        Generate comprehensive analytics for campaign.

        Args:
            campaign: Campaign to analyze
            messages: List of campaign email messages
            data_submissions: Optional dict of supplier_id -> submission_timestamp

        Returns:
            Campaign analytics
        """
        # Email metrics
        emails_sent = len([m for m in messages if m.status != EmailStatus.PENDING])
        emails_delivered = len([m for m in messages if m.status in [
            EmailStatus.DELIVERED, EmailStatus.OPENED, EmailStatus.CLICKED
        ]])
        emails_opened = len([m for m in messages if m.opened_at is not None])
        emails_clicked = len([m for m in messages if m.clicked_at is not None])
        emails_bounced = len([m for m in messages if m.status == EmailStatus.BOUNCED])

        # Calculate rates
        delivery_rate = emails_delivered / emails_sent if emails_sent > 0 else 0.0
        open_rate = emails_opened / emails_delivered if emails_delivered > 0 else 0.0
        click_rate = emails_clicked / emails_opened if emails_opened > 0 else 0.0

        # Portal metrics (from campaign)
        portal_visits = campaign.portal_visits
        unique_visitors = len(set(m.supplier_id for m in messages if m.clicked_at))

        # Data submissions
        submissions_count = campaign.data_submissions

        # Response rate
        response_rate = (
            submissions_count / len(campaign.target_suppliers)
            if len(campaign.target_suppliers) > 0
            else 0.0
        )

        # Time to response
        avg_time_to_response = None
        if data_submissions:
            response_times = []
            for supplier_id, submission_time in data_submissions.items():
                # Find first email sent to this supplier
                supplier_messages = [
                    m for m in messages
                    if m.supplier_id == supplier_id and m.sent_at
                ]
                if supplier_messages:
                    first_sent = min(m.sent_at for m in supplier_messages)
                    time_diff = (submission_time - first_sent).total_seconds() / 3600  # hours
                    response_times.append(time_diff)

            if response_times:
                avg_time_to_response = sum(response_times) / len(response_times)

        analytics = CampaignAnalytics(
            campaign_id=campaign.campaign_id,
            campaign_name=campaign.name,
            emails_sent=emails_sent,
            emails_delivered=emails_delivered,
            emails_opened=emails_opened,
            emails_clicked=emails_clicked,
            emails_bounced=emails_bounced,
            portal_visits=portal_visits,
            unique_visitors=unique_visitors,
            data_submissions=submissions_count,
            delivery_rate=delivery_rate,
            open_rate=open_rate,
            click_rate=click_rate,
            response_rate=response_rate,
            avg_time_to_response_hours=avg_time_to_response
        )

        logger.info(
            f"Generated analytics for campaign {campaign.campaign_id}: "
            f"{response_rate:.1%} response rate"
        )

        return analytics

    def compare_to_target(
        self,
        analytics: CampaignAnalytics,
        target_response_rate: float
    ) -> Dict[str, any]:
        """
        Compare campaign performance to target.

        Args:
            analytics: Campaign analytics
            target_response_rate: Target response rate

        Returns:
            Comparison results
        """
        actual_response_rate = analytics.response_rate
        variance = actual_response_rate - target_response_rate
        variance_pct = (variance / target_response_rate * 100) if target_response_rate > 0 else 0

        status = "on_track" if actual_response_rate >= target_response_rate else "below_target"

        return {
            "target_response_rate": target_response_rate,
            "actual_response_rate": actual_response_rate,
            "variance": variance,
            "variance_percentage": variance_pct,
            "status": status,
            "on_track": status == "on_track"
        }

    def get_engagement_funnel(
        self,
        analytics: CampaignAnalytics
    ) -> Dict[str, Dict]:
        """
        Generate engagement funnel metrics.

        Args:
            analytics: Campaign analytics

        Returns:
            Funnel stages with counts and conversion rates
        """
        funnel = {
            "emails_sent": {
                "count": analytics.emails_sent,
                "rate": 1.0
            },
            "emails_delivered": {
                "count": analytics.emails_delivered,
                "rate": analytics.delivery_rate
            },
            "emails_opened": {
                "count": analytics.emails_opened,
                "rate": analytics.open_rate
            },
            "emails_clicked": {
                "count": analytics.emails_clicked,
                "rate": analytics.click_rate
            },
            "portal_visits": {
                "count": analytics.portal_visits,
                "rate": (
                    analytics.portal_visits / analytics.emails_sent
                    if analytics.emails_sent > 0
                    else 0.0
                )
            },
            "data_submissions": {
                "count": analytics.data_submissions,
                "rate": analytics.response_rate
            }
        }

        return funnel

    def get_touch_performance(
        self,
        messages: List[EmailMessage]
    ) -> Dict[int, Dict]:
        """
        Analyze performance by touch number.

        Args:
            messages: List of campaign messages

        Returns:
            Performance metrics by touch
        """
        touch_stats = {}

        for message in messages:
            touch_num = message.tracking_metadata.get('touch_number', 0)

            if touch_num not in touch_stats:
                touch_stats[touch_num] = {
                    "sent": 0,
                    "delivered": 0,
                    "opened": 0,
                    "clicked": 0
                }

            stats = touch_stats[touch_num]

            if message.status != EmailStatus.PENDING:
                stats["sent"] += 1

            if message.status in [EmailStatus.DELIVERED, EmailStatus.OPENED, EmailStatus.CLICKED]:
                stats["delivered"] += 1

            if message.opened_at:
                stats["opened"] += 1

            if message.clicked_at:
                stats["clicked"] += 1

        # Calculate rates
        for touch_num, stats in touch_stats.items():
            stats["delivery_rate"] = (
                stats["delivered"] / stats["sent"]
                if stats["sent"] > 0
                else 0.0
            )
            stats["open_rate"] = (
                stats["opened"] / stats["delivered"]
                if stats["delivered"] > 0
                else 0.0
            )
            stats["click_rate"] = (
                stats["clicked"] / stats["opened"]
                if stats["opened"] > 0
                else 0.0
            )

        return touch_stats

    def get_supplier_engagement(
        self,
        messages: List[EmailMessage]
    ) -> Dict[str, Dict]:
        """
        Analyze engagement by supplier.

        Args:
            messages: List of campaign messages

        Returns:
            Engagement metrics by supplier
        """
        supplier_stats = {}

        for message in messages:
            supplier_id = message.supplier_id

            if supplier_id not in supplier_stats:
                supplier_stats[supplier_id] = {
                    "emails_sent": 0,
                    "emails_opened": 0,
                    "emails_clicked": 0,
                    "last_opened": None,
                    "last_clicked": None,
                    "engagement_score": 0.0
                }

            stats = supplier_stats[supplier_id]

            if message.status != EmailStatus.PENDING:
                stats["emails_sent"] += 1

            if message.opened_at:
                stats["emails_opened"] += 1
                if not stats["last_opened"] or message.opened_at > stats["last_opened"]:
                    stats["last_opened"] = message.opened_at

            if message.clicked_at:
                stats["emails_clicked"] += 1
                if not stats["last_clicked"] or message.clicked_at > stats["last_clicked"]:
                    stats["last_clicked"] = message.clicked_at

        # Calculate engagement scores
        for supplier_id, stats in supplier_stats.items():
            # Simple engagement score: (opens * 1 + clicks * 2) / emails_sent
            score = (
                (stats["emails_opened"] + stats["emails_clicked"] * 2) / stats["emails_sent"]
                if stats["emails_sent"] > 0
                else 0.0
            )
            stats["engagement_score"] = min(score, 3.0)  # Cap at 3.0

        return supplier_stats

    def export_analytics(
        self,
        analytics: CampaignAnalytics,
        format: str = "dict"
    ) -> Dict:
        """
        Export analytics in specified format.

        Args:
            analytics: Campaign analytics
            format: Export format (dict, json)

        Returns:
            Exported analytics
        """
        if format == "dict":
            return analytics.model_dump()
        elif format == "json":
            return analytics.model_dump_json()
        else:
            return analytics.model_dump()
