"""
Gamification features for supplier portal.

Provides leaderboards, badges, and progress tracking.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime

from ..models import (
    SupplierProgress,
    SupplierBadge,
    BadgeType,
    Leaderboard
)


logger = logging.getLogger(__name__)


class GamificationEngine:
    """
    Gamification engine for supplier engagement.

    Features:
    - Badge awarding
    - Leaderboard generation
    - Progress tracking
    - Recognition system
    """

    def __init__(self):
        """Initialize gamification engine."""
        self.supplier_progress: Dict[str, SupplierProgress] = {}
        self.badge_criteria = self._load_badge_criteria()
        logger.info("GamificationEngine initialized")

    def _load_badge_criteria(self) -> Dict[BadgeType, Dict]:
        """Load badge awarding criteria."""
        return {
            BadgeType.EARLY_ADOPTER: {
                "name": "Early Adopter",
                "description": "Among first 10 suppliers to submit data",
                "criteria": lambda progress, context: context.get("submission_rank", 999) <= 10
            },
            BadgeType.DATA_CHAMPION: {
                "name": "Data Champion",
                "description": "Data quality score >= 0.90",
                "criteria": lambda progress, context: (
                    progress.data_quality_score and progress.data_quality_score >= 0.90
                )
            },
            BadgeType.COMPLETE_PROFILE: {
                "name": "Complete Profile",
                "description": "100% field completion",
                "criteria": lambda progress, context: progress.completion_percentage >= 100.0
            },
            BadgeType.QUALITY_LEADER: {
                "name": "Quality Leader",
                "description": "Highest DQI in cohort",
                "criteria": lambda progress, context: context.get("is_quality_leader", False)
            },
            BadgeType.FAST_RESPONDER: {
                "name": "Fast Responder",
                "description": "Response within 7 days",
                "criteria": lambda progress, context: context.get("response_days", 999) <= 7
            }
        }

    def track_progress(
        self,
        supplier_id: str,
        campaign_id: str,
        completion_percentage: float,
        data_quality_score: Optional[float] = None
    ) -> SupplierProgress:
        """
        Track supplier progress.

        Args:
            supplier_id: Supplier identifier
            campaign_id: Campaign identifier
            completion_percentage: Progress percentage (0-100)
            data_quality_score: DQI score (0-1)

        Returns:
            Updated supplier progress
        """
        key = f"{supplier_id}:{campaign_id}"

        if key not in self.supplier_progress:
            self.supplier_progress[key] = SupplierProgress(
                supplier_id=supplier_id,
                campaign_id=campaign_id,
                completion_percentage=0.0
            )

        progress = self.supplier_progress[key]
        progress.completion_percentage = completion_percentage
        if data_quality_score is not None:
            progress.data_quality_score = data_quality_score
        progress.last_updated = datetime.utcnow()

        logger.debug(
            f"Updated progress for supplier {supplier_id}: {completion_percentage:.1f}%"
        )

        return progress

    def award_badge(
        self,
        supplier_id: str,
        campaign_id: str,
        badge_type: BadgeType,
        criteria_met: str
    ):
        """
        Award badge to supplier.

        Args:
            supplier_id: Supplier identifier
            campaign_id: Campaign identifier
            badge_type: Type of badge
            criteria_met: Description of criteria met
        """
        key = f"{supplier_id}:{campaign_id}"

        if key not in self.supplier_progress:
            return

        progress = self.supplier_progress[key]

        # Check if badge already awarded
        if any(b.badge_type == badge_type for b in progress.badges_earned):
            return

        badge = SupplierBadge(
            badge_type=badge_type,
            criteria_met=criteria_met
        )

        progress.badges_earned.append(badge)

        logger.info(
            f"Awarded {badge_type.value} badge to supplier {supplier_id}"
        )

    def check_and_award_badges(
        self,
        supplier_id: str,
        campaign_id: str,
        context: Optional[Dict] = None
    ) -> List[BadgeType]:
        """
        Check eligibility and award applicable badges.

        Args:
            supplier_id: Supplier identifier
            campaign_id: Campaign identifier
            context: Additional context for badge criteria

        Returns:
            List of newly awarded badge types
        """
        key = f"{supplier_id}:{campaign_id}"

        if key not in self.supplier_progress:
            return []

        progress = self.supplier_progress[key]
        context = context or {}
        newly_awarded = []

        # Check each badge criteria
        for badge_type, badge_info in self.badge_criteria.items():
            # Skip if already awarded
            if any(b.badge_type == badge_type for b in progress.badges_earned):
                continue

            # Check criteria
            if badge_info["criteria"](progress, context):
                self.award_badge(
                    supplier_id,
                    campaign_id,
                    badge_type,
                    badge_info["description"]
                )
                newly_awarded.append(badge_type)

        return newly_awarded

    def generate_leaderboard(
        self,
        campaign_id: str,
        top_n: int = 10,
        sort_by: str = "data_quality_score"
    ) -> Leaderboard:
        """
        Generate supplier leaderboard for campaign.

        Args:
            campaign_id: Campaign identifier
            top_n: Number of top suppliers to include
            sort_by: Sort criterion (data_quality_score, completion_percentage)

        Returns:
            Leaderboard
        """
        # Get all progress for campaign
        campaign_progress = [
            (key.split(":")[0], progress)
            for key, progress in self.supplier_progress.items()
            if progress.campaign_id == campaign_id
        ]

        # Sort by criterion
        if sort_by == "data_quality_score":
            campaign_progress.sort(
                key=lambda x: x[1].data_quality_score or 0.0,
                reverse=True
            )
        elif sort_by == "completion_percentage":
            campaign_progress.sort(
                key=lambda x: x[1].completion_percentage,
                reverse=True
            )

        # Generate leaderboard entries
        entries = []
        for rank, (supplier_id, progress) in enumerate(campaign_progress[:top_n], 1):
            progress.leaderboard_rank = rank
            entries.append({
                "rank": rank,
                "supplier_id": supplier_id,
                "completion_percentage": progress.completion_percentage,
                "data_quality_score": progress.data_quality_score,
                "badges_count": len(progress.badges_earned),
                "badges": [b.badge_type.value for b in progress.badges_earned]
            })

        leaderboard = Leaderboard(
            campaign_id=campaign_id,
            entries=entries
        )

        logger.info(f"Generated leaderboard for campaign {campaign_id} with {len(entries)} entries")

        return leaderboard

    def get_supplier_progress(
        self,
        supplier_id: str,
        campaign_id: str
    ) -> Optional[SupplierProgress]:
        """
        Get supplier progress.

        Args:
            supplier_id: Supplier identifier
            campaign_id: Campaign identifier

        Returns:
            Supplier progress or None
        """
        key = f"{supplier_id}:{campaign_id}"
        return self.supplier_progress.get(key)

    def get_campaign_statistics(self, campaign_id: str) -> Dict:
        """
        Get gamification statistics for campaign.

        Args:
            campaign_id: Campaign identifier

        Returns:
            Campaign gamification statistics
        """
        campaign_progress = [
            progress for progress in self.supplier_progress.values()
            if progress.campaign_id == campaign_id
        ]

        total_suppliers = len(campaign_progress)
        avg_completion = (
            sum(p.completion_percentage for p in campaign_progress) / total_suppliers
            if total_suppliers > 0
            else 0.0
        )

        # Count badges by type
        badge_counts = {}
        for progress in campaign_progress:
            for badge in progress.badges_earned:
                badge_type = badge.badge_type.value
                badge_counts[badge_type] = badge_counts.get(badge_type, 0) + 1

        return {
            "total_suppliers": total_suppliers,
            "average_completion": avg_completion,
            "badge_counts": badge_counts,
            "fully_completed": sum(
                1 for p in campaign_progress
                if p.completion_percentage >= 100.0
            )
        }
