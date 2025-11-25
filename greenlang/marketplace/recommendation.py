# -*- coding: utf-8 -*-
"""
Recommendation Engine

Implements collaborative filtering, content-based filtering, and
popularity-based recommendations for marketplace agents.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging
import math

from sqlalchemy import func, and_, or_, desc
from sqlalchemy.orm import Session

from greenlang.determinism import FinancialDecimal
from greenlang.marketplace.models import (
    MarketplaceAgent,
    AgentInstall,
    AgentPurchase,
    AgentCategory,
    AgentTagModel,
    agent_tags,
)

logger = logging.getLogger(__name__)


@dataclass
class RecommendationScore:
    """Recommendation with score"""
    agent_id: str
    agent: MarketplaceAgent
    score: float
    reason: str
    metadata: Dict[str, Any]


class CollaborativeFilter:
    """
    Collaborative filtering recommender.

    Uses "users who installed X also installed Y" approach.
    """

    def __init__(self, session: Session):
        self.session = session

    def get_similar_users(self, user_id: str, min_common: int = 2) -> List[str]:
        """
        Find users with similar installation patterns.

        Args:
            user_id: Target user UUID
            min_common: Minimum number of common installs

        Returns:
            List of similar user IDs
        """
        # Get user's installed agents
        user_installs = self.session.query(AgentInstall.agent_id).filter(
            and_(
                AgentInstall.user_id == user_id,
                AgentInstall.active == True
            )
        ).all()

        user_agent_ids = {str(install.agent_id) for install in user_installs}

        if not user_agent_ids:
            return []

        # Find users who installed the same agents
        similar_users = self.session.query(
            AgentInstall.user_id,
            func.count(AgentInstall.agent_id).label('common_count')
        ).filter(
            and_(
                AgentInstall.agent_id.in_(user_agent_ids),
                AgentInstall.user_id != user_id,
                AgentInstall.active == True
            )
        ).group_by(AgentInstall.user_id).having(
            func.count(AgentInstall.agent_id) >= min_common
        ).order_by(desc('common_count')).limit(50).all()

        return [str(u.user_id) for u in similar_users]

    def recommend_from_similar_users(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[RecommendationScore]:
        """
        Recommend agents based on similar users' installations.

        Args:
            user_id: Target user UUID
            limit: Maximum recommendations

        Returns:
            List of recommendations with scores
        """
        # Get user's installed agents
        user_installs = self.session.query(AgentInstall.agent_id).filter(
            and_(
                AgentInstall.user_id == user_id,
                AgentInstall.active == True
            )
        ).all()

        user_agent_ids = {str(install.agent_id) for install in user_installs}

        # Get similar users
        similar_users = self.get_similar_users(user_id)

        if not similar_users:
            return []

        # Get agents installed by similar users
        candidate_installs = self.session.query(
            AgentInstall.agent_id,
            func.count(AgentInstall.user_id).label('install_count')
        ).filter(
            and_(
                AgentInstall.user_id.in_(similar_users),
                AgentInstall.active == True
            )
        ).group_by(AgentInstall.agent_id).all()

        # Filter out already installed agents
        recommendations = []
        for install in candidate_installs:
            agent_id_str = str(install.agent_id)
            if agent_id_str not in user_agent_ids:
                agent = self.session.query(MarketplaceAgent).filter(
                    MarketplaceAgent.id == install.agent_id
                ).first()

                if agent and agent.status == "published":
                    # Score based on how many similar users installed it
                    score = install.install_count / len(similar_users)

                    recommendations.append(RecommendationScore(
                        agent_id=agent_id_str,
                        agent=agent,
                        score=score,
                        reason="Users with similar interests also installed this",
                        metadata={"similar_user_count": install.install_count}
                    ))

        # Sort by score and limit
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:limit]

    def get_frequently_installed_together(
        self,
        agent_id: str,
        limit: int = 5
    ) -> List[RecommendationScore]:
        """
        Find agents frequently installed together with given agent.

        Args:
            agent_id: Target agent UUID
            limit: Maximum recommendations

        Returns:
            List of recommendations
        """
        # Get users who installed this agent
        users_with_agent = self.session.query(AgentInstall.user_id).filter(
            and_(
                AgentInstall.agent_id == agent_id,
                AgentInstall.active == True
            )
        ).all()

        user_ids = [str(u.user_id) for u in users_with_agent]

        if not user_ids:
            return []

        # Get other agents these users installed
        co_installs = self.session.query(
            AgentInstall.agent_id,
            func.count(AgentInstall.user_id).label('co_install_count')
        ).filter(
            and_(
                AgentInstall.user_id.in_(user_ids),
                AgentInstall.agent_id != agent_id,
                AgentInstall.active == True
            )
        ).group_by(AgentInstall.agent_id).order_by(
            desc('co_install_count')
        ).limit(limit).all()

        recommendations = []
        for install in co_installs:
            agent = self.session.query(MarketplaceAgent).filter(
                MarketplaceAgent.id == install.agent_id
            ).first()

            if agent and agent.status == "published":
                # Calculate lift (how much more likely to install together)
                score = install.co_install_count / len(user_ids)

                recommendations.append(RecommendationScore(
                    agent_id=str(agent.id),
                    agent=agent,
                    score=score,
                    reason="Frequently installed together",
                    metadata={"co_install_count": install.co_install_count}
                ))

        return recommendations


class ContentBasedFilter:
    """
    Content-based filtering recommender.

    Recommends agents based on similarity in categories, tags, and attributes.
    """

    def __init__(self, session: Session):
        self.session = session

    def calculate_similarity(
        self,
        agent1: MarketplaceAgent,
        agent2: MarketplaceAgent
    ) -> float:
        """
        Calculate similarity score between two agents.

        Uses:
        - Category similarity (40%)
        - Tag overlap (40%)
        - Price range similarity (10%)
        - Author similarity (10%)

        Args:
            agent1: First agent
            agent2: Second agent

        Returns:
            Similarity score (0-1)
        """
        score = 0.0

        # Category similarity (40%)
        if agent1.category_id == agent2.category_id:
            score += 0.4
        elif agent1.category and agent2.category:
            # Check if parent categories match
            if agent1.category.parent_id == agent2.category.parent_id:
                score += 0.2

        # Tag overlap (40%)
        tags1 = {tag.name for tag in agent1.tags}
        tags2 = {tag.name for tag in agent2.tags}

        if tags1 and tags2:
            overlap = len(tags1 & tags2)
            union = len(tags1 | tags2)
            jaccard = overlap / union if union > 0 else 0
            score += 0.4 * jaccard

        # Price range similarity (10%)
        price1 = FinancialDecimal.from_string(agent1.price) if agent1.price else 0
        price2 = FinancialDecimal.from_string(agent2.price) if agent2.price else 0

        if agent1.pricing_type == agent2.pricing_type:
            price_diff = abs(price1 - price2)
            max_price = max(price1, price2, 1)
            price_sim = 1 - min(price_diff / max_price, 1)
            score += 0.1 * price_sim

        # Same author (10%)
        if agent1.author_id == agent2.author_id:
            score += 0.1

        return min(score, 1.0)

    def find_similar_agents(
        self,
        agent_id: str,
        limit: int = 10,
        min_similarity: float = 0.3
    ) -> List[RecommendationScore]:
        """
        Find agents similar to given agent.

        Args:
            agent_id: Target agent UUID
            limit: Maximum recommendations
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar agents
        """
        target_agent = self.session.query(MarketplaceAgent).filter(
            MarketplaceAgent.id == agent_id
        ).first()

        if not target_agent:
            return []

        # Get candidate agents (same category or overlapping tags)
        candidates = self.session.query(MarketplaceAgent).filter(
            and_(
                MarketplaceAgent.id != agent_id,
                MarketplaceAgent.status == "published"
            )
        )

        # Filter by category if available
        if target_agent.category_id:
            candidates = candidates.filter(
                or_(
                    MarketplaceAgent.category_id == target_agent.category_id,
                    MarketplaceAgent.category.has(
                        parent_id=target_agent.category.parent_id
                    ) if target_agent.category.parent_id else False
                )
            )

        candidates = candidates.limit(100).all()

        # Calculate similarities
        recommendations = []
        for candidate in candidates:
            similarity = self.calculate_similarity(target_agent, candidate)

            if similarity >= min_similarity:
                recommendations.append(RecommendationScore(
                    agent_id=str(candidate.id),
                    agent=candidate,
                    score=similarity,
                    reason="Similar to agents you've used",
                    metadata={"similarity": similarity}
                ))

        # Sort by similarity
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:limit]

    def recommend_by_category(
        self,
        category_id: int,
        limit: int = 10,
        exclude_agent_ids: Optional[Set[str]] = None
    ) -> List[RecommendationScore]:
        """
        Recommend top agents in a category.

        Args:
            category_id: Category ID
            limit: Maximum recommendations
            exclude_agent_ids: Agent IDs to exclude

        Returns:
            List of recommendations
        """
        query = self.session.query(MarketplaceAgent).filter(
            and_(
                MarketplaceAgent.category_id == category_id,
                MarketplaceAgent.status == "published"
            )
        )

        if exclude_agent_ids:
            query = query.filter(
                ~MarketplaceAgent.id.in_(exclude_agent_ids)
            )

        agents = query.order_by(
            MarketplaceAgent.wilson_score.desc(),
            MarketplaceAgent.downloads.desc()
        ).limit(limit).all()

        recommendations = []
        for agent in agents:
            recommendations.append(RecommendationScore(
                agent_id=str(agent.id),
                agent=agent,
                score=agent.wilson_score,
                reason=f"Popular in {agent.category.name}",
                metadata={"downloads": agent.downloads}
            ))

        return recommendations


class PopularityBasedRecommender:
    """
    Popularity-based recommender.

    Recommends trending and popular agents.
    """

    def __init__(self, session: Session):
        self.session = session

    def get_trending(
        self,
        days: int = 7,
        limit: int = 10
    ) -> List[RecommendationScore]:
        """
        Get trending agents based on recent downloads.

        Args:
            days: Number of days to look back
            limit: Maximum recommendations

        Returns:
            List of trending agents
        """
        since = DeterministicClock.utcnow() - timedelta(days=days)

        # Count recent installs
        recent_installs = self.session.query(
            AgentInstall.agent_id,
            func.count(AgentInstall.id).label('recent_count')
        ).filter(
            AgentInstall.installed_at >= since
        ).group_by(AgentInstall.agent_id).subquery()

        # Join with agents
        agents = self.session.query(
            MarketplaceAgent,
            recent_installs.c.recent_count
        ).join(
            recent_installs,
            MarketplaceAgent.id == recent_installs.c.agent_id
        ).filter(
            MarketplaceAgent.status == "published"
        ).order_by(
            desc(recent_installs.c.recent_count)
        ).limit(limit).all()

        recommendations = []
        for agent, count in agents:
            recommendations.append(RecommendationScore(
                agent_id=str(agent.id),
                agent=agent,
                score=float(count),
                reason=f"Trending this week ({count} recent installs)",
                metadata={"recent_installs": count}
            ))

        return recommendations

    def get_most_downloaded(
        self,
        limit: int = 10,
        category_id: Optional[int] = None
    ) -> List[RecommendationScore]:
        """
        Get most downloaded agents.

        Args:
            limit: Maximum recommendations
            category_id: Optional category filter

        Returns:
            List of most downloaded agents
        """
        query = self.session.query(MarketplaceAgent).filter(
            MarketplaceAgent.status == "published"
        )

        if category_id:
            query = query.filter(MarketplaceAgent.category_id == category_id)

        agents = query.order_by(
            MarketplaceAgent.downloads.desc()
        ).limit(limit).all()

        recommendations = []
        for agent in agents:
            recommendations.append(RecommendationScore(
                agent_id=str(agent.id),
                agent=agent,
                score=float(agent.downloads),
                reason=f"Popular ({agent.downloads:,} downloads)",
                metadata={"downloads": agent.downloads}
            ))

        return recommendations

    def get_new_and_noteworthy(
        self,
        days: int = 30,
        limit: int = 10,
        min_rating: float = 4.0
    ) -> List[RecommendationScore]:
        """
        Get new agents with good ratings.

        Args:
            days: Maximum age in days
            limit: Maximum recommendations
            min_rating: Minimum rating threshold

        Returns:
            List of new and noteworthy agents
        """
        since = DeterministicClock.utcnow() - timedelta(days=days)

        agents = self.session.query(MarketplaceAgent).filter(
            and_(
                MarketplaceAgent.status == "published",
                MarketplaceAgent.published_at >= since,
                MarketplaceAgent.rating_avg >= min_rating,
                MarketplaceAgent.rating_count >= 3  # At least 3 ratings
            )
        ).order_by(
            MarketplaceAgent.wilson_score.desc(),
            MarketplaceAgent.published_at.desc()
        ).limit(limit).all()

        recommendations = []
        for agent in agents:
            days_old = (DeterministicClock.utcnow() - agent.published_at).days
            recommendations.append(RecommendationScore(
                agent_id=str(agent.id),
                agent=agent,
                score=agent.wilson_score,
                reason=f"New and highly rated ({days_old} days old)",
                metadata={
                    "days_old": days_old,
                    "rating": agent.rating_avg
                }
            ))

        return recommendations


class RecommendationEngine:
    """
    Main recommendation engine.

    Combines multiple recommendation strategies.
    """

    def __init__(self, session: Session):
        self.session = session
        self.collaborative = CollaborativeFilter(session)
        self.content_based = ContentBasedFilter(session)
        self.popularity = PopularityBasedRecommender(session)

    def get_personalized_recommendations(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[RecommendationScore]:
        """
        Get personalized recommendations for a user.

        Combines:
        - Collaborative filtering (50%)
        - Content-based filtering (30%)
        - Popularity-based (20%)

        Args:
            user_id: Target user UUID
            limit: Maximum recommendations

        Returns:
            List of recommendations
        """
        all_recommendations = []
        seen_agent_ids = set()

        # Get user's installed agents
        user_installs = self.session.query(AgentInstall.agent_id).filter(
            and_(
                AgentInstall.user_id == user_id,
                AgentInstall.active == True
            )
        ).all()

        user_agent_ids = {str(install.agent_id) for install in user_installs}

        # Collaborative filtering (50% weight)
        collab_recs = self.collaborative.recommend_from_similar_users(
            user_id,
            limit=limit * 2
        )

        for rec in collab_recs[:limit // 2]:
            if rec.agent_id not in seen_agent_ids:
                rec.score *= 0.5
                all_recommendations.append(rec)
                seen_agent_ids.add(rec.agent_id)

        # Content-based filtering (30% weight)
        if user_agent_ids:
            # Use most recent install as seed
            recent_install = self.session.query(AgentInstall).filter(
                and_(
                    AgentInstall.user_id == user_id,
                    AgentInstall.active == True
                )
            ).order_by(AgentInstall.installed_at.desc()).first()

            if recent_install:
                content_recs = self.content_based.find_similar_agents(
                    str(recent_install.agent_id),
                    limit=limit
                )

                for rec in content_recs:
                    if rec.agent_id not in seen_agent_ids and rec.agent_id not in user_agent_ids:
                        rec.score *= 0.3
                        all_recommendations.append(rec)
                        seen_agent_ids.add(rec.agent_id)

        # Popularity-based (20% weight)
        popular_recs = self.popularity.get_trending(limit=limit)

        for rec in popular_recs:
            if rec.agent_id not in seen_agent_ids and rec.agent_id not in user_agent_ids:
                rec.score *= 0.2
                all_recommendations.append(rec)
                seen_agent_ids.add(rec.agent_id)

        # Sort by combined score
        all_recommendations.sort(key=lambda x: x.score, reverse=True)

        return all_recommendations[:limit]

    def get_recommendations_for_agent(
        self,
        agent_id: str,
        limit: int = 5
    ) -> List[RecommendationScore]:
        """
        Get recommendations to show on an agent's detail page.

        Args:
            agent_id: Target agent UUID
            limit: Maximum recommendations

        Returns:
            List of recommendations
        """
        recommendations = []
        seen_agent_ids = {agent_id}

        # Frequently installed together (top priority)
        freq_recs = self.collaborative.get_frequently_installed_together(
            agent_id,
            limit=limit // 2
        )

        for rec in freq_recs:
            if rec.agent_id not in seen_agent_ids:
                recommendations.append(rec)
                seen_agent_ids.add(rec.agent_id)

        # Similar agents (fill remaining slots)
        similar_recs = self.content_based.find_similar_agents(
            agent_id,
            limit=limit
        )

        for rec in similar_recs:
            if rec.agent_id not in seen_agent_ids:
                recommendations.append(rec)
                seen_agent_ids.add(rec.agent_id)

            if len(recommendations) >= limit:
                break

        return recommendations[:limit]

    def get_homepage_recommendations(
        self,
        user_id: Optional[str] = None,
        limit: int = 20
    ) -> Dict[str, List[RecommendationScore]]:
        """
        Get recommendations for homepage display.

        Returns different sections:
        - For you (personalized)
        - Trending
        - New and noteworthy
        - Most downloaded

        Args:
            user_id: Optional user UUID for personalization
            limit: Recommendations per section

        Returns:
            Dictionary of recommendation lists by section
        """
        sections = {}

        # Personalized section (if user logged in)
        if user_id:
            sections["for_you"] = self.get_personalized_recommendations(
                user_id,
                limit=limit
            )

        # Trending
        sections["trending"] = self.popularity.get_trending(
            days=7,
            limit=limit
        )

        # New and noteworthy
        sections["new_and_noteworthy"] = self.popularity.get_new_and_noteworthy(
            days=30,
            limit=limit
        )

        # Most downloaded
        sections["most_downloaded"] = self.popularity.get_most_downloaded(
            limit=limit
        )

        return sections
