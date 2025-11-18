"""
Traffic Router for A/B Testing

This module handles consistent user assignment to experiment variants using
deterministic hashing to ensure users always see the same variant.

Example:
    >>> router = TrafficRouter(redis_client)
    >>> await router.configure_experiment(experiment_id, variants)
    >>> variant = await router.get_variant(experiment_id, user_id)
"""

from typing import Dict, List, Optional, Any
import hashlib
import logging
import redis.asyncio as redis

from .experiment_models import ExperimentVariant, ExperimentAssignment

logger = logging.getLogger(__name__)


class TrafficRouter:
    """
    Routes users to experiment variants consistently.

    Uses deterministic hashing to ensure users always see the same variant
    throughout an experiment. Stores assignments in Redis for fast lookup.

    Attributes:
        redis_client: Redis client for caching assignments
    """

    def __init__(self, redis_client: redis.Redis):
        """
        Initialize TrafficRouter.

        Args:
            redis_client: Redis client instance
        """
        self.redis_client = redis_client
        self.cache_ttl = 86400 * 30  # 30 days

        logger.info("TrafficRouter initialized")

    async def configure_experiment(
        self,
        experiment_id: str,
        variants: List[ExperimentVariant]
    ) -> None:
        """
        Configure experiment variant routing.

        Stores variant configuration in Redis for fast routing decisions.

        Args:
            experiment_id: Experiment identifier
            variants: List of variants with traffic splits
        """
        try:
            # Store variant configuration
            key = f"experiment:{experiment_id}:config"
            config = {
                "variants": [
                    {
                        "name": v.name,
                        "traffic_split": v.traffic_split,
                        "is_control": v.is_control
                    }
                    for v in variants
                ]
            }

            await self.redis_client.set(
                key,
                str(config),
                ex=self.cache_ttl
            )

            logger.info(
                f"Experiment configured: {experiment_id}, variants={len(variants)}"
            )

        except Exception as e:
            logger.error(f"Failed to configure experiment: {e}", exc_info=True)
            raise

    async def get_variant(
        self,
        experiment_id: str,
        user_id: str
    ) -> Optional[str]:
        """
        Get variant assignment for a user.

        Uses deterministic hashing to ensure consistent assignment.
        Caches assignment in Redis for fast lookups.

        Args:
            experiment_id: Experiment identifier
            user_id: User identifier

        Returns:
            Variant name assigned to user, or None if experiment not configured
        """
        try:
            # Check cache for existing assignment
            cache_key = f"assignment:{experiment_id}:{user_id}"
            cached_variant = await self.redis_client.get(cache_key)

            if cached_variant:
                logger.debug(f"Cache hit for {user_id} in {experiment_id}: {cached_variant}")
                return cached_variant

            # Get experiment configuration
            config_key = f"experiment:{experiment_id}:config"
            config_str = await self.redis_client.get(config_key)

            if not config_str:
                logger.warning(f"Experiment not configured: {experiment_id}")
                return None

            # Parse config
            config = eval(config_str)  # In production, use json.loads
            variants = config['variants']

            # Deterministic variant assignment using hash
            variant_name = self._assign_variant(
                user_id=user_id,
                experiment_id=experiment_id,
                variants=variants
            )

            # Cache assignment
            await self.redis_client.set(
                cache_key,
                variant_name,
                ex=self.cache_ttl
            )

            logger.debug(
                f"New assignment for {user_id} in {experiment_id}: {variant_name}"
            )

            return variant_name

        except Exception as e:
            logger.error(f"Failed to get variant: {e}", exc_info=True)
            # Return None on error - don't break user experience
            return None

    def _assign_variant(
        self,
        user_id: str,
        experiment_id: str,
        variants: List[Dict[str, Any]]
    ) -> str:
        """
        Deterministically assign user to variant using hash.

        Uses SHA-256 hash of user_id + experiment_id to generate a
        deterministic number in [0, 1), then maps to variant based on
        traffic splits.

        Args:
            user_id: User identifier
            experiment_id: Experiment identifier
            variants: List of variant configurations

        Returns:
            Assigned variant name
        """
        # Create deterministic hash
        hash_input = f"{user_id}:{experiment_id}"
        hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()

        # Convert first 8 hex chars to int, normalize to [0, 1)
        hash_value = int(hash_digest[:8], 16)
        max_value = 16 ** 8  # Max value for 8 hex chars
        normalized_value = hash_value / max_value

        # Map to variant based on cumulative traffic splits
        cumulative = 0.0
        for variant in variants:
            cumulative += variant['traffic_split']
            if normalized_value <= cumulative:
                return variant['name']

        # Fallback to last variant (shouldn't happen if splits sum to 1.0)
        return variants[-1]['name']

    async def get_assignment_stats(
        self,
        experiment_id: str
    ) -> Dict[str, int]:
        """
        Get assignment statistics for an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dictionary of variant -> assignment count
        """
        try:
            # Scan for all assignments to this experiment
            pattern = f"assignment:{experiment_id}:*"
            cursor = 0
            assignments: Dict[str, int] = {}

            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )

                for key in keys:
                    variant = await self.redis_client.get(key)
                    if variant:
                        assignments[variant] = assignments.get(variant, 0) + 1

                if cursor == 0:
                    break

            logger.info(
                f"Assignment stats for {experiment_id}: {assignments}"
            )

            return assignments

        except Exception as e:
            logger.error(f"Failed to get assignment stats: {e}", exc_info=True)
            return {}

    async def clear_experiment(self, experiment_id: str) -> int:
        """
        Clear all assignments and configuration for an experiment.

        Args:
            experiment_id: Experiment to clear

        Returns:
            Number of keys deleted
        """
        try:
            # Clear configuration
            config_key = f"experiment:{experiment_id}:config"
            await self.redis_client.delete(config_key)

            # Clear all assignments
            pattern = f"assignment:{experiment_id}:*"
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )

                if keys:
                    deleted += await self.redis_client.delete(*keys)

                if cursor == 0:
                    break

            logger.info(
                f"Cleared experiment {experiment_id}: {deleted} assignments deleted"
            )

            return deleted

        except Exception as e:
            logger.error(f"Failed to clear experiment: {e}", exc_info=True)
            raise

    async def override_assignment(
        self,
        experiment_id: str,
        user_id: str,
        variant_name: str
    ) -> None:
        """
        Override variant assignment for a specific user.

        Useful for testing or special cases.

        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            variant_name: Variant to assign
        """
        try:
            cache_key = f"assignment:{experiment_id}:{user_id}"
            await self.redis_client.set(
                cache_key,
                variant_name,
                ex=self.cache_ttl
            )

            logger.info(
                f"Override assignment: {user_id} -> {variant_name} in {experiment_id}"
            )

        except Exception as e:
            logger.error(f"Failed to override assignment: {e}", exc_info=True)
            raise
