# -*- coding: utf-8 -*-
"""
Percentage Rollout - INFRA-008 Targeting Subsystem

Provides deterministic percentage-based feature flag rollout using consistent
MD5 hashing. Given the same (flag_key, user_id) pair, the evaluation always
produces the same boolean result, ensuring a stable user experience across
requests, pods, and deployments.

Also supports weighted variant selection for MULTIVARIATE flags. Variant
assignment is deterministic via the same consistent hashing approach with an
additional salt to decorrelate variant assignment from the boolean rollout.

Design principles:
    - Zero external dependencies (stdlib hashlib only).
    - Deterministic: same inputs always produce the same output.
    - Fast: a single MD5 hash per evaluation (~0.3 us on modern hardware).
    - Anonymous fallback: when user_id is absent, uses random bucketing.

Example:
    >>> rollout = PercentageRollout()
    >>> rollout.evaluate("enable-scope3", "user-42", 25.0)
    True
    >>> rollout.evaluate("enable-scope3", "user-42", 25.0)
    True  # deterministic
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import List, Optional

from greenlang.infrastructure.feature_flags.models import FlagVariant

logger = logging.getLogger(__name__)


class PercentageRollout:
    """Consistent-hash-based percentage rollout and variant selection.

    Uses MD5 to map a (flag_key, user_id) pair to a bucket in the range
    [0, 99]. The bucket is compared against the rollout percentage to
    determine inclusion.

    For variant selection, each variant is allocated a proportional range
    of buckets based on its weight. The user is assigned to the variant
    whose range contains their hash bucket.

    Attributes:
        _random: Random instance used only for anonymous (no user_id) fallback.
    """

    def __init__(self) -> None:
        """Initialize PercentageRollout with a local Random instance."""
        self._random = random.Random()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        flag_key: str,
        user_id: Optional[str],
        rollout_percentage: float,
    ) -> bool:
        """Determine if a user is included in a percentage rollout.

        The evaluation is deterministic for the same (flag_key, user_id)
        pair. Anonymous users (user_id is None or empty) receive a random
        bucket on every call.

        Args:
            flag_key: Unique flag identifier.
            user_id: Unique user identifier, or None for anonymous.
            rollout_percentage: Rollout percentage in the range [0.0, 100.0].

        Returns:
            True if the user falls within the rollout percentage.
        """
        if rollout_percentage <= 0.0:
            return False
        if rollout_percentage >= 100.0:
            return True

        bucket = self._get_bucket(flag_key, user_id)
        included = bucket < rollout_percentage

        logger.debug(
            "PercentageRollout.evaluate flag_key=%s user_id=%s "
            "percentage=%.1f bucket=%d included=%s",
            flag_key,
            user_id,
            rollout_percentage,
            bucket,
            included,
        )
        return included

    def get_variant(
        self,
        flag_key: str,
        user_id: Optional[str],
        variants: List[FlagVariant],
    ) -> Optional[str]:
        """Select a variant for a user using weighted consistent hashing.

        Variants are sorted by variant_key to ensure a stable ordering. Each
        variant is allocated a range of the [0, 100) bucket space proportional
        to its weight. The user's hash bucket (computed with a variant-specific
        salt) determines which variant range they fall into.

        Args:
            flag_key: Unique flag identifier.
            user_id: Unique user identifier, or None for anonymous.
            variants: List of FlagVariant instances with weights.

        Returns:
            The variant_key of the selected variant, or None if no variants
            have positive weight.
        """
        if not variants:
            logger.debug(
                "PercentageRollout.get_variant flag_key=%s no variants provided",
                flag_key,
            )
            return None

        # Filter to variants with positive weight and sort for determinism
        active_variants = sorted(
            [v for v in variants if v.weight > 0],
            key=lambda v: v.variant_key,
        )
        if not active_variants:
            logger.debug(
                "PercentageRollout.get_variant flag_key=%s all variant weights are zero",
                flag_key,
            )
            return None

        total_weight = sum(v.weight for v in active_variants)
        if total_weight <= 0:
            return None

        # Use a variant-specific salt so variant assignment is decorrelated
        # from the boolean rollout decision
        hash_input = f"{flag_key}:variant_salt:{user_id or ''}"
        bucket = self._hash_to_bucket(hash_input)

        # Scale bucket to the total weight space
        scaled_bucket = (bucket / 100.0) * total_weight
        cumulative = 0.0

        for variant in active_variants:
            cumulative += variant.weight
            if scaled_bucket < cumulative:
                logger.debug(
                    "PercentageRollout.get_variant flag_key=%s user_id=%s "
                    "selected_variant=%s bucket=%d scaled=%.2f",
                    flag_key,
                    user_id,
                    variant.variant_key,
                    bucket,
                    scaled_bucket,
                )
                return variant.variant_key

        # Edge case: floating point drift -- assign to last variant
        selected = active_variants[-1].variant_key
        logger.debug(
            "PercentageRollout.get_variant flag_key=%s user_id=%s "
            "selected_variant=%s (fallback to last)",
            flag_key,
            user_id,
            selected,
        )
        return selected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_bucket(self, flag_key: str, user_id: Optional[str]) -> int:
        """Get the hash bucket for a (flag_key, user_id) pair.

        Anonymous users receive a random bucket (non-deterministic).

        Args:
            flag_key: Unique flag identifier.
            user_id: User identifier, or None for anonymous.

        Returns:
            Integer bucket in [0, 99].
        """
        if not user_id:
            return self._random.randint(0, 99)
        hash_input = f"{flag_key}:{user_id}"
        return self._hash_to_bucket(hash_input)

    def _hash_to_bucket(self, hash_input: str) -> int:
        """Compute a deterministic bucket in [0, 99] from a string.

        Uses MD5 for speed and uniform distribution. MD5 is not used for
        security here -- only for consistent bucketing.

        Args:
            hash_input: Arbitrary string to hash.

        Returns:
            Integer in the range [0, 99].
        """
        digest = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
        # Use the first 8 hex characters (32 bits) for the modulo
        hash_int = int(digest[:8], 16)
        return hash_int % 100
