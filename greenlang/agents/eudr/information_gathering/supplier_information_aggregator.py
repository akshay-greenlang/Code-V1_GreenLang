# -*- coding: utf-8 -*-
"""
AGENT-EUDR-027: Information Gathering Agent - Supplier Information Aggregator

Aggregates, deduplicates, and reconciles supplier information from multiple
data sources into unified ``SupplierProfile`` objects. Implements entity
resolution via Jaro-Winkler string similarity, source priority ranking
for conflict resolution (government > certification > customs > trade_db
> supplier_self > public), discrepancy detection, and completeness scoring.

Production infrastructure includes:
    - Full Jaro-Winkler similarity algorithm implementation
    - Source priority ranking for authoritative conflict resolution
    - Multi-source profile merging with provenance tracking
    - Duplicate detection across supplier portfolios
    - Discrepancy detection between competing data sources
    - Completeness scoring (Decimal precision)
    - Batch aggregation with concurrent execution
    - SHA-256 provenance hash on every aggregated profile

Zero-Hallucination Guarantees:
    - Jaro-Winkler similarity computed via deterministic formula
    - Completeness scores use weighted Decimal arithmetic only
    - Conflict resolution follows static priority lookup table
    - No LLM involvement in entity resolution or scoring
    - All provenance hashes computed from canonical JSON

Regulatory References:
    - EUDR Article 9(1)(e): Supplier identification information
    - EUDR Article 9(1)(f): Country of production identification
    - EUDR Article 10(2): Supplier risk assessment data
    - EUDR Article 12(1): Competent authority supplier verification
    - EUDR Article 31: 5-year record retention for supplier profiles

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (Engine 4: Supplier Information Aggregator)
Agent ID: GL-EUDR-IGA-027
Status: Production Ready
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.information_gathering.config import (
    InformationGatheringConfig,
    get_config,
)
from greenlang.agents.eudr.information_gathering.models import (
    CertificateVerificationResult,
    DataDiscrepancy,
    DataSourcePriority,
    EUDRCommodity,
    SupplierProfile,
)
from greenlang.agents.eudr.information_gathering.provenance import ProvenanceTracker
from greenlang.agents.eudr.information_gathering.metrics import (
    record_supplier_aggregated,
    observe_aggregation_duration,
    record_api_error,
)

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Source Priority Ranking (lower number = higher authority)
# ---------------------------------------------------------------------------

_SOURCE_PRIORITY: Dict[str, int] = {
    DataSourcePriority.GOVERNMENT_REGISTRY.value: 1,
    DataSourcePriority.CERTIFICATION_BODY.value: 2,
    DataSourcePriority.CUSTOMS_RECORD.value: 3,
    DataSourcePriority.TRADE_DATABASE.value: 4,
    DataSourcePriority.SUPPLIER_SELF_DECLARED.value: 5,
    DataSourcePriority.PUBLIC_DATABASE.value: 6,
}

# Completeness field weights for supplier profile scoring
_PROFILE_FIELD_WEIGHTS: Dict[str, Decimal] = {
    "name": Decimal("0.15"),
    "country_code": Decimal("0.15"),
    "postal_address": Decimal("0.10"),
    "registration_number": Decimal("0.15"),
    "commodities": Decimal("0.10"),
    "certifications": Decimal("0.10"),
    "plot_ids": Decimal("0.10"),
    "email": Decimal("0.05"),
    "alternative_names": Decimal("0.05"),
    "tier_depth": Decimal("0.05"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute deterministic SHA-256 hash of data.

    Args:
        data: Any JSON-serializable object.

    Returns:
        64-character lowercase hex SHA-256 hash string.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _get_source_priority(source: str) -> int:
    """Return numeric priority for a data source (lower = higher authority).

    Unknown sources are assigned lowest priority (99).

    Args:
        source: Source identifier string.

    Returns:
        Integer priority (1 = highest, 99 = unknown).
    """
    return _SOURCE_PRIORITY.get(source, 99)


# ---------------------------------------------------------------------------
# Jaro-Winkler Similarity (Full Implementation)
# ---------------------------------------------------------------------------


def _jaro_similarity(s1: str, s2: str) -> float:
    """Compute Jaro similarity between two strings.

    The Jaro similarity is defined as:
        jaro = (1/3) * (m/|s1| + m/|s2| + (m - t)/m)

    where m is the number of matching characters and t is the number
    of transpositions.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Jaro similarity in range [0.0, 1.0].
    """
    if s1 == s2:
        return 1.0

    len1 = len(s1)
    len2 = len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0

    # Maximum matching distance
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    # Find matching characters
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        matches / len1 + matches / len2 + (matches - transpositions / 2) / matches
    ) / 3.0

    return jaro


def _jaro_winkler_similarity(
    s1: str,
    s2: str,
    winkler_prefix_weight: float = 0.1,
    max_prefix_length: int = 4,
) -> float:
    """Compute Jaro-Winkler similarity between two strings.

    Extends Jaro similarity with a prefix bonus for strings sharing
    a common prefix up to ``max_prefix_length`` characters.

    Formula: jw = jaro + (prefix_length * p * (1 - jaro))
    where p is the Winkler prefix weight (default 0.1).

    Args:
        s1: First string.
        s2: Second string.
        winkler_prefix_weight: Winkler scaling factor (max 0.25).
        max_prefix_length: Maximum prefix chars considered (max 4).

    Returns:
        Jaro-Winkler similarity in range [0.0, 1.0].
    """
    # Normalize inputs
    s1_lower = s1.strip().lower()
    s2_lower = s2.strip().lower()

    jaro = _jaro_similarity(s1_lower, s2_lower)

    # Compute common prefix length (up to max_prefix_length)
    prefix_len = 0
    limit = min(len(s1_lower), len(s2_lower), max_prefix_length)
    for i in range(limit):
        if s1_lower[i] == s2_lower[i]:
            prefix_len += 1
        else:
            break

    # Clamp Winkler weight
    p = min(winkler_prefix_weight, 0.25)

    return jaro + prefix_len * p * (1.0 - jaro)


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------


class SupplierInformationAggregator:
    """Engine for aggregating supplier information from multiple sources.

    Implements entity resolution, profile merging with source priority,
    duplicate detection, discrepancy identification, and completeness
    scoring for EUDR supplier due diligence.

    Args:
        config: Agent configuration (uses singleton if None).

    Example:
        >>> agg = SupplierInformationAggregator()
        >>> profile = await agg.aggregate_supplier(
        ...     "SUP-001",
        ...     sources={"government_registry": {...}, "supplier_self_declared": {...}}
        ... )
        >>> assert profile.completeness_score > Decimal("0")
    """

    def __init__(self, config: Optional[InformationGatheringConfig] = None) -> None:
        self._config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._profiles: Dict[str, SupplierProfile] = {}
        logger.info(
            "SupplierInformationAggregator initialized "
            "(fuzzy_threshold=%.2f, dedup=%s, algo=%s)",
            self._config.fuzzy_match_threshold,
            self._config.dedup_enabled,
            self._config.entity_resolution_algorithm,
        )

    async def aggregate_supplier(
        self,
        supplier_id: str,
        sources: Dict[str, Dict[str, Any]],
    ) -> SupplierProfile:
        """Aggregate supplier data from multiple sources into a unified profile.

        Sources are merged in priority order (government > certification >
        customs > trade > self-declared > public). Higher-priority sources
        override lower-priority ones for conflicting fields.

        Args:
            supplier_id: Supplier identifier.
            sources: Dict mapping source type to source data dict.
                Each source data dict may contain: name, country_code,
                postal_address, registration_number, commodities,
                certifications, plot_ids, email, alternative_names.

        Returns:
            Unified SupplierProfile with completeness score and provenance.
        """
        start_time = time.monotonic()

        # Sort sources by priority (highest authority first)
        sorted_sources = sorted(
            sources.items(),
            key=lambda kv: _get_source_priority(kv[0]),
        )

        # Merge fields from highest to lowest priority
        profile = self._merge_from_sources(supplier_id, sorted_sources)

        # Detect discrepancies
        profile.discrepancies = self.detect_discrepancies(sources)

        # Compute completeness
        profile.completeness_score = self.get_completeness_score(profile)

        # Compute confidence based on source count and priority
        profile.confidence_score = self._compute_confidence(sorted_sources)

        # Provenance hash
        profile.provenance_hash = _compute_hash({
            "supplier_id": supplier_id,
            "sources": list(sources.keys()),
            "completeness": str(profile.completeness_score),
        })
        profile.last_updated = _utcnow()

        # Store in cache
        self._profiles[supplier_id] = profile

        elapsed = time.monotonic() - start_time
        observe_aggregation_duration(elapsed)
        record_supplier_aggregated(
            profile.commodities[0].value if profile.commodities else "unknown"
        )

        self._provenance.create_entry(
            step="supplier_aggregation",
            source=",".join(sources.keys()),
            input_hash=_compute_hash(sources),
            output_hash=profile.provenance_hash,
        )

        logger.info(
            "Aggregated supplier %s from %d sources: completeness=%.1f%%, "
            "confidence=%.1f%%, discrepancies=%d (%.0fms)",
            supplier_id,
            len(sources),
            float(profile.completeness_score),
            float(profile.confidence_score),
            len(profile.discrepancies),
            elapsed * 1000,
        )
        return profile

    def _merge_from_sources(
        self,
        supplier_id: str,
        sorted_sources: List[Tuple[str, Dict[str, Any]]],
    ) -> SupplierProfile:
        """Merge supplier data from priority-sorted sources.

        Higher-priority sources take precedence for scalar fields.
        List fields (commodities, certifications, plot_ids, alternative_names)
        are accumulated from all sources.

        Args:
            supplier_id: Supplier identifier.
            sorted_sources: List of (source_type, data) tuples sorted
                by descending authority.

        Returns:
            Merged SupplierProfile.
        """
        name = ""
        country_code = ""
        postal_address = ""
        email: Optional[str] = None
        registration_number: Optional[str] = None
        tier_depth = 0
        all_commodities: List[EUDRCommodity] = []
        all_certs: List[CertificateVerificationResult] = []
        all_plots: List[str] = []
        all_alt_names: List[str] = []
        source_names: List[str] = []

        for source_type, data in sorted_sources:
            source_names.append(source_type)

            # Scalar fields: first (highest priority) non-empty wins
            if not name and data.get("name"):
                name = str(data["name"])
            if not country_code and data.get("country_code"):
                country_code = str(data["country_code"])
            if not postal_address and data.get("postal_address"):
                postal_address = str(data["postal_address"])
            if email is None and data.get("email"):
                email = str(data["email"])
            if registration_number is None and data.get("registration_number"):
                registration_number = str(data["registration_number"])
            if data.get("tier_depth") and int(data["tier_depth"]) > tier_depth:
                tier_depth = int(data["tier_depth"])

            # List fields: accumulate and deduplicate
            for c in data.get("commodities", []):
                if isinstance(c, EUDRCommodity):
                    if c not in all_commodities:
                        all_commodities.append(c)
                elif isinstance(c, str):
                    try:
                        commodity = EUDRCommodity(c)
                        if commodity not in all_commodities:
                            all_commodities.append(commodity)
                    except ValueError:
                        pass

            for cert in data.get("certifications", []):
                if isinstance(cert, CertificateVerificationResult):
                    all_certs.append(cert)

            for pid in data.get("plot_ids", []):
                if str(pid) not in all_plots:
                    all_plots.append(str(pid))

            for alt in data.get("alternative_names", []):
                alt_str = str(alt).strip()
                if alt_str and alt_str not in all_alt_names and alt_str != name:
                    all_alt_names.append(alt_str)

        return SupplierProfile(
            supplier_id=supplier_id,
            name=name,
            alternative_names=all_alt_names,
            postal_address=postal_address,
            country_code=country_code,
            email=email,
            registration_number=registration_number,
            commodities=all_commodities,
            certifications=all_certs,
            plot_ids=all_plots,
            tier_depth=tier_depth,
            data_sources=source_names,
            last_updated=_utcnow(),
        )

    def _compute_confidence(
        self,
        sorted_sources: List[Tuple[str, Dict[str, Any]]],
    ) -> Decimal:
        """Compute confidence score based on source diversity and authority.

        Score increases with more sources and higher-authority sources.
        Formula: base_per_source * (authority_weight / total_possible).

        Args:
            sorted_sources: Priority-sorted source list.

        Returns:
            Confidence score 0-100 as Decimal.
        """
        if not sorted_sources:
            return Decimal("0")

        base_per_source = Decimal("15")
        authority_bonus = Decimal("0")

        for source_type, _ in sorted_sources:
            priority = _get_source_priority(source_type)
            # Higher authority (lower number) gets more bonus
            if priority <= 2:
                authority_bonus += Decimal("10")
            elif priority <= 4:
                authority_bonus += Decimal("5")
            else:
                authority_bonus += Decimal("2")

        raw_score = min(
            base_per_source * Decimal(str(len(sorted_sources))) + authority_bonus,
            Decimal("100"),
        )
        return raw_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    async def batch_aggregate(
        self,
        supplier_sources: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> List[SupplierProfile]:
        """Aggregate multiple suppliers concurrently.

        Args:
            supplier_sources: Dict mapping supplier_id to sources dict.
                Structure: {supplier_id: {source_type: data_dict}}.

        Returns:
            List of aggregated SupplierProfile objects.
        """
        logger.info("Batch aggregating %d suppliers", len(supplier_sources))
        tasks = [
            self.aggregate_supplier(sid, sources)
            for sid, sources in supplier_sources.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: List[SupplierProfile] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                sid = list(supplier_sources.keys())[i]
                logger.error("Batch aggregate for %s raised: %s", sid, str(result))
                record_api_error("supplier_aggregation")
            else:
                output.append(result)
        return output

    def resolve_entity(
        self,
        name: str,
        country: Optional[str] = None,
        reg_number: Optional[str] = None,
    ) -> Optional[SupplierProfile]:
        """Resolve a supplier entity from known profiles using fuzzy matching.

        First checks for exact registration number match. Then falls back
        to Jaro-Winkler name similarity with optional country filter.

        Args:
            name: Supplier name to resolve.
            country: Optional country code filter.
            reg_number: Optional registration number for exact match.

        Returns:
            Best matching SupplierProfile, or None if no match exceeds
            the configured fuzzy_match_threshold.
        """
        # Exact registration number match
        if reg_number:
            for profile in self._profiles.values():
                if (
                    profile.registration_number
                    and profile.registration_number == reg_number
                ):
                    logger.debug(
                        "Entity resolved by reg_number: %s -> %s",
                        reg_number,
                        profile.supplier_id,
                    )
                    return profile

        # Fuzzy name matching
        best_match: Optional[SupplierProfile] = None
        best_score = 0.0
        threshold = self._config.fuzzy_match_threshold

        for profile in self._profiles.values():
            # Country filter
            if country and profile.country_code and profile.country_code != country:
                continue

            # Check primary name
            score = _jaro_winkler_similarity(name, profile.name)

            # Also check alternative names
            for alt_name in profile.alternative_names:
                alt_score = _jaro_winkler_similarity(name, alt_name)
                score = max(score, alt_score)

            if score > best_score and score >= threshold:
                best_score = score
                best_match = profile

        if best_match:
            logger.debug(
                "Entity resolved by name similarity: '%s' -> %s (score=%.3f)",
                name,
                best_match.supplier_id,
                best_score,
            )
        else:
            logger.debug(
                "Entity resolution failed for '%s' (best=%.3f, threshold=%.2f)",
                name,
                best_score,
                threshold,
            )
        return best_match

    def detect_duplicates(
        self,
        profiles: List[SupplierProfile],
    ) -> List[Tuple[str, str, float]]:
        """Detect potential duplicate suppliers using pairwise similarity.

        Compares all pairs of profiles using Jaro-Winkler similarity on
        names and registration numbers. Returns pairs exceeding the
        configured threshold.

        Args:
            profiles: List of supplier profiles to check.

        Returns:
            List of (supplier_id_a, supplier_id_b, similarity) tuples
            for pairs exceeding the dedup threshold, sorted by
            similarity descending.
        """
        if not self._config.dedup_enabled:
            logger.debug("Dedup disabled; returning empty list")
            return []

        threshold = self._config.fuzzy_match_threshold
        duplicates: List[Tuple[str, str, float]] = []

        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                p1 = profiles[i]
                p2 = profiles[j]

                # Name similarity
                name_sim = _jaro_winkler_similarity(p1.name, p2.name)

                # Registration number exact match boost
                reg_boost = 0.0
                if (
                    p1.registration_number
                    and p2.registration_number
                    and p1.registration_number == p2.registration_number
                ):
                    reg_boost = 0.3

                # Country match boost
                country_boost = 0.0
                if p1.country_code and p1.country_code == p2.country_code:
                    country_boost = 0.05

                # Check alternative names too
                alt_sim = 0.0
                for alt1 in p1.alternative_names:
                    alt_sim = max(alt_sim, _jaro_winkler_similarity(alt1, p2.name))
                for alt2 in p2.alternative_names:
                    alt_sim = max(alt_sim, _jaro_winkler_similarity(p1.name, alt2))

                best_name = max(name_sim, alt_sim)
                combined = min(best_name + reg_boost + country_boost, 1.0)

                if combined >= threshold:
                    duplicates.append((p1.supplier_id, p2.supplier_id, combined))

        # Sort by similarity descending
        duplicates.sort(key=lambda x: x[2], reverse=True)
        logger.info(
            "Duplicate detection on %d profiles: found %d potential duplicates",
            len(profiles),
            len(duplicates),
        )
        return duplicates

    def detect_discrepancies(
        self,
        profile_data: Dict[str, Dict[str, Any]],
    ) -> List[DataDiscrepancy]:
        """Detect discrepancies between data sources for the same supplier.

        Compares field values across all source pairs. Fields compared:
        name, country_code, postal_address, registration_number.

        Args:
            profile_data: Dict mapping source_type to data dict.

        Returns:
            List of DataDiscrepancy objects for conflicting field values.
        """
        discrepancies: List[DataDiscrepancy] = []
        fields_to_compare = [
            "name", "country_code", "postal_address", "registration_number",
        ]
        severity_map = {
            "name": "medium",
            "country_code": "high",
            "postal_address": "low",
            "registration_number": "critical",
        }
        recommendation_map = {
            "name": "Verify legal entity name via government registry",
            "country_code": "Confirm country of establishment; impacts risk classification",
            "postal_address": "Request updated address from supplier",
            "registration_number": "CRITICAL: Verify registration against official records",
        }

        source_list = list(profile_data.items())
        for i in range(len(source_list)):
            for j in range(i + 1, len(source_list)):
                src_a, data_a = source_list[i]
                src_b, data_b = source_list[j]

                for field_name in fields_to_compare:
                    val_a = str(data_a.get(field_name, "")).strip()
                    val_b = str(data_b.get(field_name, "")).strip()

                    # Skip if either is empty
                    if not val_a or not val_b:
                        continue

                    # Normalize for comparison
                    if val_a.lower() != val_b.lower():
                        discrepancies.append(
                            DataDiscrepancy(
                                field_name=field_name,
                                source_a=src_a,
                                value_a=val_a,
                                source_b=src_b,
                                value_b=val_b,
                                severity=severity_map.get(field_name, "medium"),
                                recommendation=recommendation_map.get(field_name, ""),
                            )
                        )

        if discrepancies:
            logger.warning(
                "Detected %d discrepancies across %d sources",
                len(discrepancies),
                len(profile_data),
            )
        return discrepancies

    def get_completeness_score(self, profile: SupplierProfile) -> Decimal:
        """Calculate weighted completeness score for a supplier profile.

        Each field contributes its weight (from ``_PROFILE_FIELD_WEIGHTS``)
        to the total score when populated. Score range: 0-100.

        Args:
            profile: SupplierProfile to score.

        Returns:
            Completeness percentage as Decimal, range 0.00 - 100.00.
        """
        total = Decimal("0")

        # name
        if profile.name:
            total += _PROFILE_FIELD_WEIGHTS["name"]
        # country_code
        if profile.country_code:
            total += _PROFILE_FIELD_WEIGHTS["country_code"]
        # postal_address
        if profile.postal_address:
            total += _PROFILE_FIELD_WEIGHTS["postal_address"]
        # registration_number
        if profile.registration_number:
            total += _PROFILE_FIELD_WEIGHTS["registration_number"]
        # commodities
        if profile.commodities:
            total += _PROFILE_FIELD_WEIGHTS["commodities"]
        # certifications
        if profile.certifications:
            total += _PROFILE_FIELD_WEIGHTS["certifications"]
        # plot_ids
        if profile.plot_ids:
            total += _PROFILE_FIELD_WEIGHTS["plot_ids"]
        # email
        if profile.email:
            total += _PROFILE_FIELD_WEIGHTS["email"]
        # alternative_names
        if profile.alternative_names:
            total += _PROFILE_FIELD_WEIGHTS["alternative_names"]
        # tier_depth (non-zero)
        if profile.tier_depth > 0:
            total += _PROFILE_FIELD_WEIGHTS["tier_depth"]

        score = (total * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        return score

    def get_profile(self, supplier_id: str) -> Optional[SupplierProfile]:
        """Retrieve a previously aggregated supplier profile.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Cached SupplierProfile or None.
        """
        return self._profiles.get(supplier_id)

    def get_all_profiles(self) -> List[SupplierProfile]:
        """Return all cached supplier profiles.

        Returns:
            List of all SupplierProfile objects in the cache.
        """
        return list(self._profiles.values())

    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Return engine statistics.

        Returns:
            Dict with profiles_cached, dedup_enabled,
            fuzzy_threshold, algorithm keys.
        """
        return {
            "profiles_cached": len(self._profiles),
            "dedup_enabled": self._config.dedup_enabled,
            "fuzzy_threshold": self._config.fuzzy_match_threshold,
            "algorithm": self._config.entity_resolution_algorithm,
        }

    def clear_profiles(self) -> None:
        """Clear all cached profiles (for testing)."""
        self._profiles.clear()
        logger.info("Supplier profile cache cleared")
