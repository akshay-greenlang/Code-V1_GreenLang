# -*- coding: utf-8 -*-
"""
Risk Scoring Engine - SEC-007 Security Scanning Pipeline

Calculates comprehensive risk scores for vulnerabilities using multiple
data sources: CVSS 3.1 base score, EPSS exploitability prediction,
CISA KEV (Known Exploited Vulnerabilities), and asset criticality.

Risk Score Formula:
    risk_score = (
        base_weight * cvss_score +
        epss_weight * epss_adjustment +
        kev_weight * kev_boost +
        asset_weight * asset_multiplier
    ) / total_weight

    Where:
    - cvss_score: CVSS 3.1 base score (0-10)
    - epss_adjustment: EPSS percentile * 10 (0-10)
    - kev_boost: +3 if in CISA KEV catalog
    - asset_multiplier: 0.8-1.5 based on asset criticality

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-007 Security Scanning Pipeline
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from functools import lru_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AssetCriticality(str, Enum):
    """Asset criticality levels for risk weighting."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class CVSSVector:
    """CVSS 3.1 vector components for detailed scoring."""

    attack_vector: str = "N"  # N=Network, A=Adjacent, L=Local, P=Physical
    attack_complexity: str = "L"  # L=Low, H=High
    privileges_required: str = "N"  # N=None, L=Low, H=High
    user_interaction: str = "N"  # N=None, R=Required
    scope: str = "U"  # U=Unchanged, C=Changed
    confidentiality_impact: str = "N"  # N=None, L=Low, H=High
    integrity_impact: str = "N"  # N=None, L=Low, H=High
    availability_impact: str = "N"  # N=None, L=Low, H=High


@dataclass
class EPSSData:
    """EPSS (Exploit Prediction Scoring System) data."""

    cve: str
    score: float  # 0.0 to 1.0 probability
    percentile: float  # 0.0 to 1.0 percentile
    date: datetime


@dataclass
class KEVEntry:
    """CISA Known Exploited Vulnerability entry."""

    cve: str
    vendor: str
    product: str
    vulnerability_name: str
    date_added: datetime
    due_date: datetime
    short_description: str
    required_action: str


@dataclass
class RiskScoreResult:
    """Detailed risk scoring result."""

    final_score: float
    cvss_component: float
    epss_component: float
    kev_component: float
    asset_component: float
    factors: Dict[str, Any]
    calculation_time_ms: float


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RiskScorerConfig:
    """Configuration for risk scoring engine.

    Attributes:
        cvss_weight: Weight for CVSS score (default 0.4).
        epss_weight: Weight for EPSS score (default 0.3).
        kev_weight: Weight for KEV presence (default 0.2).
        asset_weight: Weight for asset criticality (default 0.1).
        kev_boost: Additional score boost for KEV (default 3.0).
        epss_api_url: FIRST.org EPSS API endpoint.
        kev_feed_url: CISA KEV JSON feed URL.
        cache_ttl_hours: Cache TTL for external data.
        enable_caching: Enable LRU caching for lookups.
    """

    cvss_weight: float = 0.4
    epss_weight: float = 0.3
    kev_weight: float = 0.2
    asset_weight: float = 0.1
    kev_boost: float = 3.0
    epss_api_url: str = "https://api.first.org/data/v1/epss"
    kev_feed_url: str = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    cache_ttl_hours: int = 24
    enable_caching: bool = True


# ---------------------------------------------------------------------------
# Risk Scorer
# ---------------------------------------------------------------------------


class RiskScorer:
    """Risk scoring engine for vulnerability prioritization.

    Combines multiple risk factors to produce a unified risk score (0-10)
    that enables effective vulnerability prioritization. Uses external
    data sources (EPSS, CISA KEV) for real-world exploitability context.

    Example:
        >>> scorer = RiskScorer()
        >>> result = await scorer.calculate_risk_score(
        ...     cvss_score=7.5,
        ...     cve="CVE-2024-1234",
        ...     asset_criticality="high"
        ... )
        >>> print(f"Risk score: {result.final_score}")
    """

    def __init__(
        self,
        config: Optional[RiskScorerConfig] = None,
        http_client: Optional[Any] = None,
    ) -> None:
        """Initialize RiskScorer.

        Args:
            config: Scoring configuration.
            http_client: Optional async HTTP client for API calls.
        """
        self._config = config or RiskScorerConfig()
        self._http_client = http_client

        # Caches
        self._epss_cache: Dict[str, EPSSData] = {}
        self._kev_set: Set[str] = set()
        self._kev_data: Dict[str, KEVEntry] = {}
        self._cache_updated_at: Optional[datetime] = None

        # Asset criticality multipliers
        self._asset_multipliers = {
            AssetCriticality.CRITICAL: 1.5,
            AssetCriticality.HIGH: 1.2,
            AssetCriticality.MEDIUM: 1.0,
            AssetCriticality.LOW: 0.8,
        }

    async def calculate_risk_score(
        self,
        cvss_score: Optional[float] = None,
        cvss_vector: Optional[str] = None,
        cve: Optional[str] = None,
        epss_score: Optional[float] = None,
        is_kev: Optional[bool] = None,
        asset_criticality: str = "medium",
    ) -> RiskScoreResult:
        """Calculate comprehensive risk score.

        Args:
            cvss_score: CVSS 3.1 base score (0-10).
            cvss_vector: Optional CVSS vector string for parsing.
            cve: CVE identifier for EPSS/KEV lookup.
            epss_score: Pre-fetched EPSS score (0-1).
            is_kev: Pre-fetched KEV status.
            asset_criticality: Asset criticality level.

        Returns:
            RiskScoreResult with detailed scoring breakdown.
        """
        start_time = datetime.utcnow()

        factors: Dict[str, Any] = {
            "cvss_score": cvss_score,
            "cvss_vector": cvss_vector,
            "cve": cve,
            "asset_criticality": asset_criticality,
        }

        # Parse CVSS vector if score not provided
        if cvss_score is None and cvss_vector:
            cvss_score = self._parse_cvss_vector(cvss_vector)

        # Default CVSS score if not available
        if cvss_score is None:
            cvss_score = 5.0  # Medium default

        # Fetch EPSS data if not provided and CVE is available
        epss_data = None
        if epss_score is None and cve:
            epss_data = await self._fetch_epss(cve)
            if epss_data:
                epss_score = epss_data.score
                factors["epss_percentile"] = epss_data.percentile

        if epss_score is None:
            epss_score = 0.0

        # Check KEV if not provided and CVE is available
        kev_entry = None
        if is_kev is None and cve:
            is_kev, kev_entry = await self._check_kev(cve)
            if kev_entry:
                factors["kev_due_date"] = kev_entry.due_date.isoformat()

        if is_kev is None:
            is_kev = False

        # Parse asset criticality
        try:
            asset_crit = AssetCriticality(asset_criticality.lower())
        except ValueError:
            asset_crit = AssetCriticality.MEDIUM

        # Calculate component scores
        cvss_component = cvss_score * self._config.cvss_weight

        # EPSS adjustment: convert probability to 0-10 scale
        epss_adjusted = epss_score * 10
        epss_component = epss_adjusted * self._config.epss_weight

        # KEV boost
        kev_component = self._config.kev_boost if is_kev else 0.0
        kev_component *= self._config.kev_weight

        # Asset criticality multiplier
        asset_multiplier = self._asset_multipliers.get(asset_crit, 1.0)
        asset_component = asset_multiplier

        # Calculate final score
        total_weight = (
            self._config.cvss_weight +
            self._config.epss_weight +
            self._config.kev_weight +
            self._config.asset_weight
        )

        raw_score = (
            cvss_component +
            epss_component +
            kev_component
        )

        # Apply asset multiplier to the weighted score
        final_score = (raw_score / total_weight) * asset_multiplier

        # Clamp to 0-10 range
        final_score = max(0.0, min(10.0, final_score))

        # Record factors
        factors.update({
            "epss_score": epss_score,
            "is_kev": is_kev,
            "asset_multiplier": asset_multiplier,
        })

        calculation_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return RiskScoreResult(
            final_score=round(final_score, 2),
            cvss_component=round(cvss_component, 3),
            epss_component=round(epss_component, 3),
            kev_component=round(kev_component, 3),
            asset_component=round(asset_multiplier, 2),
            factors=factors,
            calculation_time_ms=round(calculation_time, 2),
        )

    async def batch_calculate(
        self,
        vulnerabilities: List[Dict[str, Any]],
    ) -> List[RiskScoreResult]:
        """Calculate risk scores for multiple vulnerabilities.

        More efficient than individual calls when processing scan results.

        Args:
            vulnerabilities: List of vulnerability dicts with scoring data.

        Returns:
            List of RiskScoreResult in same order as input.
        """
        # Pre-fetch all EPSS data
        cves = [v.get("cve") for v in vulnerabilities if v.get("cve")]
        if cves:
            await self._bulk_fetch_epss(cves)

        # Calculate scores
        results = []
        for vuln in vulnerabilities:
            result = await self.calculate_risk_score(
                cvss_score=vuln.get("cvss_score"),
                cvss_vector=vuln.get("cvss_vector"),
                cve=vuln.get("cve"),
                epss_score=vuln.get("epss_score"),
                is_kev=vuln.get("is_kev"),
                asset_criticality=vuln.get("asset_criticality", "medium"),
            )
            results.append(result)

        return results

    # -------------------------------------------------------------------------
    # CVSS Parsing
    # -------------------------------------------------------------------------

    def _parse_cvss_vector(self, vector: str) -> float:
        """Parse CVSS 3.1 vector string to base score.

        Args:
            vector: CVSS vector string (e.g., "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")

        Returns:
            Calculated CVSS base score.
        """
        try:
            # Parse vector components
            parts = vector.upper().split("/")
            components = {}

            for part in parts:
                if ":" in part:
                    key, value = part.split(":", 1)
                    components[key] = value

            # Extract metrics
            av = components.get("AV", "N")
            ac = components.get("AC", "L")
            pr = components.get("PR", "N")
            ui = components.get("UI", "N")
            scope = components.get("S", "U")
            c = components.get("C", "N")
            i = components.get("I", "N")
            a = components.get("A", "N")

            # CVSS 3.1 metric values
            av_values = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
            ac_values = {"L": 0.77, "H": 0.44}
            pr_values_unchanged = {"N": 0.85, "L": 0.62, "H": 0.27}
            pr_values_changed = {"N": 0.85, "L": 0.68, "H": 0.50}
            ui_values = {"N": 0.85, "R": 0.62}
            cia_values = {"H": 0.56, "L": 0.22, "N": 0}

            # Calculate exploitability
            av_score = av_values.get(av, 0.85)
            ac_score = ac_values.get(ac, 0.77)

            if scope == "C":
                pr_score = pr_values_changed.get(pr, 0.85)
            else:
                pr_score = pr_values_unchanged.get(pr, 0.85)

            ui_score = ui_values.get(ui, 0.85)

            exploitability = 8.22 * av_score * ac_score * pr_score * ui_score

            # Calculate impact
            c_score = cia_values.get(c, 0)
            i_score = cia_values.get(i, 0)
            a_score = cia_values.get(a, 0)

            isc_base = 1 - ((1 - c_score) * (1 - i_score) * (1 - a_score))

            if scope == "U":
                impact = 6.42 * isc_base
            else:
                impact = 7.52 * (isc_base - 0.029) - 3.25 * ((isc_base - 0.02) ** 15)

            # Calculate base score
            if impact <= 0:
                return 0.0

            if scope == "U":
                base_score = min(impact + exploitability, 10)
            else:
                base_score = min(1.08 * (impact + exploitability), 10)

            # Round up to nearest 0.1
            return round(base_score, 1)

        except Exception as e:
            logger.warning("Failed to parse CVSS vector '%s': %s", vector, e)
            return 5.0  # Default to medium

    # -------------------------------------------------------------------------
    # EPSS Integration
    # -------------------------------------------------------------------------

    async def _fetch_epss(self, cve: str) -> Optional[EPSSData]:
        """Fetch EPSS data for a single CVE.

        Args:
            cve: CVE identifier.

        Returns:
            EPSSData or None if not found.
        """
        # Check cache first
        if self._config.enable_caching and cve in self._epss_cache:
            cached = self._epss_cache[cve]
            if datetime.utcnow() - cached.date < timedelta(hours=self._config.cache_ttl_hours):
                return cached

        if not self._http_client:
            logger.debug("HTTP client not configured, skipping EPSS fetch")
            return None

        try:
            url = f"{self._config.epss_api_url}?cve={cve}"
            response = await self._http_client.get(url, timeout=10)

            if response.status_code != 200:
                logger.warning("EPSS API returned %d for %s", response.status_code, cve)
                return None

            data = response.json()
            if not data.get("data"):
                return None

            epss_entry = data["data"][0]
            epss_data = EPSSData(
                cve=cve,
                score=float(epss_entry.get("epss", 0)),
                percentile=float(epss_entry.get("percentile", 0)),
                date=datetime.utcnow(),
            )

            # Cache the result
            if self._config.enable_caching:
                self._epss_cache[cve] = epss_data

            return epss_data

        except Exception as e:
            logger.warning("Failed to fetch EPSS for %s: %s", cve, e)
            return None

    async def _bulk_fetch_epss(self, cves: List[str]) -> Dict[str, EPSSData]:
        """Fetch EPSS data for multiple CVEs.

        Args:
            cves: List of CVE identifiers.

        Returns:
            Dictionary mapping CVE to EPSSData.
        """
        results: Dict[str, EPSSData] = {}

        # Check cache for already-fetched CVEs
        uncached = []
        for cve in cves:
            if cve in self._epss_cache:
                cached = self._epss_cache[cve]
                if datetime.utcnow() - cached.date < timedelta(hours=self._config.cache_ttl_hours):
                    results[cve] = cached
                else:
                    uncached.append(cve)
            else:
                uncached.append(cve)

        if not uncached or not self._http_client:
            return results

        try:
            # FIRST.org API supports comma-separated CVEs
            cve_list = ",".join(uncached[:100])  # API limit
            url = f"{self._config.epss_api_url}?cve={cve_list}"
            response = await self._http_client.get(url, timeout=30)

            if response.status_code != 200:
                logger.warning("EPSS bulk API returned %d", response.status_code)
                return results

            data = response.json()
            for entry in data.get("data", []):
                cve = entry.get("cve")
                if cve:
                    epss_data = EPSSData(
                        cve=cve,
                        score=float(entry.get("epss", 0)),
                        percentile=float(entry.get("percentile", 0)),
                        date=datetime.utcnow(),
                    )
                    results[cve] = epss_data
                    if self._config.enable_caching:
                        self._epss_cache[cve] = epss_data

        except Exception as e:
            logger.warning("Failed to bulk fetch EPSS: %s", e)

        return results

    # -------------------------------------------------------------------------
    # CISA KEV Integration
    # -------------------------------------------------------------------------

    async def _check_kev(self, cve: str) -> tuple[bool, Optional[KEVEntry]]:
        """Check if CVE is in CISA KEV catalog.

        Args:
            cve: CVE identifier.

        Returns:
            Tuple of (is_kev, KEVEntry or None).
        """
        # Refresh KEV data if needed
        await self._refresh_kev_data()

        is_in_kev = cve.upper() in self._kev_set
        entry = self._kev_data.get(cve.upper())

        return is_in_kev, entry

    async def _refresh_kev_data(self) -> None:
        """Refresh CISA KEV data from feed."""
        # Check if cache is still valid
        if self._cache_updated_at:
            age = datetime.utcnow() - self._cache_updated_at
            if age < timedelta(hours=self._config.cache_ttl_hours):
                return

        if not self._http_client:
            logger.debug("HTTP client not configured, skipping KEV refresh")
            return

        try:
            response = await self._http_client.get(
                self._config.kev_feed_url,
                timeout=30,
            )

            if response.status_code != 200:
                logger.warning("KEV feed returned %d", response.status_code)
                return

            data = response.json()
            vulnerabilities = data.get("vulnerabilities", [])

            # Build KEV set and data dict
            new_kev_set: Set[str] = set()
            new_kev_data: Dict[str, KEVEntry] = {}

            for vuln in vulnerabilities:
                cve = vuln.get("cveID", "").upper()
                if cve:
                    new_kev_set.add(cve)
                    try:
                        new_kev_data[cve] = KEVEntry(
                            cve=cve,
                            vendor=vuln.get("vendorProject", ""),
                            product=vuln.get("product", ""),
                            vulnerability_name=vuln.get("vulnerabilityName", ""),
                            date_added=datetime.fromisoformat(
                                vuln.get("dateAdded", "2020-01-01")
                            ),
                            due_date=datetime.fromisoformat(
                                vuln.get("dueDate", "2099-12-31")
                            ),
                            short_description=vuln.get("shortDescription", ""),
                            required_action=vuln.get("requiredAction", ""),
                        )
                    except Exception as parse_err:
                        logger.debug("Failed to parse KEV entry %s: %s", cve, parse_err)

            self._kev_set = new_kev_set
            self._kev_data = new_kev_data
            self._cache_updated_at = datetime.utcnow()

            logger.info("Refreshed KEV data: %d vulnerabilities", len(new_kev_set))

        except Exception as e:
            logger.warning("Failed to refresh KEV data: %s", e)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def severity_to_score(self, severity: str) -> float:
        """Convert severity string to representative CVSS score.

        Args:
            severity: Severity level (critical, high, medium, low, info).

        Returns:
            Representative CVSS score.
        """
        mapping = {
            "critical": 9.5,
            "high": 7.5,
            "medium": 5.0,
            "low": 2.5,
            "info": 0.5,
        }
        return mapping.get(severity.lower(), 5.0)

    def score_to_severity(self, score: float) -> str:
        """Convert CVSS score to severity string.

        Args:
            score: CVSS score (0-10).

        Returns:
            Severity level string.
        """
        if score >= 9.0:
            return "critical"
        elif score >= 7.0:
            return "high"
        elif score >= 4.0:
            return "medium"
        elif score >= 0.1:
            return "low"
        else:
            return "info"

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._epss_cache.clear()
        self._kev_set.clear()
        self._kev_data.clear()
        self._cache_updated_at = None
        logger.info("Risk scorer cache cleared")


# ---------------------------------------------------------------------------
# Global Instance
# ---------------------------------------------------------------------------

_global_risk_scorer: Optional[RiskScorer] = None


def get_risk_scorer() -> Optional[RiskScorer]:
    """Get the global risk scorer instance.

    Returns:
        The global RiskScorer or None if not configured.
    """
    return _global_risk_scorer


def configure_risk_scorer(
    config: Optional[RiskScorerConfig] = None,
    http_client: Optional[Any] = None,
) -> RiskScorer:
    """Configure the global risk scorer.

    Args:
        config: Scoring configuration.
        http_client: Optional async HTTP client.

    Returns:
        The configured RiskScorer.
    """
    global _global_risk_scorer

    _global_risk_scorer = RiskScorer(
        config=config,
        http_client=http_client,
    )

    logger.info("RiskScorer configured")
    return _global_risk_scorer


__all__ = [
    "RiskScorer",
    "RiskScorerConfig",
    "RiskScoreResult",
    "AssetCriticality",
    "CVSSVector",
    "EPSSData",
    "KEVEntry",
    "get_risk_scorer",
    "configure_risk_scorer",
]
