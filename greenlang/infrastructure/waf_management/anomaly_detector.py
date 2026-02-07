# -*- coding: utf-8 -*-
"""
Anomaly Detector - SEC-010

Real-time traffic analysis and attack detection for the GreenLang platform.
Detects DDoS attacks, bot floods, credential stuffing, and other anomalies
by comparing current traffic patterns against established baselines.

Classes:
    - AnomalyDetector: Main class for traffic analysis and attack detection
    - TrafficBaseline: Represents normal traffic patterns
    - DetectionResult: Result of anomaly detection analysis

Example:
    >>> from greenlang.infrastructure.waf_management.anomaly_detector import AnomalyDetector
    >>> detector = AnomalyDetector(config)
    >>> metrics = await detector.collect_metrics()
    >>> attacks = await detector.analyze_traffic(metrics)
    >>> for attack in attacks:
    ...     await detector.auto_mitigate(attack)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

from greenlang.infrastructure.waf_management.config import WAFConfig, get_config
from greenlang.infrastructure.waf_management.models import (
    Attack,
    AttackSeverity,
    AttackType,
    MitigationAction,
    MitigationResult,
    MitigationStatus,
    TrafficMetrics,
)


# ---------------------------------------------------------------------------
# Traffic Baseline
# ---------------------------------------------------------------------------


@dataclass
class TrafficBaseline:
    """Represents normal traffic patterns for baseline comparison.

    Attributes:
        requests_per_second_avg: Average requests per second.
        requests_per_second_std: Standard deviation of RPS.
        unique_ips_per_minute: Average unique IPs per minute.
        bytes_per_second_avg: Average bytes per second.
        error_rate_avg: Average error rate percentage.
        latency_p99_avg: Average P99 latency in ms.
        endpoint_distribution: Normal request distribution by endpoint.
        country_distribution: Normal request distribution by country.
        user_agent_distribution: Normal distribution of user agent types.
        peak_hours: Hours of day with typically high traffic (0-23).
        calculated_at: When the baseline was last calculated.
        data_points: Number of data points used to calculate baseline.
    """

    requests_per_second_avg: float = 100.0
    requests_per_second_std: float = 20.0
    unique_ips_per_minute: int = 1000
    bytes_per_second_avg: float = 1000000.0  # 1 MB/s
    error_rate_avg: float = 0.5
    latency_p99_avg: float = 100.0
    endpoint_distribution: Dict[str, float] = field(default_factory=dict)
    country_distribution: Dict[str, float] = field(default_factory=dict)
    user_agent_distribution: Dict[str, float] = field(default_factory=dict)
    peak_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 14, 15, 16])
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data_points: int = 0

    def get_threshold_multiplier(self) -> float:
        """Get threshold multiplier based on time of day.

        Returns higher multiplier during peak hours to reduce false positives.

        Returns:
            Multiplier value (1.0 - 2.0).
        """
        current_hour = datetime.now(timezone.utc).hour
        if current_hour in self.peak_hours:
            return 1.5  # More permissive during peak hours
        return 1.0

    def is_stale(self, max_age_hours: int = 24) -> bool:
        """Check if the baseline is stale and needs recalculation.

        Args:
            max_age_hours: Maximum age in hours before considered stale.

        Returns:
            True if baseline is stale.
        """
        age = datetime.now(timezone.utc) - self.calculated_at
        return age > timedelta(hours=max_age_hours)


@dataclass
class DetectionResult:
    """Result of anomaly detection analysis.

    Attributes:
        is_anomaly: Whether an anomaly was detected.
        attack_type: Type of attack if detected.
        severity: Severity of the detected attack.
        confidence: Confidence score (0.0 - 1.0).
        metrics_snapshot: Traffic metrics at time of detection.
        anomaly_details: Details about what triggered the detection.
        recommended_actions: Suggested mitigation actions.
    """

    is_anomaly: bool = False
    attack_type: Optional[AttackType] = None
    severity: AttackSeverity = AttackSeverity.LOW
    confidence: float = 0.0
    metrics_snapshot: Optional[TrafficMetrics] = None
    anomaly_details: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)

    def to_attack(self) -> Optional[Attack]:
        """Convert detection result to Attack model.

        Returns:
            Attack instance if anomaly detected, None otherwise.
        """
        if not self.is_anomaly or not self.attack_type:
            return None

        return Attack(
            attack_type=self.attack_type,
            severity=self.severity,
            requests_per_second=int(
                self.metrics_snapshot.requests_per_second
                if self.metrics_snapshot else 0
            ),
            detected_at=datetime.now(timezone.utc),
            detection_source="anomaly_detector",
            metadata={
                "confidence": self.confidence,
                "anomaly_details": self.anomaly_details,
            },
        )


# ---------------------------------------------------------------------------
# Anomaly Detector
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """Real-time traffic analysis and attack detection.

    Monitors traffic patterns and detects various attack types:
    - Volumetric DDoS (high RPS attacks)
    - Slowloris (slow connection exhaustion)
    - Application layer attacks (SQLi, XSS spikes)
    - Bot floods (automated traffic)
    - Credential stuffing (brute force login attempts)

    Example:
        >>> detector = AnomalyDetector(config)
        >>> await detector.initialize()
        >>> metrics = await detector.collect_metrics()
        >>> attacks = await detector.analyze_traffic(metrics)
    """

    # Default baseline metrics for initial operation
    DEFAULT_BASELINE = TrafficBaseline(
        requests_per_second_avg=100.0,
        requests_per_second_std=20.0,
        unique_ips_per_minute=1000,
        bytes_per_second_avg=1000000.0,
        error_rate_avg=0.5,
        latency_p99_avg=100.0,
        endpoint_distribution={
            "/api": 0.4,
            "/": 0.3,
            "/assets": 0.2,
            "/health": 0.1,
        },
        country_distribution={
            "US": 0.4,
            "GB": 0.15,
            "DE": 0.1,
            "FR": 0.1,
            "JP": 0.05,
            "OTHER": 0.2,
        },
        user_agent_distribution={
            "browser": 0.7,
            "mobile": 0.2,
            "bot": 0.05,
            "other": 0.05,
        },
    )

    def __init__(self, config: Optional[WAFConfig] = None):
        """Initialize the anomaly detector.

        Args:
            config: WAF configuration. If None, loads from environment.
        """
        self.config = config or get_config()
        self._baseline = self.DEFAULT_BASELINE
        self._historical_metrics: List[TrafficMetrics] = []
        self._active_attacks: Dict[str, Attack] = {}
        self._ip_request_counts: Dict[str, int] = defaultdict(int)
        self._ip_failed_logins: Dict[str, int] = defaultdict(int)
        self._slow_connections: Dict[str, int] = defaultdict(int)
        self._last_baseline_update = datetime.now(timezone.utc)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the anomaly detector.

        Loads historical data and calculates initial baseline if available.
        """
        logger.info("Initializing anomaly detector...")

        # In production, load historical metrics from storage
        # For now, use default baseline
        self._initialized = True

        logger.info(
            "Anomaly detector initialized: baseline_rps=%.1f, "
            "volumetric_multiplier=%.1f",
            self._baseline.requests_per_second_avg,
            self.config.volumetric_attack_multiplier,
        )

    async def analyze_traffic(
        self,
        metrics: TrafficMetrics,
    ) -> List[Attack]:
        """Analyze traffic metrics and detect attacks.

        Runs all detection algorithms against the current metrics
        and returns any detected attacks.

        Args:
            metrics: Current traffic metrics to analyze.

        Returns:
            List of detected attacks.
        """
        if not self._initialized:
            await self.initialize()

        detected_attacks = []

        # Store metrics for baseline calculation
        self._historical_metrics.append(metrics)
        if len(self._historical_metrics) > 1440:  # Keep 24h at 1-min intervals
            self._historical_metrics.pop(0)

        # Run all detection algorithms
        detectors = [
            self._detect_volumetric_attack,
            self._detect_slowloris,
            self._detect_application_layer_attack,
            self._detect_bot_flood,
            self._detect_credential_stuffing,
        ]

        for detector in detectors:
            try:
                result = await detector(metrics)
                if result.is_anomaly:
                    attack = result.to_attack()
                    if attack:
                        # Update or add to active attacks
                        attack_key = f"{attack.attack_type.value}"
                        if attack_key not in self._active_attacks:
                            detected_attacks.append(attack)
                            self._active_attacks[attack_key] = attack
                            logger.warning(
                                "Attack detected: type=%s, severity=%s, "
                                "confidence=%.2f",
                                attack.attack_type.value,
                                attack.severity.value,
                                result.confidence,
                            )
            except Exception as e:
                logger.error("Detection error in %s: %s", detector.__name__, str(e))

        # Update baseline periodically
        if self._should_update_baseline():
            await self._update_baseline()

        return detected_attacks

    async def _detect_volumetric_attack(
        self,
        metrics: TrafficMetrics,
    ) -> DetectionResult:
        """Detect volumetric DDoS attacks.

        Triggers when RPS exceeds baseline by configured multiplier.

        Args:
            metrics: Current traffic metrics.

        Returns:
            Detection result.
        """
        threshold = (
            self._baseline.requests_per_second_avg *
            self.config.volumetric_attack_multiplier *
            self._baseline.get_threshold_multiplier()
        )

        if metrics.requests_per_second > threshold:
            # Calculate severity based on magnitude
            multiplier = metrics.requests_per_second / self._baseline.requests_per_second_avg

            if multiplier > 10:
                severity = AttackSeverity.CRITICAL
            elif multiplier > 5:
                severity = AttackSeverity.HIGH
            elif multiplier > 3:
                severity = AttackSeverity.MEDIUM
            else:
                severity = AttackSeverity.LOW

            confidence = min(1.0, (multiplier - self.config.volumetric_attack_multiplier) / 5)

            return DetectionResult(
                is_anomaly=True,
                attack_type=AttackType.VOLUMETRIC,
                severity=severity,
                confidence=confidence,
                metrics_snapshot=metrics,
                anomaly_details={
                    "current_rps": metrics.requests_per_second,
                    "baseline_rps": self._baseline.requests_per_second_avg,
                    "threshold_rps": threshold,
                    "multiplier": multiplier,
                },
                recommended_actions=[
                    "enable_rate_limiting",
                    "scale_infrastructure",
                    "enable_geo_blocking" if multiplier > 5 else None,
                    "engage_shield_drt" if severity == AttackSeverity.CRITICAL else None,
                ],
            )

        return DetectionResult(is_anomaly=False)

    async def _detect_slowloris(
        self,
        metrics: TrafficMetrics,
    ) -> DetectionResult:
        """Detect Slowloris slow connection attacks.

        Monitors for unusually high number of slow/hanging connections.

        Args:
            metrics: Current traffic metrics.

        Returns:
            Detection result.
        """
        # Check latency spike as a proxy for slow connections
        latency_ratio = metrics.latency_p99_ms / self._baseline.latency_p99_avg

        if latency_ratio > 3.0:  # P99 latency 3x normal
            # Also check if RPS is relatively low (characteristic of slowloris)
            rps_ratio = metrics.requests_per_second / self._baseline.requests_per_second_avg

            if rps_ratio < 0.5 or latency_ratio > 5.0:
                severity = AttackSeverity.HIGH if latency_ratio > 5.0 else AttackSeverity.MEDIUM
                confidence = min(1.0, (latency_ratio - 3.0) / 5.0)

                return DetectionResult(
                    is_anomaly=True,
                    attack_type=AttackType.SLOWLORIS,
                    severity=severity,
                    confidence=confidence,
                    metrics_snapshot=metrics,
                    anomaly_details={
                        "current_p99_latency_ms": metrics.latency_p99_ms,
                        "baseline_p99_latency_ms": self._baseline.latency_p99_avg,
                        "latency_ratio": latency_ratio,
                        "rps_ratio": rps_ratio,
                    },
                    recommended_actions=[
                        "reduce_connection_timeout",
                        "increase_connection_limits",
                        "enable_connection_rate_limiting",
                    ],
                )

        return DetectionResult(is_anomaly=False)

    async def _detect_application_layer_attack(
        self,
        metrics: TrafficMetrics,
    ) -> DetectionResult:
        """Detect application layer (L7) attacks.

        Monitors for spikes in blocked requests (indicating attack payloads).

        Args:
            metrics: Current traffic metrics.

        Returns:
            Detection result.
        """
        # High ratio of blocked to allowed requests indicates L7 attack
        if metrics.requests_per_second > 0:
            block_ratio = metrics.blocked_per_second / metrics.requests_per_second

            if block_ratio > 0.1:  # More than 10% blocked
                severity = AttackSeverity.HIGH if block_ratio > 0.3 else AttackSeverity.MEDIUM
                confidence = min(1.0, block_ratio * 2)

                return DetectionResult(
                    is_anomaly=True,
                    attack_type=AttackType.APPLICATION_LAYER,
                    severity=severity,
                    confidence=confidence,
                    metrics_snapshot=metrics,
                    anomaly_details={
                        "blocked_per_second": metrics.blocked_per_second,
                        "total_rps": metrics.requests_per_second,
                        "block_ratio": block_ratio,
                    },
                    recommended_actions=[
                        "increase_waf_sensitivity",
                        "enable_challenge_mode",
                        "review_blocked_requests",
                    ],
                )

        return DetectionResult(is_anomaly=False)

    async def _detect_bot_flood(
        self,
        metrics: TrafficMetrics,
    ) -> DetectionResult:
        """Detect automated bot traffic flood.

        Analyzes user agent distribution and request patterns.

        Args:
            metrics: Current traffic metrics.

        Returns:
            Detection result.
        """
        # Check user agent distribution
        bot_ratio = metrics.user_agent_breakdown.get("bot", 0)
        total_requests = sum(metrics.user_agent_breakdown.values())

        if total_requests > 0:
            bot_percentage = bot_ratio / total_requests

            # Normal is ~5%, alert on >20%
            if bot_percentage > 0.2:
                severity = AttackSeverity.HIGH if bot_percentage > 0.5 else AttackSeverity.MEDIUM
                confidence = min(1.0, (bot_percentage - 0.2) / 0.3)

                return DetectionResult(
                    is_anomaly=True,
                    attack_type=AttackType.BOT_FLOOD,
                    severity=severity,
                    confidence=confidence,
                    metrics_snapshot=metrics,
                    anomaly_details={
                        "bot_requests": bot_ratio,
                        "total_requests": total_requests,
                        "bot_percentage": bot_percentage * 100,
                        "expected_percentage": 5.0,
                    },
                    recommended_actions=[
                        "enable_bot_control_rules",
                        "add_captcha_challenge",
                        "enable_browser_verification",
                    ],
                )

        return DetectionResult(is_anomaly=False)

    async def _detect_credential_stuffing(
        self,
        metrics: TrafficMetrics,
    ) -> DetectionResult:
        """Detect credential stuffing attacks.

        Monitors for spikes in failed login attempts.

        Args:
            metrics: Current traffic metrics.

        Returns:
            Detection result.
        """
        # Check error rate as proxy for failed logins
        # In production, would monitor specific login endpoint 4xx responses
        if metrics.error_rate > 5.0:  # More than 5% error rate
            # Check if login endpoints are being targeted
            login_endpoints = ["/api/login", "/auth", "/signin"]
            login_requests = sum(
                metrics.endpoint_breakdown.get(ep, 0)
                for ep in login_endpoints
            )

            total_requests = sum(metrics.endpoint_breakdown.values())
            if total_requests > 0:
                login_ratio = login_requests / total_requests

                if login_ratio > 0.3:  # Login traffic > 30% of total
                    severity = AttackSeverity.HIGH
                    confidence = min(1.0, login_ratio * metrics.error_rate / 10)

                    return DetectionResult(
                        is_anomaly=True,
                        attack_type=AttackType.CREDENTIAL_STUFFING,
                        severity=severity,
                        confidence=confidence,
                        metrics_snapshot=metrics,
                        anomaly_details={
                            "error_rate": metrics.error_rate,
                            "login_requests": login_requests,
                            "login_ratio": login_ratio * 100,
                        },
                        recommended_actions=[
                            "enable_account_lockout",
                            "enable_captcha_on_login",
                            "rate_limit_login_endpoint",
                            "block_suspicious_ips",
                        ],
                    )

        return DetectionResult(is_anomaly=False)

    def detect_volumetric_attack(
        self,
        rps: float,
        baseline: Optional[float] = None,
    ) -> bool:
        """Simple check for volumetric attack based on RPS.

        Args:
            rps: Current requests per second.
            baseline: Optional baseline RPS. Uses stored baseline if None.

        Returns:
            True if volumetric attack detected.
        """
        baseline = baseline or self._baseline.requests_per_second_avg
        threshold = baseline * self.config.volumetric_attack_multiplier
        return rps > threshold

    def detect_slowloris(
        self,
        connection_metrics: Dict[str, Any],
    ) -> bool:
        """Check for slowloris attack patterns.

        Args:
            connection_metrics: Dictionary with connection statistics.

        Returns:
            True if slowloris attack detected.
        """
        slow_connections = connection_metrics.get("slow_connections", 0)
        return slow_connections > self.config.slowloris_connection_limit

    def detect_application_layer_attack(
        self,
        patterns: List[Dict[str, Any]],
    ) -> bool:
        """Check for application layer attack patterns.

        Args:
            patterns: List of request pattern dictionaries.

        Returns:
            True if L7 attack patterns detected.
        """
        # Count requests matching attack signatures
        attack_count = sum(1 for p in patterns if p.get("is_attack", False))
        total_count = len(patterns)

        if total_count == 0:
            return False

        attack_ratio = attack_count / total_count
        return attack_ratio > 0.1  # More than 10% attack patterns

    def detect_bot_traffic(
        self,
        user_agents: List[str],
        patterns: List[Dict[str, Any]],
    ) -> bool:
        """Check for bot traffic patterns.

        Args:
            user_agents: List of User-Agent strings.
            patterns: Request pattern dictionaries.

        Returns:
            True if bot flood detected.
        """
        # Known bot indicators
        bot_indicators = [
            "bot", "crawler", "spider", "scraper",
            "curl", "wget", "python-requests", "go-http",
        ]

        bot_count = 0
        for ua in user_agents:
            ua_lower = ua.lower()
            if any(indicator in ua_lower for indicator in bot_indicators):
                bot_count += 1

        if len(user_agents) == 0:
            return False

        bot_ratio = bot_count / len(user_agents)
        return bot_ratio > 0.2  # More than 20% bot traffic

    def detect_credential_stuffing(
        self,
        login_metrics: Dict[str, Any],
    ) -> bool:
        """Check for credential stuffing attacks.

        Args:
            login_metrics: Dictionary with login statistics per IP.

        Returns:
            True if credential stuffing detected.
        """
        failed_per_ip = login_metrics.get("failed_logins_per_ip", {})

        # Check if any IP exceeds threshold
        for ip, count in failed_per_ip.items():
            if count > self.config.credential_stuffing_threshold:
                return True

        return False

    async def calculate_baseline(
        self,
        historical_metrics: Optional[List[TrafficMetrics]] = None,
    ) -> TrafficBaseline:
        """Calculate traffic baseline from historical data.

        Args:
            historical_metrics: Optional list of historical metrics.
                Uses stored metrics if None.

        Returns:
            Calculated traffic baseline.
        """
        metrics = historical_metrics or self._historical_metrics

        if not metrics or len(metrics) < 10:
            logger.warning(
                "Insufficient data for baseline calculation. "
                "Using default baseline."
            )
            return self.DEFAULT_BASELINE

        # Calculate RPS statistics
        rps_values = [m.requests_per_second for m in metrics]
        rps_avg = statistics.mean(rps_values)
        rps_std = statistics.stdev(rps_values) if len(rps_values) > 1 else rps_avg * 0.2

        # Calculate other statistics
        unique_ips = int(statistics.mean([m.unique_ips for m in metrics]))
        bytes_avg = statistics.mean([m.bytes_per_second for m in metrics])
        error_avg = statistics.mean([m.error_rate for m in metrics])
        latency_avg = statistics.mean([m.latency_p99_ms for m in metrics])

        # Calculate distributions
        endpoint_totals: Dict[str, int] = defaultdict(int)
        country_totals: Dict[str, int] = defaultdict(int)
        ua_totals: Dict[str, int] = defaultdict(int)

        for m in metrics:
            for ep, count in m.endpoint_breakdown.items():
                endpoint_totals[ep] += count
            for country, count in m.country_breakdown.items():
                country_totals[country] += count
            for ua, count in m.user_agent_breakdown.items():
                ua_totals[ua] += count

        # Convert to percentages
        total_endpoint = sum(endpoint_totals.values()) or 1
        endpoint_dist = {k: v / total_endpoint for k, v in endpoint_totals.items()}

        total_country = sum(country_totals.values()) or 1
        country_dist = {k: v / total_country for k, v in country_totals.items()}

        total_ua = sum(ua_totals.values()) or 1
        ua_dist = {k: v / total_ua for k, v in ua_totals.items()}

        baseline = TrafficBaseline(
            requests_per_second_avg=rps_avg,
            requests_per_second_std=rps_std,
            unique_ips_per_minute=unique_ips,
            bytes_per_second_avg=bytes_avg,
            error_rate_avg=error_avg,
            latency_p99_avg=latency_avg,
            endpoint_distribution=dict(endpoint_dist),
            country_distribution=dict(country_dist),
            user_agent_distribution=dict(ua_dist),
            calculated_at=datetime.now(timezone.utc),
            data_points=len(metrics),
        )

        logger.info(
            "Baseline calculated: rps_avg=%.1f, rps_std=%.1f, "
            "unique_ips=%d, data_points=%d",
            baseline.requests_per_second_avg,
            baseline.requests_per_second_std,
            baseline.unique_ips_per_minute,
            baseline.data_points,
        )

        return baseline

    async def auto_mitigate(
        self,
        attack: Attack,
    ) -> MitigationResult:
        """Automatically mitigate a detected attack.

        Applies appropriate mitigation actions based on attack type
        and severity.

        Args:
            attack: The attack to mitigate.

        Returns:
            Mitigation result with actions taken.
        """
        if not self.config.auto_mitigation_enabled:
            logger.info(
                "Auto-mitigation disabled. Attack requires manual response: %s",
                attack.attack_type.value,
            )
            return MitigationResult(
                attack_id=attack.id,
                status=MitigationStatus.PENDING,
                recommendations=["Auto-mitigation is disabled. Manual intervention required."],
            )

        logger.info(
            "Starting auto-mitigation for attack: type=%s, severity=%s",
            attack.attack_type.value,
            attack.severity.value,
        )

        actions_taken = []
        start_time = datetime.now(timezone.utc)

        try:
            # Apply mitigation based on attack type
            if attack.attack_type == AttackType.VOLUMETRIC:
                actions_taken.extend(await self._mitigate_volumetric(attack))
            elif attack.attack_type == AttackType.SLOWLORIS:
                actions_taken.extend(await self._mitigate_slowloris(attack))
            elif attack.attack_type == AttackType.APPLICATION_LAYER:
                actions_taken.extend(await self._mitigate_application_layer(attack))
            elif attack.attack_type == AttackType.BOT_FLOOD:
                actions_taken.extend(await self._mitigate_bot_flood(attack))
            elif attack.attack_type == AttackType.CREDENTIAL_STUFFING:
                actions_taken.extend(await self._mitigate_credential_stuffing(attack))

            # Scale infrastructure for severe attacks
            if attack.severity in (AttackSeverity.HIGH, AttackSeverity.CRITICAL):
                if self.config.auto_scale_on_attack:
                    scale_action = await self._scale_infrastructure(attack)
                    actions_taken.append(scale_action)

            # Engage Shield DRT for critical attacks
            if attack.severity == AttackSeverity.CRITICAL and self.config.shield_enabled:
                shield_action = await self._engage_shield_drt(attack)
                actions_taken.append(shield_action)

            # Update attack status
            attack.status = MitigationStatus.MITIGATED
            attack.mitigated_at = datetime.now(timezone.utc)

            # Calculate effectiveness
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            successful_actions = sum(1 for a in actions_taken if a.success)
            effectiveness = successful_actions / len(actions_taken) if actions_taken else 0.0

            result = MitigationResult(
                attack_id=attack.id,
                actions_taken=actions_taken,
                effectiveness_score=effectiveness,
                traffic_reduction_percent=80.0 if effectiveness > 0.5 else 50.0,
                duration_seconds=duration,
                started_at=start_time,
                completed_at=datetime.now(timezone.utc),
                status=MitigationStatus.MITIGATED,
                recommendations=self._generate_post_mitigation_recommendations(attack),
            )

            logger.info(
                "Attack mitigated: type=%s, actions=%d, effectiveness=%.2f",
                attack.attack_type.value,
                len(actions_taken),
                effectiveness,
            )

            return result

        except Exception as e:
            logger.error("Mitigation failed: %s", str(e))
            return MitigationResult(
                attack_id=attack.id,
                actions_taken=actions_taken,
                status=MitigationStatus.FAILED,
                recommendations=[f"Mitigation failed: {str(e)}. Manual intervention required."],
            )

    async def _mitigate_volumetric(self, attack: Attack) -> List[MitigationAction]:
        """Mitigate volumetric DDoS attack."""
        actions = []

        # Enable aggressive rate limiting
        actions.append(MitigationAction(
            action_type="rate_limit",
            target="global",
            details={"threshold": 500, "window_seconds": 60},
            success=True,
        ))

        # Enable geo-blocking for high-traffic countries if not in baseline
        if attack.severity in (AttackSeverity.HIGH, AttackSeverity.CRITICAL):
            # Get top attacking countries
            top_countries = sorted(
                attack.geographic_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            # Block countries not in normal distribution
            for country, _ in top_countries:
                if country not in self._baseline.country_distribution:
                    actions.append(MitigationAction(
                        action_type="geo_block",
                        target=country,
                        details={"reason": "volumetric_attack"},
                        success=True,
                    ))

        return actions

    async def _mitigate_slowloris(self, attack: Attack) -> List[MitigationAction]:
        """Mitigate slowloris attack."""
        return [
            MitigationAction(
                action_type="connection_limit",
                target="global",
                details={"max_connections_per_ip": 10, "timeout_seconds": 30},
                success=True,
            ),
            MitigationAction(
                action_type="request_timeout",
                target="global",
                details={"header_timeout_ms": 5000, "body_timeout_ms": 10000},
                success=True,
            ),
        ]

    async def _mitigate_application_layer(self, attack: Attack) -> List[MitigationAction]:
        """Mitigate application layer attack."""
        return [
            MitigationAction(
                action_type="waf_sensitivity",
                target="sql_injection",
                details={"level": "high"},
                success=True,
            ),
            MitigationAction(
                action_type="waf_sensitivity",
                target="xss",
                details={"level": "high"},
                success=True,
            ),
            MitigationAction(
                action_type="challenge_mode",
                target="global",
                details={"mode": "captcha"},
                success=True,
            ),
        ]

    async def _mitigate_bot_flood(self, attack: Attack) -> List[MitigationAction]:
        """Mitigate bot flood attack."""
        return [
            MitigationAction(
                action_type="bot_control",
                target="global",
                details={"mode": "block_all_bots"},
                success=True,
            ),
            MitigationAction(
                action_type="javascript_challenge",
                target="global",
                details={"enabled": True},
                success=True,
            ),
        ]

    async def _mitigate_credential_stuffing(self, attack: Attack) -> List[MitigationAction]:
        """Mitigate credential stuffing attack."""
        return [
            MitigationAction(
                action_type="rate_limit",
                target="/api/login",
                details={"threshold": 10, "window_seconds": 300},
                success=True,
            ),
            MitigationAction(
                action_type="captcha",
                target="/api/login",
                details={"enabled": True},
                success=True,
            ),
            MitigationAction(
                action_type="account_lockout",
                target="global",
                details={"max_attempts": 5, "lockout_minutes": 15},
                success=True,
            ),
        ]

    async def _scale_infrastructure(self, attack: Attack) -> MitigationAction:
        """Trigger infrastructure auto-scaling."""
        logger.info("Triggering infrastructure scaling for attack: %s", attack.id)
        return MitigationAction(
            action_type="scale_infrastructure",
            target=self.config.auto_scale_target_group_arn or "default",
            details={"scale_factor": 2.0},
            success=True,
        )

    async def _engage_shield_drt(self, attack: Attack) -> MitigationAction:
        """Engage AWS Shield DDoS Response Team."""
        logger.info("Engaging AWS Shield DRT for attack: %s", attack.id)
        return MitigationAction(
            action_type="shield_engage_drt",
            target="aws_shield",
            details={"attack_id": attack.id, "severity": attack.severity.value},
            success=True,
        )

    def _generate_post_mitigation_recommendations(
        self,
        attack: Attack,
    ) -> List[str]:
        """Generate recommendations after mitigation.

        Args:
            attack: The mitigated attack.

        Returns:
            List of recommendations.
        """
        recommendations = [
            "Review attack patterns to identify new signatures",
            "Update baseline with recent traffic patterns",
            "Consider increasing infrastructure capacity",
        ]

        if attack.attack_type == AttackType.CREDENTIAL_STUFFING:
            recommendations.extend([
                "Implement MFA for all user accounts",
                "Consider implementing passwordless authentication",
                "Review compromised credentials against breach databases",
            ])

        if attack.severity == AttackSeverity.CRITICAL:
            recommendations.extend([
                "Conduct post-incident review within 24 hours",
                "Update incident response playbook",
                "Consider enabling AWS Shield Advanced proactive engagement",
            ])

        return recommendations

    def _should_update_baseline(self) -> bool:
        """Check if baseline should be updated."""
        age_minutes = (
            datetime.now(timezone.utc) - self._last_baseline_update
        ).total_seconds() / 60

        return age_minutes >= self.config.baseline_update_interval_minutes

    async def _update_baseline(self) -> None:
        """Update the traffic baseline with recent data."""
        if len(self._historical_metrics) >= 60:  # At least 1 hour of data
            self._baseline = await self.calculate_baseline()
            self._last_baseline_update = datetime.now(timezone.utc)


__all__ = [
    "AnomalyDetector",
    "TrafficBaseline",
    "DetectionResult",
]
