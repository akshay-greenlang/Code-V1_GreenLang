"""
GL-007 FURNACEPULSE - Hotspot Detector Tests

Unit tests for hotspot detection including:
- Hotspot detection with TMT above threshold
- Rate-of-rise calculation
- Spatial clustering detection
- Alert tier assignment (Advisory/Warning/Urgent)
- False positive scenarios

Coverage Target: >85%
"""

import pytest
import math
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Import test fixtures
from .conftest import TMTReading


class AlertTier(Enum):
    """Alert tier enumeration for hotspot severity."""
    NONE = "NONE"
    ADVISORY = "ADVISORY"
    WARNING = "WARNING"
    URGENT = "URGENT"


class TestHotspotDetectionThreshold:
    """Tests for hotspot detection based on temperature thresholds."""

    def test_detect_hotspot_above_warning_threshold(
        self, sample_tmt_readings_hotspot, alert_thresholds
    ):
        """Test hotspot detection when TMT exceeds warning threshold."""
        warning_threshold = alert_thresholds["TMT"]["WARNING"]  # 900C

        # Find readings above warning threshold
        hotspots = [
            r for r in sample_tmt_readings_hotspot
            if r.temperature_C > warning_threshold
        ]

        assert len(hotspots) >= 1
        assert hotspots[0].temperature_C > warning_threshold

    def test_detect_hotspot_above_urgent_threshold(
        self, sample_tmt_readings_critical, alert_thresholds
    ):
        """Test hotspot detection when TMT exceeds urgent threshold."""
        urgent_threshold = alert_thresholds["TMT"]["URGENT"]  # 950C

        # Find readings above urgent threshold
        critical_hotspots = [
            r for r in sample_tmt_readings_critical
            if r.temperature_C > urgent_threshold
        ]

        assert len(critical_hotspots) >= 1
        assert critical_hotspots[0].temperature_C > urgent_threshold

    def test_no_hotspot_normal_readings(
        self, sample_tmt_readings_normal, alert_thresholds
    ):
        """Test that normal readings do not trigger hotspot alert."""
        advisory_threshold = alert_thresholds["TMT"]["ADVISORY"]  # 850C

        # All readings should be below threshold
        hotspots = [
            r for r in sample_tmt_readings_normal
            if r.temperature_C > advisory_threshold
        ]

        assert len(hotspots) == 0

    def test_hotspot_exactly_at_threshold(self, alert_thresholds):
        """Test behavior when TMT is exactly at threshold."""
        warning_threshold = alert_thresholds["TMT"]["WARNING"]

        reading = TMTReading(
            tube_id="T-R1-01",
            zone="RADIANT",
            temperature_C=warning_threshold,  # Exactly at threshold
            rate_of_rise_C_min=0.5,
            design_limit_C=950.0,
            timestamp=datetime.now(),
            signal_quality="GOOD",
        )

        # At threshold should be considered a warning
        is_warning = reading.temperature_C >= warning_threshold
        assert is_warning

    @pytest.mark.parametrize(
        "temperature_C,expected_tier",
        [
            (800.0, AlertTier.NONE),
            (850.0, AlertTier.ADVISORY),
            (860.0, AlertTier.ADVISORY),
            (900.0, AlertTier.WARNING),
            (920.0, AlertTier.WARNING),
            (950.0, AlertTier.URGENT),
            (980.0, AlertTier.URGENT),
        ],
    )
    def test_alert_tier_assignment(
        self, temperature_C, expected_tier, alert_thresholds
    ):
        """Test correct alert tier assignment based on temperature."""
        thresholds = alert_thresholds["TMT"]

        if temperature_C >= thresholds["URGENT"]:
            tier = AlertTier.URGENT
        elif temperature_C >= thresholds["WARNING"]:
            tier = AlertTier.WARNING
        elif temperature_C >= thresholds["ADVISORY"]:
            tier = AlertTier.ADVISORY
        else:
            tier = AlertTier.NONE

        assert tier == expected_tier


class TestRateOfRiseCalculation:
    """Tests for TMT rate of rise calculation."""

    def test_rate_of_rise_calculation(self):
        """Test rate of rise calculation from consecutive readings."""
        # Two readings 1 minute apart
        reading1 = TMTReading(
            tube_id="T-R1-01",
            zone="RADIANT",
            temperature_C=820.0,
            rate_of_rise_C_min=0.0,
            design_limit_C=950.0,
            timestamp=datetime.now() - timedelta(minutes=1),
            signal_quality="GOOD",
        )
        reading2 = TMTReading(
            tube_id="T-R1-01",
            zone="RADIANT",
            temperature_C=825.0,
            rate_of_rise_C_min=0.0,
            design_limit_C=950.0,
            timestamp=datetime.now(),
            signal_quality="GOOD",
        )

        # Calculate rate of rise
        time_diff_min = (reading2.timestamp - reading1.timestamp).total_seconds() / 60
        rate_of_rise = (reading2.temperature_C - reading1.temperature_C) / time_diff_min

        assert abs(rate_of_rise - 5.0) < 0.1  # 5 C/min

    def test_rate_of_rise_above_urgent_threshold(
        self, sample_tmt_readings_rate_of_rise_alert, alert_thresholds
    ):
        """Test detection of urgent rate of rise."""
        urgent_threshold = alert_thresholds["RATE_OF_RISE"]["URGENT"]  # 10 C/min

        # Find readings with rate of rise above threshold
        urgent_ror = [
            r for r in sample_tmt_readings_rate_of_rise_alert
            if r.rate_of_rise_C_min > urgent_threshold
        ]

        assert len(urgent_ror) >= 1
        assert urgent_ror[0].rate_of_rise_C_min > urgent_threshold

    def test_rate_of_rise_warning_threshold(
        self, sample_tmt_readings_rate_of_rise_alert, alert_thresholds
    ):
        """Test detection of warning rate of rise."""
        warning_threshold = alert_thresholds["RATE_OF_RISE"]["WARNING"]  # 5 C/min
        urgent_threshold = alert_thresholds["RATE_OF_RISE"]["URGENT"]  # 10 C/min

        # Find readings with rate of rise in warning range
        warning_ror = [
            r for r in sample_tmt_readings_rate_of_rise_alert
            if warning_threshold <= r.rate_of_rise_C_min < urgent_threshold
        ]

        assert len(warning_ror) >= 1

    def test_negative_rate_of_rise_cooling(self):
        """Test negative rate of rise (cooling)."""
        reading = TMTReading(
            tube_id="T-R1-01",
            zone="RADIANT",
            temperature_C=820.0,
            rate_of_rise_C_min=-2.0,  # Cooling
            design_limit_C=950.0,
            timestamp=datetime.now(),
            signal_quality="GOOD",
        )

        # Negative rate of rise is normal (cooling)
        is_cooling = reading.rate_of_rise_C_min < 0
        assert is_cooling

    def test_rate_of_rise_from_history(self):
        """Test rate of rise calculation from historical readings."""
        base_time = datetime.now()
        history = [
            TMTReading(
                tube_id="T-R1-01",
                zone="RADIANT",
                temperature_C=800.0 + i * 3.0,  # Rising 3C/min
                rate_of_rise_C_min=0.0,
                design_limit_C=950.0,
                timestamp=base_time + timedelta(minutes=i),
                signal_quality="GOOD",
            )
            for i in range(10)
        ]

        # Calculate average rate of rise
        total_rise = history[-1].temperature_C - history[0].temperature_C
        total_time = (history[-1].timestamp - history[0].timestamp).total_seconds() / 60
        avg_rate = total_rise / total_time

        assert abs(avg_rate - 3.0) < 0.1

    @pytest.mark.parametrize(
        "rate_of_rise_C_min,expected_tier",
        [
            (1.0, AlertTier.NONE),
            (3.0, AlertTier.ADVISORY),
            (5.0, AlertTier.WARNING),
            (7.0, AlertTier.WARNING),
            (10.0, AlertTier.URGENT),
            (15.0, AlertTier.URGENT),
        ],
    )
    def test_rate_of_rise_tier_assignment(
        self, rate_of_rise_C_min, expected_tier, alert_thresholds
    ):
        """Test alert tier assignment based on rate of rise."""
        thresholds = alert_thresholds["RATE_OF_RISE"]

        if rate_of_rise_C_min >= thresholds["URGENT"]:
            tier = AlertTier.URGENT
        elif rate_of_rise_C_min >= thresholds["WARNING"]:
            tier = AlertTier.WARNING
        elif rate_of_rise_C_min >= thresholds["ADVISORY"]:
            tier = AlertTier.ADVISORY
        else:
            tier = AlertTier.NONE

        assert tier == expected_tier


class TestSpatialClustering:
    """Tests for spatial clustering of hotspots."""

    def test_detect_spatial_cluster(self, sample_tmt_readings_hotspot):
        """Test detection of spatially clustered hotspots."""
        # Group readings by position (x, y)
        elevated_readings = [
            r for r in sample_tmt_readings_hotspot
            if r.temperature_C > 880.0  # Elevated threshold
        ]

        # Check if elevated readings are spatially adjacent
        if len(elevated_readings) >= 2:
            r1, r2 = elevated_readings[0], elevated_readings[1]
            distance = math.sqrt(
                (r2.position_x - r1.position_x) ** 2 +
                (r2.position_y - r1.position_y) ** 2
            )

            # Adjacent tubes (distance <= 2.0)
            is_clustered = distance <= 2.0
            assert is_clustered

    def test_isolated_hotspot(self):
        """Test detection of isolated (non-clustered) hotspot."""
        readings = [
            TMTReading(
                tube_id="T-R1-01",
                zone="RADIANT",
                temperature_C=920.0,  # Hotspot
                rate_of_rise_C_min=1.0,
                design_limit_C=950.0,
                timestamp=datetime.now(),
                signal_quality="GOOD",
                position_x=1.0,
                position_y=1.0,
            ),
            TMTReading(
                tube_id="T-R1-10",
                zone="RADIANT",
                temperature_C=800.0,  # Normal, far away
                rate_of_rise_C_min=0.5,
                design_limit_C=950.0,
                timestamp=datetime.now(),
                signal_quality="GOOD",
                position_x=10.0,  # Far from hotspot
                position_y=10.0,
            ),
        ]

        # Check if hotspot is isolated
        hotspot = readings[0]
        neighbors = [
            r for r in readings[1:]
            if math.sqrt(
                (r.position_x - hotspot.position_x) ** 2 +
                (r.position_y - hotspot.position_y) ** 2
            ) <= 2.0
        ]

        # No elevated neighbors = isolated
        elevated_neighbors = [n for n in neighbors if n.temperature_C > 880.0]
        is_isolated = len(elevated_neighbors) == 0
        assert is_isolated

    def test_cluster_size_calculation(self):
        """Test calculation of hotspot cluster size."""
        base_time = datetime.now()
        # Create a cluster of 4 adjacent hotspots
        readings = [
            TMTReading(
                tube_id=f"T-R1-0{i}",
                zone="RADIANT",
                temperature_C=910.0 + i,
                rate_of_rise_C_min=1.0,
                design_limit_C=950.0,
                timestamp=base_time,
                signal_quality="GOOD",
                position_x=float(i % 2),
                position_y=float(i // 2),
            )
            for i in range(4)
        ]

        # Count hotspots in cluster
        cluster_threshold = 900.0
        hotspots = [r for r in readings if r.temperature_C > cluster_threshold]
        cluster_size = len(hotspots)

        assert cluster_size == 4

    def test_cluster_centroid_calculation(self):
        """Test calculation of cluster centroid."""
        readings = [
            TMTReading(
                tube_id="T-R1-01",
                zone="RADIANT",
                temperature_C=920.0,
                rate_of_rise_C_min=1.0,
                design_limit_C=950.0,
                timestamp=datetime.now(),
                signal_quality="GOOD",
                position_x=0.0,
                position_y=0.0,
            ),
            TMTReading(
                tube_id="T-R1-02",
                zone="RADIANT",
                temperature_C=910.0,
                rate_of_rise_C_min=1.0,
                design_limit_C=950.0,
                timestamp=datetime.now(),
                signal_quality="GOOD",
                position_x=2.0,
                position_y=0.0,
            ),
            TMTReading(
                tube_id="T-R1-03",
                zone="RADIANT",
                temperature_C=915.0,
                rate_of_rise_C_min=1.0,
                design_limit_C=950.0,
                timestamp=datetime.now(),
                signal_quality="GOOD",
                position_x=1.0,
                position_y=2.0,
            ),
        ]

        # Calculate centroid
        x_sum = sum(r.position_x for r in readings)
        y_sum = sum(r.position_y for r in readings)
        centroid_x = x_sum / len(readings)
        centroid_y = y_sum / len(readings)

        assert abs(centroid_x - 1.0) < 0.01
        assert abs(centroid_y - 0.67) < 0.1


class TestAlertTierAssignment:
    """Tests for alert tier (Advisory/Warning/Urgent) assignment."""

    def test_advisory_tier_conditions(self, alert_thresholds):
        """Test conditions that trigger ADVISORY tier."""
        # Advisory: TMT approaching threshold OR moderate rate of rise
        tmt_thresholds = alert_thresholds["TMT"]
        ror_thresholds = alert_thresholds["RATE_OF_RISE"]

        test_cases = [
            (855.0, 1.0, AlertTier.ADVISORY),  # TMT in advisory range
            (800.0, 3.5, AlertTier.ADVISORY),  # Rate of rise in advisory range
            (860.0, 4.0, AlertTier.ADVISORY),  # Both in advisory range
        ]

        for temp, ror, expected_tier in test_cases:
            # Determine tier based on temperature and rate of rise
            if (
                temp >= tmt_thresholds["URGENT"] or
                ror >= ror_thresholds["URGENT"]
            ):
                tier = AlertTier.URGENT
            elif (
                temp >= tmt_thresholds["WARNING"] or
                ror >= ror_thresholds["WARNING"]
            ):
                tier = AlertTier.WARNING
            elif (
                temp >= tmt_thresholds["ADVISORY"] or
                ror >= ror_thresholds["ADVISORY"]
            ):
                tier = AlertTier.ADVISORY
            else:
                tier = AlertTier.NONE

            assert tier == expected_tier

    def test_warning_tier_conditions(self, alert_thresholds):
        """Test conditions that trigger WARNING tier."""
        tmt_thresholds = alert_thresholds["TMT"]
        ror_thresholds = alert_thresholds["RATE_OF_RISE"]

        test_cases = [
            (905.0, 1.0, AlertTier.WARNING),  # TMT in warning range
            (800.0, 6.0, AlertTier.WARNING),  # Rate of rise in warning range
            (910.0, 7.0, AlertTier.WARNING),  # Both in warning range
        ]

        for temp, ror, expected_tier in test_cases:
            if (
                temp >= tmt_thresholds["URGENT"] or
                ror >= ror_thresholds["URGENT"]
            ):
                tier = AlertTier.URGENT
            elif (
                temp >= tmt_thresholds["WARNING"] or
                ror >= ror_thresholds["WARNING"]
            ):
                tier = AlertTier.WARNING
            elif (
                temp >= tmt_thresholds["ADVISORY"] or
                ror >= ror_thresholds["ADVISORY"]
            ):
                tier = AlertTier.ADVISORY
            else:
                tier = AlertTier.NONE

            assert tier == expected_tier

    def test_urgent_tier_conditions(self, alert_thresholds):
        """Test conditions that trigger URGENT tier."""
        tmt_thresholds = alert_thresholds["TMT"]
        ror_thresholds = alert_thresholds["RATE_OF_RISE"]

        test_cases = [
            (955.0, 1.0, AlertTier.URGENT),  # TMT above urgent
            (800.0, 12.0, AlertTier.URGENT),  # Rate of rise above urgent
            (960.0, 15.0, AlertTier.URGENT),  # Both above urgent
        ]

        for temp, ror, expected_tier in test_cases:
            if (
                temp >= tmt_thresholds["URGENT"] or
                ror >= ror_thresholds["URGENT"]
            ):
                tier = AlertTier.URGENT
            elif (
                temp >= tmt_thresholds["WARNING"] or
                ror >= ror_thresholds["WARNING"]
            ):
                tier = AlertTier.WARNING
            elif (
                temp >= tmt_thresholds["ADVISORY"] or
                ror >= ror_thresholds["ADVISORY"]
            ):
                tier = AlertTier.ADVISORY
            else:
                tier = AlertTier.NONE

            assert tier == expected_tier

    def test_highest_tier_takes_precedence(self, alert_thresholds):
        """Test that highest applicable tier takes precedence."""
        # TMT at WARNING, rate of rise at URGENT -> should be URGENT
        temp = 910.0  # WARNING range
        ror = 12.0  # URGENT range

        tmt_thresholds = alert_thresholds["TMT"]
        ror_thresholds = alert_thresholds["RATE_OF_RISE"]

        if temp >= tmt_thresholds["URGENT"] or ror >= ror_thresholds["URGENT"]:
            tier = AlertTier.URGENT
        elif temp >= tmt_thresholds["WARNING"] or ror >= ror_thresholds["WARNING"]:
            tier = AlertTier.WARNING
        elif temp >= tmt_thresholds["ADVISORY"] or ror >= ror_thresholds["ADVISORY"]:
            tier = AlertTier.ADVISORY
        else:
            tier = AlertTier.NONE

        assert tier == AlertTier.URGENT


class TestFalsePositiveScenarios:
    """Tests for false positive prevention."""

    def test_bad_signal_quality_ignored(self):
        """Test that BAD quality readings are not flagged as hotspots."""
        reading = TMTReading(
            tube_id="T-R1-01",
            zone="RADIANT",
            temperature_C=980.0,  # Would be urgent
            rate_of_rise_C_min=15.0,  # Would be urgent
            design_limit_C=950.0,
            timestamp=datetime.now(),
            signal_quality="BAD",  # Bad quality - should be ignored
        )

        # Bad quality readings should not trigger alerts
        is_valid_reading = reading.signal_quality == "GOOD"
        should_alert = is_valid_reading and reading.temperature_C > 900.0

        assert not should_alert

    def test_suspect_signal_flagged_for_review(self):
        """Test that SUSPECT quality readings are flagged but not alerted."""
        reading = TMTReading(
            tube_id="T-R1-01",
            zone="RADIANT",
            temperature_C=920.0,
            rate_of_rise_C_min=8.0,
            design_limit_C=950.0,
            timestamp=datetime.now(),
            signal_quality="SUSPECT",
        )

        # Suspect readings need human review
        needs_review = reading.signal_quality == "SUSPECT"
        auto_alert = reading.signal_quality == "GOOD"

        assert needs_review
        assert not auto_alert

    def test_missing_signal_not_alerted(self):
        """Test that MISSING signals do not trigger false alerts."""
        reading = TMTReading(
            tube_id="T-R1-01",
            zone="RADIANT",
            temperature_C=0.0,  # Missing value often shows as 0
            rate_of_rise_C_min=0.0,
            design_limit_C=950.0,
            timestamp=datetime.now(),
            signal_quality="MISSING",
        )

        # Missing signals should not trigger alerts
        is_valid = reading.signal_quality not in ["BAD", "MISSING"]
        assert not is_valid

    def test_transient_spike_filtered(self):
        """Test that transient spikes are filtered out."""
        base_time = datetime.now()
        readings = [
            TMTReading(
                tube_id="T-R1-01",
                zone="RADIANT",
                temperature_C=820.0,
                rate_of_rise_C_min=0.5,
                design_limit_C=950.0,
                timestamp=base_time,
                signal_quality="GOOD",
            ),
            TMTReading(
                tube_id="T-R1-01",
                zone="RADIANT",
                temperature_C=920.0,  # Spike
                rate_of_rise_C_min=100.0,  # Unrealistic spike
                design_limit_C=950.0,
                timestamp=base_time + timedelta(seconds=10),
                signal_quality="GOOD",
            ),
            TMTReading(
                tube_id="T-R1-01",
                zone="RADIANT",
                temperature_C=822.0,  # Back to normal
                rate_of_rise_C_min=0.2,
                design_limit_C=950.0,
                timestamp=base_time + timedelta(seconds=20),
                signal_quality="GOOD",
            ),
        ]

        # Detect spike (unrealistic rate of rise)
        max_realistic_ror = 20.0  # Max realistic rate of rise
        is_spike = readings[1].rate_of_rise_C_min > max_realistic_ror

        assert is_spike

    def test_debounce_prevents_rapid_alerts(self):
        """Test that debounce prevents multiple rapid alerts."""
        debounce_seconds = 60
        last_alert_time = datetime.now() - timedelta(seconds=30)
        current_time = datetime.now()

        time_since_last = (current_time - last_alert_time).total_seconds()
        can_alert = time_since_last >= debounce_seconds

        assert not can_alert  # Should be debounced

    def test_sensor_stuck_detection(self):
        """Test detection of stuck sensor (constant reading)."""
        base_time = datetime.now()
        # Sensor stuck at same value
        readings = [
            TMTReading(
                tube_id="T-R1-01",
                zone="RADIANT",
                temperature_C=820.0,  # Exact same value
                rate_of_rise_C_min=0.0,
                design_limit_C=950.0,
                timestamp=base_time + timedelta(minutes=i),
                signal_quality="GOOD",
            )
            for i in range(10)
        ]

        # Detect stuck sensor (no variation over time)
        temps = [r.temperature_C for r in readings]
        variance = sum((t - temps[0]) ** 2 for t in temps) / len(temps)

        is_stuck = variance < 0.001  # Nearly zero variance
        assert is_stuck

    def test_cross_zone_validation(self, sample_tmt_readings_normal):
        """Test cross-zone validation for plausibility."""
        # Radiant zone should be hotter than convection zone
        radiant_temps = [
            r.temperature_C for r in sample_tmt_readings_normal
            if r.zone == "RADIANT"
        ]
        convection_temps = [
            r.temperature_C for r in sample_tmt_readings_normal
            if r.zone == "CONVECTION"
        ]

        if radiant_temps and convection_temps:
            avg_radiant = sum(radiant_temps) / len(radiant_temps)
            avg_convection = sum(convection_temps) / len(convection_temps)

            # Radiant should be hotter
            is_plausible = avg_radiant > avg_convection
            assert is_plausible


class TestHotspotConfidence:
    """Tests for hotspot detection confidence scoring."""

    def test_confidence_score_high_single_sensor(self):
        """Test confidence score for single sensor detection."""
        # Single sensor detection has lower confidence
        num_confirming_sensors = 1
        signal_quality_score = 1.0  # GOOD quality

        base_confidence = 0.5  # Single sensor baseline
        quality_factor = signal_quality_score
        confidence = base_confidence * quality_factor

        assert 0.4 <= confidence <= 0.6

    def test_confidence_score_high_multiple_sensors(self):
        """Test confidence score with multiple confirming sensors."""
        num_confirming_sensors = 4
        signal_quality_score = 1.0

        # More sensors = higher confidence
        base_confidence = min(0.5 + (num_confirming_sensors - 1) * 0.15, 1.0)
        confidence = base_confidence * signal_quality_score

        assert confidence >= 0.9

    def test_confidence_reduced_for_suspect_quality(self):
        """Test that suspect quality reduces confidence."""
        signal_quality_score = 0.5  # SUSPECT quality
        base_confidence = 0.8

        confidence = base_confidence * signal_quality_score

        assert confidence == 0.4

    @pytest.mark.parametrize(
        "num_sensors,quality,expected_min_confidence",
        [
            (1, "GOOD", 0.4),
            (2, "GOOD", 0.6),
            (3, "GOOD", 0.75),
            (4, "GOOD", 0.9),
            (2, "SUSPECT", 0.3),
            (4, "SUSPECT", 0.45),
        ],
    )
    def test_confidence_scoring_parametrized(
        self, num_sensors, quality, expected_min_confidence
    ):
        """Test confidence scoring with various conditions."""
        quality_score = 1.0 if quality == "GOOD" else 0.5
        base_confidence = min(0.5 + (num_sensors - 1) * 0.15, 1.0)
        confidence = base_confidence * quality_score

        assert confidence >= expected_min_confidence


class TestZoneSpecificBehavior:
    """Tests for zone-specific hotspot behavior."""

    def test_radiant_zone_higher_threshold(self, alert_thresholds):
        """Test that radiant zone may have different thresholds."""
        # Radiant zone typically has higher operating temps
        radiant_warning = alert_thresholds["TMT"]["WARNING"]

        # Example: Radiant zone might accept 50C higher
        radiant_adjusted_warning = radiant_warning  # Could be zone-specific

        reading = TMTReading(
            tube_id="T-R1-01",
            zone="RADIANT",
            temperature_C=890.0,
            rate_of_rise_C_min=1.0,
            design_limit_C=950.0,
            timestamp=datetime.now(),
            signal_quality="GOOD",
        )

        is_below_warning = reading.temperature_C < radiant_adjusted_warning
        assert is_below_warning

    def test_convection_zone_lower_limits(self):
        """Test convection zone has lower temperature limits."""
        convection_design_limit = 850.0  # Lower than radiant

        reading = TMTReading(
            tube_id="T-C1-01",
            zone="CONVECTION",
            temperature_C=870.0,  # Would be OK for radiant, high for convection
            rate_of_rise_C_min=1.0,
            design_limit_C=convection_design_limit,
            timestamp=datetime.now(),
            signal_quality="GOOD",
        )

        exceeds_design_limit = reading.temperature_C > reading.design_limit_C
        assert exceeds_design_limit

    def test_shield_zone_monitoring(self):
        """Test shield zone specific monitoring."""
        reading = TMTReading(
            tube_id="T-S1-01",
            zone="SHIELD",
            temperature_C=750.0,
            rate_of_rise_C_min=0.5,
            design_limit_C=900.0,
            timestamp=datetime.now(),
            signal_quality="GOOD",
        )

        # Shield tubes protect convection section
        margin_to_limit = reading.design_limit_C - reading.temperature_C
        adequate_margin = margin_to_limit > 100.0  # >100C margin

        assert adequate_margin


class TestHotspotProvenanceTracking:
    """Tests for hotspot detection provenance and audit trail."""

    def test_detection_hash_generated(self, sample_tmt_readings_hotspot):
        """Test that detection generates provenance hash."""
        import hashlib
        import json

        # Create detection record
        detection = {
            "timestamp": datetime.now().isoformat(),
            "readings": [
                {
                    "tube_id": r.tube_id,
                    "temperature_C": r.temperature_C,
                    "rate_of_rise_C_min": r.rate_of_rise_C_min,
                }
                for r in sample_tmt_readings_hotspot
            ],
            "thresholds_applied": {"warning": 900.0, "urgent": 950.0},
            "method_version": "1.0.0",
        }

        # Generate hash
        detection_json = json.dumps(detection, sort_keys=True, default=str)
        detection_hash = hashlib.sha256(detection_json.encode()).hexdigest()

        assert len(detection_hash) == 64

    def test_detection_reproducible(self, sample_tmt_readings_hotspot):
        """Test that detection results are reproducible."""
        import hashlib
        import json

        def detect_hotspots(readings: List[TMTReading]) -> Dict[str, Any]:
            threshold = 900.0
            hotspots = [
                {"tube_id": r.tube_id, "temperature_C": r.temperature_C}
                for r in readings
                if r.temperature_C > threshold
            ]
            return {"hotspots": hotspots, "count": len(hotspots)}

        result1 = detect_hotspots(sample_tmt_readings_hotspot)
        result2 = detect_hotspots(sample_tmt_readings_hotspot)

        assert result1 == result2

    def test_audit_trail_includes_all_inputs(self, sample_tmt_readings_hotspot):
        """Test that audit trail includes all input readings."""
        audit_record = {
            "input_count": len(sample_tmt_readings_hotspot),
            "input_tube_ids": [r.tube_id for r in sample_tmt_readings_hotspot],
            "input_timestamps": [r.timestamp.isoformat() for r in sample_tmt_readings_hotspot],
            "analysis_timestamp": datetime.now().isoformat(),
        }

        assert audit_record["input_count"] == len(sample_tmt_readings_hotspot)
        assert len(audit_record["input_tube_ids"]) == len(sample_tmt_readings_hotspot)


class TestPerformance:
    """Performance tests for hotspot detection."""

    def test_large_dataset_processing(self, large_tmt_dataset):
        """Test processing of large TMT dataset."""
        import time

        start_time = time.time()

        # Process all readings
        threshold = 900.0
        hotspots = [r for r in large_tmt_dataset if r.temperature_C > threshold]
        high_ror = [r for r in large_tmt_dataset if r.rate_of_rise_C_min > 5.0]

        elapsed = time.time() - start_time

        # Should process in < 100ms
        assert elapsed < 0.1

    def test_streaming_processing(self):
        """Test processing in streaming mode (one reading at a time)."""
        import time

        base_time = datetime.now()
        start_time = time.time()

        alert_count = 0
        for i in range(1000):
            reading = TMTReading(
                tube_id=f"T-R1-{i:03d}",
                zone="RADIANT",
                temperature_C=800.0 + (i % 200),
                rate_of_rise_C_min=0.5,
                design_limit_C=950.0,
                timestamp=base_time + timedelta(seconds=i),
                signal_quality="GOOD",
            )

            if reading.temperature_C > 900.0:
                alert_count += 1

        elapsed = time.time() - start_time

        # Should process 1000 readings in < 50ms
        assert elapsed < 0.05
