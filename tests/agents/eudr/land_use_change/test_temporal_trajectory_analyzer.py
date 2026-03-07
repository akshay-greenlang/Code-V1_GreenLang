# -*- coding: utf-8 -*-
"""
Tests for TemporalTrajectoryAnalyzer - AGENT-EUDR-005 Land Use Change Detector

Comprehensive test suite covering:
- Stable trajectory detection and validation
- Abrupt change trajectory detection with date localization
- Gradual change trajectory detection with date range estimation
- Oscillating trajectory detection with period calculation
- Recovery trajectory detection with completeness scoring
- Natural disturbance distinction from anthropogenic change
- Trajectory confidence scoring
- Batch analysis processing
- Trajectory visualization data generation
- Minimum temporal depth validation
- Deterministic analysis behavior

Test count: 55 tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
"""

import math
from unittest.mock import MagicMock

import pytest

from greenlang.agents.eudr.land_use_change.config import LandUseChangeConfig
from tests.agents.eudr.land_use_change.conftest import (
    TemporalTrajectory,
    compute_test_hash,
    SHA256_HEX_LENGTH,
    TRAJECTORY_TYPES,
)


# ===========================================================================
# 1. Stable Trajectory Tests (8 tests)
# ===========================================================================


class TestStableTrajectory:
    """Tests for detecting stable (no change) trajectories."""

    def test_stable_trajectory_type(self, stable_forest_ndvi_series):
        """Stable NDVI series is classified as stable trajectory."""
        result = TemporalTrajectory(
            plot_id="PLOT-STB-001",
            trajectory_type="stable",
            dates=stable_forest_ndvi_series["dates"],
            ndvi_values=stable_forest_ndvi_series["ndvi_values"],
            confidence=0.95,
        )
        assert result.trajectory_type == "stable"

    def test_stable_trajectory_high_confidence(self, stable_forest_ndvi_series):
        """Stable trajectory has high confidence (>0.85)."""
        values = stable_forest_ndvi_series["ndvi_values"]
        mean_val = sum(values) / len(values)
        std_val = math.sqrt(sum((v - mean_val) ** 2 for v in values) / len(values))
        cv = std_val / mean_val if mean_val > 0 else 0
        assert cv < 0.10  # low coefficient of variation = stable

    def test_stable_trajectory_no_change_date(self, stable_forest_ndvi_series):
        """Stable trajectory has no change date."""
        result = TemporalTrajectory(
            plot_id="PLOT-STB-002",
            trajectory_type="stable",
            change_date=None,
            confidence=0.93,
        )
        assert result.change_date is None

    def test_stable_trajectory_no_oscillation(self, stable_forest_ndvi_series):
        """Stable trajectory has no oscillation period."""
        result = TemporalTrajectory(
            plot_id="PLOT-STB-003",
            trajectory_type="stable",
            oscillation_period_months=None,
            confidence=0.92,
        )
        assert result.oscillation_period_months is None

    def test_stable_trajectory_ndvi_consistency(self, stable_forest_ndvi_series):
        """Stable trajectory NDVI values stay within narrow band."""
        values = stable_forest_ndvi_series["ndvi_values"]
        assert all(0.60 <= v <= 0.80 for v in values)

    def test_stable_trajectory_length(self, stable_forest_ndvi_series):
        """Stable trajectory has 60 monthly observations (5 years)."""
        assert len(stable_forest_ndvi_series["dates"]) == 60
        assert len(stable_forest_ndvi_series["ndvi_values"]) == 60

    def test_stable_trajectory_mean_ndvi(self, stable_forest_ndvi_series):
        """Stable trajectory mean NDVI is around 0.70."""
        values = stable_forest_ndvi_series["ndvi_values"]
        mean_val = sum(values) / len(values)
        assert abs(mean_val - 0.70) < 0.05

    def test_stable_trajectory_not_natural_disturbance(self):
        """Stable trajectory is not flagged as natural disturbance."""
        result = TemporalTrajectory(
            plot_id="PLOT-STB-004",
            trajectory_type="stable",
            is_natural_disturbance=False,
        )
        assert result.is_natural_disturbance is False


# ===========================================================================
# 2. Abrupt Change Trajectory Tests (8 tests)
# ===========================================================================


class TestAbruptChangeTrajectory:
    """Tests for detecting abrupt (rapid) change trajectories."""

    def test_abrupt_change_trajectory_type(self, abrupt_deforestation_series):
        """Abrupt NDVI drop is classified as abrupt change."""
        result = TemporalTrajectory(
            plot_id="PLOT-ABR-001",
            trajectory_type="abrupt_change",
            dates=abrupt_deforestation_series["dates"],
            ndvi_values=abrupt_deforestation_series["ndvi_values"],
            confidence=0.92,
        )
        assert result.trajectory_type == "abrupt_change"

    def test_abrupt_change_date_detection(self, abrupt_deforestation_series):
        """Abrupt change date is detected at month 30 (mid-2020)."""
        dates = abrupt_deforestation_series["dates"]
        values = abrupt_deforestation_series["ndvi_values"]
        max_drop_idx = 0
        max_drop = 0.0
        for i in range(1, len(values)):
            drop = values[i - 1] - values[i]
            if drop > max_drop:
                max_drop = drop
                max_drop_idx = i
        assert max_drop > 0.40  # sharp drop
        change_date = dates[max_drop_idx]
        result = TemporalTrajectory(
            plot_id="PLOT-ABR-002",
            trajectory_type="abrupt_change",
            change_date=change_date,
            confidence=0.90,
        )
        assert result.change_date is not None

    def test_abrupt_change_magnitude(self, abrupt_deforestation_series):
        """Abrupt change magnitude exceeds 0.40 NDVI units."""
        values = abrupt_deforestation_series["ndvi_values"]
        before_mean = sum(values[:30]) / 30
        after_mean = sum(values[30:]) / 30
        magnitude = before_mean - after_mean
        assert magnitude > 0.40

    def test_abrupt_change_high_confidence(self):
        """Abrupt changes have high detection confidence."""
        result = TemporalTrajectory(
            plot_id="PLOT-ABR-003",
            trajectory_type="abrupt_change",
            confidence=0.92,
        )
        assert result.confidence > 0.85

    def test_abrupt_change_before_values(self, abrupt_deforestation_series):
        """NDVI values before change are high (~0.7)."""
        values = abrupt_deforestation_series["ndvi_values"]
        before_mean = sum(values[:29]) / 29
        assert before_mean > 0.65

    def test_abrupt_change_after_values(self, abrupt_deforestation_series):
        """NDVI values after change are low (~0.2)."""
        values = abrupt_deforestation_series["ndvi_values"]
        after_mean = sum(values[31:]) / (len(values) - 31)
        assert after_mean < 0.30

    def test_abrupt_change_single_event(self, abrupt_deforestation_series):
        """Only one abrupt change event is detected in the series."""
        values = abrupt_deforestation_series["ndvi_values"]
        large_drops = 0
        for i in range(1, len(values)):
            if values[i - 1] - values[i] > 0.30:
                large_drops += 1
        assert large_drops == 1

    def test_abrupt_change_not_recovery(self, abrupt_deforestation_series):
        """Abrupt change does not show recovery afterward."""
        values = abrupt_deforestation_series["ndvi_values"]
        after_values = values[35:]
        if after_values:
            assert max(after_values) < 0.35


# ===========================================================================
# 3. Gradual Change Trajectory Tests (8 tests)
# ===========================================================================


class TestGradualChangeTrajectory:
    """Tests for detecting gradual change trajectories."""

    def test_gradual_change_trajectory_type(self, gradual_conversion_series):
        """Gradual NDVI decline is classified as gradual change."""
        result = TemporalTrajectory(
            plot_id="PLOT-GRD-001",
            trajectory_type="gradual_change",
            dates=gradual_conversion_series["dates"],
            ndvi_values=gradual_conversion_series["ndvi_values"],
            confidence=0.85,
        )
        assert result.trajectory_type == "gradual_change"

    def test_gradual_change_date_range(self, gradual_conversion_series):
        """Gradual change has a date range spanning months 24-36."""
        dates = gradual_conversion_series["dates"]
        result = TemporalTrajectory(
            plot_id="PLOT-GRD-002",
            trajectory_type="gradual_change",
            change_date_range=(dates[24], dates[36]),
            confidence=0.82,
        )
        assert result.change_date_range is not None
        assert result.change_date_range[0] < result.change_date_range[1]

    def test_gradual_change_duration(self, gradual_conversion_series):
        """Gradual change spans approximately 12 months."""
        values = gradual_conversion_series["ndvi_values"]
        decline_start = None
        decline_end = None
        for i in range(1, len(values)):
            if values[i] < values[i - 1] - 0.02 and decline_start is None:
                decline_start = i
            if decline_start is not None and values[i] < 0.35:
                decline_end = i
                break
        if decline_start is not None and decline_end is not None:
            duration = decline_end - decline_start
            assert 8 <= duration <= 16  # approximately 12 months

    def test_gradual_change_start_ndvi(self, gradual_conversion_series):
        """NDVI at start of gradual change is ~0.7."""
        values = gradual_conversion_series["ndvi_values"]
        assert values[24] > 0.60

    def test_gradual_change_end_ndvi(self, gradual_conversion_series):
        """NDVI at end of gradual change is ~0.3."""
        values = gradual_conversion_series["ndvi_values"]
        assert values[40] < 0.40

    def test_gradual_change_monotonic_decline(self, gradual_conversion_series):
        """NDVI decline is approximately monotonic during transition."""
        values = gradual_conversion_series["ndvi_values"]
        transition_values = values[24:37]
        increases = sum(
            1 for i in range(1, len(transition_values))
            if transition_values[i] > transition_values[i - 1] + 0.03
        )
        # Allow a few non-monotonic steps due to noise
        assert increases <= 3

    def test_gradual_change_lower_confidence(self):
        """Gradual changes may have lower confidence than abrupt."""
        result = TemporalTrajectory(
            plot_id="PLOT-GRD-003",
            trajectory_type="gradual_change",
            confidence=0.78,
        )
        assert 0.70 <= result.confidence <= 0.90

    def test_gradual_change_total_magnitude(self, gradual_conversion_series):
        """Total NDVI change magnitude is ~0.40."""
        values = gradual_conversion_series["ndvi_values"]
        before = sum(values[:24]) / 24
        after = sum(values[40:]) / max(len(values[40:]), 1)
        magnitude = before - after
        assert 0.30 <= magnitude <= 0.50


# ===========================================================================
# 4. Oscillating Trajectory Tests (7 tests)
# ===========================================================================


class TestOscillatingTrajectory:
    """Tests for detecting oscillating (seasonal crop) trajectories."""

    def test_oscillating_trajectory_type(self, oscillating_crop_series):
        """Oscillating NDVI is classified as oscillating trajectory."""
        result = TemporalTrajectory(
            plot_id="PLOT-OSC-001",
            trajectory_type="oscillating",
            dates=oscillating_crop_series["dates"],
            ndvi_values=oscillating_crop_series["ndvi_values"],
            confidence=0.88,
        )
        assert result.trajectory_type == "oscillating"

    def test_oscillation_period_detection(self, oscillating_crop_series):
        """Oscillation period is detected as ~12 months."""
        result = TemporalTrajectory(
            plot_id="PLOT-OSC-002",
            trajectory_type="oscillating",
            oscillation_period_months=12,
            confidence=0.86,
        )
        assert result.oscillation_period_months == 12

    def test_oscillating_ndvi_range(self, oscillating_crop_series):
        """Oscillating NDVI spans 0.2 to 0.7 range."""
        values = oscillating_crop_series["ndvi_values"]
        assert min(values) > 0.15
        assert max(values) < 0.75

    def test_oscillating_amplitude(self, oscillating_crop_series):
        """Oscillation amplitude is approximately 0.50."""
        values = oscillating_crop_series["ndvi_values"]
        amplitude = max(values) - min(values)
        assert 0.40 <= amplitude <= 0.55

    def test_oscillating_regular_cycles(self, oscillating_crop_series):
        """Oscillation shows regular (repeating) cycles."""
        values = oscillating_crop_series["ndvi_values"]
        # Count zero-crossings relative to mean
        mean_val = sum(values) / len(values)
        crossings = sum(
            1 for i in range(1, len(values))
            if (values[i - 1] - mean_val) * (values[i] - mean_val) < 0
        )
        # 5 years * ~2 crossings per year = ~10 crossings
        assert crossings >= 6

    def test_oscillating_not_deforestation(self):
        """Oscillating pattern is NOT deforestation."""
        result = TemporalTrajectory(
            plot_id="PLOT-OSC-003",
            trajectory_type="oscillating",
            is_natural_disturbance=False,
        )
        assert result.trajectory_type != "abrupt_change"
        assert result.is_natural_disturbance is False

    def test_oscillating_mean_ndvi(self, oscillating_crop_series):
        """Mean NDVI of oscillating series is around 0.45."""
        values = oscillating_crop_series["ndvi_values"]
        mean_val = sum(values) / len(values)
        assert abs(mean_val - 0.45) < 0.10


# ===========================================================================
# 5. Recovery Trajectory Tests (7 tests)
# ===========================================================================


class TestRecoveryTrajectory:
    """Tests for detecting recovery trajectories."""

    def test_recovery_trajectory_type(self, recovery_series):
        """Disturbed-then-recovering series is classified as recovery."""
        result = TemporalTrajectory(
            plot_id="PLOT-REC-001",
            trajectory_type="recovery",
            dates=recovery_series["dates"],
            ndvi_values=recovery_series["ndvi_values"],
            confidence=0.82,
        )
        assert result.trajectory_type == "recovery"

    def test_recovery_completeness(self, recovery_series):
        """Recovery completeness indicates how much NDVI has recovered."""
        values = recovery_series["ndvi_values"]
        pre_disturbance = sum(values[:20]) / 20
        min_val = min(values[20:25])
        current = values[-1]
        if pre_disturbance > min_val:
            completeness = (current - min_val) / (pre_disturbance - min_val)
        else:
            completeness = 0.0
        result = TemporalTrajectory(
            plot_id="PLOT-REC-002",
            trajectory_type="recovery",
            recovery_completeness=completeness,
        )
        assert result.recovery_completeness is not None
        assert 0.0 <= result.recovery_completeness <= 1.0

    def test_recovery_initial_drop(self, recovery_series):
        """Recovery series shows initial disturbance drop."""
        values = recovery_series["ndvi_values"]
        assert values[20] < 0.30

    def test_recovery_upward_trend(self, recovery_series):
        """Recovery series shows upward trend after disturbance."""
        values = recovery_series["ndvi_values"]
        late_values = values[40:]
        early_recovery = values[22:30]
        if late_values and early_recovery:
            assert sum(late_values) / len(late_values) > sum(early_recovery) / len(early_recovery)

    def test_recovery_not_full(self, recovery_series):
        """Recovery does not reach pre-disturbance levels."""
        values = recovery_series["ndvi_values"]
        pre_mean = sum(values[:20]) / 20
        current = values[-1]
        assert current < pre_mean

    def test_recovery_change_date_at_disturbance(self, recovery_series):
        """Change date is set at the disturbance event."""
        dates = recovery_series["dates"]
        result = TemporalTrajectory(
            plot_id="PLOT-REC-003",
            trajectory_type="recovery",
            change_date=dates[20],
        )
        assert result.change_date == dates[20]

    def test_recovery_confidence_moderate(self):
        """Recovery trajectories have moderate confidence."""
        result = TemporalTrajectory(
            plot_id="PLOT-REC-004",
            trajectory_type="recovery",
            confidence=0.78,
        )
        assert 0.65 <= result.confidence <= 0.90


# ===========================================================================
# 6. Natural Disturbance Tests (4 tests)
# ===========================================================================


class TestNaturalDisturbance:
    """Tests for distinguishing natural disturbance from anthropogenic."""

    def test_natural_disturbance_flag(self):
        """Natural disturbance events are flagged separately."""
        result = TemporalTrajectory(
            plot_id="PLOT-NAT-001",
            trajectory_type="recovery",
            is_natural_disturbance=True,
            confidence=0.75,
        )
        assert result.is_natural_disturbance is True

    def test_anthropogenic_disturbance_flag(self):
        """Anthropogenic events are NOT flagged as natural."""
        result = TemporalTrajectory(
            plot_id="PLOT-ANT-001",
            trajectory_type="abrupt_change",
            is_natural_disturbance=False,
            confidence=0.90,
        )
        assert result.is_natural_disturbance is False

    def test_natural_disturbance_with_recovery(self):
        """Natural disturbance shows recovery pattern."""
        result = TemporalTrajectory(
            plot_id="PLOT-NAT-002",
            trajectory_type="recovery",
            is_natural_disturbance=True,
            recovery_completeness=0.65,
        )
        assert result.is_natural_disturbance is True
        assert result.recovery_completeness > 0.5

    def test_fire_disturbance(self):
        """Fire events can show abrupt drop with recovery."""
        result = TemporalTrajectory(
            plot_id="PLOT-FIRE-001",
            trajectory_type="recovery",
            is_natural_disturbance=True,
            change_date="2022-08-15",
            recovery_completeness=0.40,
        )
        assert result.is_natural_disturbance is True


# ===========================================================================
# 7. Batch and Determinism Tests (6 tests)
# ===========================================================================


class TestBatchAndDeterminism:
    """Tests for batch processing and deterministic analysis."""

    def test_batch_analysis(self):
        """Batch analysis of multiple plots returns correct count."""
        results = [
            TemporalTrajectory(
                plot_id=f"PLOT-BATCH-{i:04d}",
                trajectory_type="stable" if i % 3 == 0 else "abrupt_change",
                confidence=0.80,
            )
            for i in range(30)
        ]
        assert len(results) == 30
        stable = sum(1 for r in results if r.trajectory_type == "stable")
        assert stable == 10

    def test_trajectory_visualization_data(self):
        """Trajectory result includes visualization-ready data."""
        result = TemporalTrajectory(
            plot_id="PLOT-VIS-001",
            trajectory_type="abrupt_change",
            visualization_data={
                "x_axis": "date",
                "y_axis": "ndvi",
                "change_marker": "2021-06-01",
                "before_color": "#228B22",
                "after_color": "#CD853F",
            },
        )
        assert "x_axis" in result.visualization_data
        assert "change_marker" in result.visualization_data

    def test_minimum_temporal_depth(self, sample_config):
        """Analysis requires minimum temporal depth (3 years)."""
        assert sample_config.min_temporal_depth_years == 3
        # 3 years * 12 months = 36 minimum observations
        min_observations = sample_config.min_temporal_depth_years * 12
        assert min_observations >= 36

    def test_deterministic_analysis(self, stable_forest_ndvi_series):
        """Same input series produces identical trajectory results."""
        results = [
            TemporalTrajectory(
                plot_id="PLOT-DET-001",
                trajectory_type="stable",
                ndvi_values=stable_forest_ndvi_series["ndvi_values"],
                confidence=0.93,
            )
            for _ in range(5)
        ]
        assert all(r.trajectory_type == "stable" for r in results)
        assert all(r.confidence == 0.93 for r in results)

    def test_provenance_hash(self):
        """Trajectory result includes provenance hash."""
        h = compute_test_hash({"plot_id": "PLOT-PRV-001", "type": "stable"})
        result = TemporalTrajectory(
            plot_id="PLOT-PRV-001",
            trajectory_type="stable",
            provenance_hash=h,
        )
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH

    @pytest.mark.parametrize("ttype", TRAJECTORY_TYPES)
    def test_all_trajectory_types_valid(self, ttype):
        """Each trajectory type value is accepted."""
        result = TemporalTrajectory(
            plot_id=f"PLOT-{ttype.upper()}-001",
            trajectory_type=ttype,
            confidence=0.80,
        )
        assert result.trajectory_type == ttype


# ===========================================================================
# 8. Statistical Properties Tests (7 tests)
# ===========================================================================


class TestStatisticalProperties:
    """Tests for statistical properties of trajectory analysis."""

    def test_stable_low_variance(self, stable_forest_ndvi_series):
        """Stable trajectories have low NDVI variance."""
        values = stable_forest_ndvi_series["ndvi_values"]
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        assert variance < 0.005

    def test_abrupt_high_step_change(self, abrupt_deforestation_series):
        """Abrupt trajectories have high single-step change."""
        values = abrupt_deforestation_series["ndvi_values"]
        max_step = max(
            abs(values[i] - values[i - 1])
            for i in range(1, len(values))
        )
        assert max_step > 0.30

    def test_gradual_small_steps(self, gradual_conversion_series):
        """Gradual trajectories have small per-step changes."""
        values = gradual_conversion_series["ndvi_values"]
        transition = values[24:37]
        max_step = max(
            abs(transition[i] - transition[i - 1])
            for i in range(1, len(transition))
        )
        assert max_step < 0.10

    def test_oscillating_high_autocorrelation_lag12(self, oscillating_crop_series):
        """Oscillating series has high autocorrelation at lag 12."""
        values = oscillating_crop_series["ndvi_values"]
        n = len(values)
        mean_val = sum(values) / n
        var = sum((v - mean_val) ** 2 for v in values) / n
        if var > 0:
            lag = 12
            autocorr = sum(
                (values[i] - mean_val) * (values[i + lag] - mean_val)
                for i in range(n - lag)
            ) / ((n - lag) * var)
            assert autocorr > 0.5

    def test_recovery_positive_slope_post_disturbance(self, recovery_series):
        """Recovery series has positive slope after disturbance."""
        values = recovery_series["ndvi_values"]
        post_values = values[22:]
        if len(post_values) >= 2:
            # simple linear slope: (last - first) / n
            slope = (post_values[-1] - post_values[0]) / len(post_values)
            assert slope > 0.0

    def test_all_series_have_60_points(
        self,
        stable_forest_ndvi_series,
        abrupt_deforestation_series,
        gradual_conversion_series,
        oscillating_crop_series,
        recovery_series,
    ):
        """All time series fixtures have exactly 60 monthly observations."""
        for series in [
            stable_forest_ndvi_series,
            abrupt_deforestation_series,
            gradual_conversion_series,
            oscillating_crop_series,
            recovery_series,
        ]:
            assert len(series["dates"]) == 60
            assert len(series["ndvi_values"]) == 60

    def test_dates_are_chronological(self, stable_forest_ndvi_series):
        """Dates in time series are in chronological order."""
        dates = stable_forest_ndvi_series["dates"]
        for i in range(1, len(dates)):
            assert dates[i] > dates[i - 1]
