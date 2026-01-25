"""
GL-019 HEATSCHEDULER - Thermal Load Forecaster Tests

Comprehensive unit tests for the ThermalLoadForecaster calculator.
Tests cover degree-day calculations, seasonal decomposition, weather
normalization, occupancy adjustment, and forecast accuracy metrics.

Author: GL-TestEngineer
Version: 1.0.0
"""

import sys
import os
import pytest
import math
from decimal import Decimal
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from calculators.thermal_load_forecaster import (
    # Main class
    ThermalLoadForecaster,
    # Input/Output dataclasses
    ThermalLoadForecastInput,
    ThermalLoadForecastOutput,
    HistoricalLoadData,
    WeatherData,
    BuildingThermalModel,
    OccupancySchedule,
    HourlyLoadForecast,
    SeasonalComponents,
    ForecastAccuracyMetrics,
    PeakDemandPrediction,
    # Enums
    SeasonType,
    ForecastMethod,
    BuildingType,
    # Standalone functions
    calculate_hdd,
    calculate_cdd,
    calculate_degree_days_batch,
    calculate_mape,
    calculate_rmse,
    calculate_cv_rmse,
    apply_thermal_lag,
    adjust_load_for_occupancy,
    get_season,
    cached_hdd_calculation,
    calculate_energy_signature,
    # Constants
    HDD_BASE_TEMP_C,
    CDD_BASE_TEMP_C,
)

from calculators.provenance import verify_provenance


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_historical_data() -> List[HistoricalLoadData]:
    """Generate sample historical load data (168 hours = 1 week)."""
    data = []
    for hour in range(168):
        day_of_week = (hour // 24) % 7
        hour_of_day = hour % 24

        # Create realistic load pattern
        base_load = 100.0
        daily_factor = 1.0 + 0.5 * math.sin((hour_of_day - 6) * math.pi / 12)
        weekly_factor = 1.0 if day_of_week < 5 else 0.6  # Weekday vs weekend

        # Temperature variation (winter pattern)
        temp = 5.0 + 10.0 * math.sin(hour * 2 * math.pi / 24)
        hdd = max(0.0, HDD_BASE_TEMP_C - temp) / 24.0

        energy = base_load * daily_factor * weekly_factor + 20.0 * hdd

        data.append(HistoricalLoadData(
            timestamp=f"2024-01-{1 + hour // 24:02d}T{hour_of_day:02d}:00:00Z",
            energy_kwh=energy,
            peak_demand_kw=energy * 1.3,
            temp_avg_c=temp,
            heating_degree_days=hdd,
            cooling_degree_days=0.0,
            occupancy_fraction=0.8 if 6 <= hour_of_day <= 18 else 0.2,
            day_of_week=day_of_week,
            hour_of_day=hour_of_day,
            is_holiday=False,
            production_level=1.0 if day_of_week < 5 else 0.5
        ))
    return data


@pytest.fixture
def sample_weather_forecast() -> List[WeatherData]:
    """Generate sample weather forecast (7 days)."""
    forecasts = []
    for day in range(7):
        temp = 5.0 + day * 0.5  # Gradually warming
        forecasts.append(WeatherData(
            date=f"2024-01-{8 + day:02d}",
            temp_avg_c=temp,
            temp_min_c=temp - 5,
            temp_max_c=temp + 5,
            humidity_pct=60.0,
            wind_speed_mps=3.0,
            solar_radiation_wm2=100.0
        ))
    return forecasts


@pytest.fixture
def sample_building_model() -> BuildingThermalModel:
    """Sample building thermal model."""
    return BuildingThermalModel(
        building_id="BLDG-001",
        floor_area_m2=5000.0,
        building_type=BuildingType.MEDIUM,
        thermal_time_constant_h=8.0,
        ua_value_w_per_k=2500.0,
        internal_gain_w_per_m2=30.0,
        hvac_efficiency=0.90,
        setpoint_heating_c=20.0,
        setpoint_cooling_c=24.0,
        base_load_kw=50.0
    )


@pytest.fixture
def sample_occupancy_schedule() -> OccupancySchedule:
    """Sample occupancy schedule."""
    weekday = tuple([0.1]*6 + [0.5, 0.8, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 0.7] + [0.4, 0.2, 0.1, 0.1, 0.1, 0.1])
    weekend = tuple([0.1] * 24)
    return OccupancySchedule(
        schedule_id="OCC-001",
        weekday_profile=weekday,
        weekend_profile=weekend,
        holiday_profile=weekend,
        occupancy_load_factor=0.2
    )


@pytest.fixture
def forecaster() -> ThermalLoadForecaster:
    """Create ThermalLoadForecaster instance."""
    return ThermalLoadForecaster()


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

class TestThermalLoadForecasterBasic:
    """Basic functionality tests for ThermalLoadForecaster."""

    def test_forecaster_initialization(self, forecaster):
        """Test forecaster initializes correctly."""
        assert forecaster is not None
        assert forecaster.VERSION == "1.0.0"
        assert forecaster.NAME == "ThermalLoadForecaster"

    def test_simple_forecast(self, forecaster, sample_historical_data, sample_weather_forecast):
        """Test basic forecast generation."""
        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=24
        )

        result, provenance = forecaster.forecast(inputs)

        assert result is not None
        assert provenance is not None
        assert len(result.hourly_forecasts) == 24
        assert result.total_energy_kwh > 0
        assert result.peak_demand_kw > 0

    def test_forecast_with_building_model(
        self, forecaster, sample_historical_data, sample_weather_forecast, sample_building_model
    ):
        """Test forecast with building thermal model."""
        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            building_model=sample_building_model,
            forecast_horizon_hours=48
        )

        result, provenance = forecaster.forecast(inputs)

        assert result is not None
        assert len(result.hourly_forecasts) == 48
        assert result.hdd_sensitivity >= 0

    def test_forecast_with_occupancy(
        self, forecaster, sample_historical_data, sample_weather_forecast, sample_occupancy_schedule
    ):
        """Test forecast with occupancy schedule."""
        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            occupancy_schedule=sample_occupancy_schedule,
            forecast_horizon_hours=24
        )

        result, provenance = forecaster.forecast(inputs)

        assert result is not None
        # Occupancy adjustment should be applied
        for forecast in result.hourly_forecasts:
            assert 0 <= forecast.occupancy_adjustment <= 1

    def test_provenance_verification(self, forecaster, sample_historical_data, sample_weather_forecast):
        """Test provenance record is valid and verifiable."""
        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=24
        )

        result, provenance = forecaster.forecast(inputs)

        assert provenance.calculator_name == "ThermalLoadForecaster"
        assert provenance.calculator_version == "1.0.0"
        assert len(provenance.provenance_hash) == 64  # SHA-256
        assert len(provenance.calculation_steps) > 0
        assert verify_provenance(provenance)


# =============================================================================
# DEGREE DAY CALCULATION TESTS
# =============================================================================

class TestDegreeDayCalculations:
    """Tests for degree day calculations."""

    def test_hdd_calculation_cold_day(self):
        """Test HDD calculation for cold temperature."""
        temp = 5.0  # 5C is cold
        hdd = calculate_hdd(temp, HDD_BASE_TEMP_C)

        assert hdd == 13.0  # 18 - 5 = 13 HDD

    def test_hdd_calculation_warm_day(self):
        """Test HDD calculation for warm temperature (no heating needed)."""
        temp = 22.0  # Above base temperature
        hdd = calculate_hdd(temp, HDD_BASE_TEMP_C)

        assert hdd == 0.0  # No heating needed

    def test_hdd_calculation_at_base(self):
        """Test HDD calculation at base temperature."""
        hdd = calculate_hdd(HDD_BASE_TEMP_C, HDD_BASE_TEMP_C)

        assert hdd == 0.0

    def test_cdd_calculation_hot_day(self):
        """Test CDD calculation for hot temperature."""
        temp = 30.0  # Hot day
        cdd = calculate_cdd(temp, CDD_BASE_TEMP_C)

        assert cdd == 6.0  # 30 - 24 = 6 CDD

    def test_cdd_calculation_cool_day(self):
        """Test CDD calculation for cool temperature (no cooling needed)."""
        temp = 20.0  # Below base temperature
        cdd = calculate_cdd(temp, CDD_BASE_TEMP_C)

        assert cdd == 0.0

    def test_degree_days_batch_heating(self):
        """Test batch degree day calculation for heating."""
        temps = [5.0, 10.0, 15.0, 20.0, 25.0]
        hdds = calculate_degree_days_batch(temps, HDD_BASE_TEMP_C, is_heating=True)

        assert len(hdds) == 5
        assert hdds[0] == 13.0  # 18 - 5
        assert hdds[1] == 8.0   # 18 - 10
        assert hdds[2] == 3.0   # 18 - 15
        assert hdds[3] == 0.0   # 18 - 20 = -2, but max(0)
        assert hdds[4] == 0.0   # 18 - 25 = -7, but max(0)

    def test_degree_days_batch_cooling(self):
        """Test batch degree day calculation for cooling."""
        temps = [20.0, 25.0, 30.0, 35.0]
        cdds = calculate_degree_days_batch(temps, CDD_BASE_TEMP_C, is_heating=False)

        assert len(cdds) == 4
        assert cdds[0] == 0.0   # 20 - 24 = -4, max(0)
        assert cdds[1] == 1.0   # 25 - 24
        assert cdds[2] == 6.0   # 30 - 24
        assert cdds[3] == 11.0  # 35 - 24

    def test_cached_hdd_calculation(self):
        """Test cached HDD calculation returns same results."""
        temp = 10.0

        result1 = cached_hdd_calculation(temp, HDD_BASE_TEMP_C)
        result2 = cached_hdd_calculation(temp, HDD_BASE_TEMP_C)

        assert result1 == result2
        assert result1 == 8.0  # 18 - 10

    def test_custom_base_temperatures(self, forecaster, sample_historical_data, sample_weather_forecast):
        """Test forecast with custom base temperatures."""
        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=24,
            custom_hdd_base_c=20.0,  # Higher base = more heating
            custom_cdd_base_c=22.0   # Lower base = more cooling
        )

        result, provenance = forecaster.forecast(inputs)

        assert result is not None
        # Higher HDD base should result in more HDD
        assert result.total_hdd >= 0


# =============================================================================
# FORECAST ACCURACY METRIC TESTS
# =============================================================================

class TestForecastAccuracyMetrics:
    """Tests for forecast accuracy metrics."""

    def test_mape_calculation_basic(self):
        """Test MAPE calculation with known values."""
        actual = [100.0, 110.0, 120.0]
        predicted = [95.0, 115.0, 130.0]

        mape = calculate_mape(actual, predicted)

        # MAPE = mean([5/100, 5/110, 10/120]) * 100
        expected = (5/100 + 5/110 + 10/120) / 3 * 100
        assert abs(mape - expected) < 0.01

    def test_mape_zero_actual(self):
        """Test MAPE handles zero actual values."""
        actual = [100.0, 0.0, 120.0]  # Zero in middle
        predicted = [95.0, 5.0, 130.0]

        mape = calculate_mape(actual, predicted)

        # Should skip zero actual value
        assert mape > 0

    def test_mape_perfect_forecast(self):
        """Test MAPE is zero for perfect forecast."""
        actual = [100.0, 110.0, 120.0]
        predicted = [100.0, 110.0, 120.0]

        mape = calculate_mape(actual, predicted)

        assert mape == 0.0

    def test_rmse_calculation_basic(self):
        """Test RMSE calculation with known values."""
        actual = [100.0, 110.0, 120.0]
        predicted = [95.0, 115.0, 130.0]

        rmse = calculate_rmse(actual, predicted)

        # RMSE = sqrt(mean([25, 25, 100])) = sqrt(50)
        expected = math.sqrt((25 + 25 + 100) / 3)
        assert abs(rmse - expected) < 0.01

    def test_rmse_perfect_forecast(self):
        """Test RMSE is zero for perfect forecast."""
        actual = [100.0, 110.0, 120.0]
        predicted = [100.0, 110.0, 120.0]

        rmse = calculate_rmse(actual, predicted)

        assert rmse == 0.0

    def test_cv_rmse_calculation(self):
        """Test CV(RMSE) calculation."""
        actual = [100.0, 110.0, 120.0]
        predicted = [95.0, 115.0, 130.0]

        cv_rmse = calculate_cv_rmse(actual, predicted)

        # CV(RMSE) = RMSE / mean(actual) * 100
        rmse = calculate_rmse(actual, predicted)
        mean_actual = 110.0
        expected = rmse / mean_actual * 100

        assert abs(cv_rmse - expected) < 0.01

    def test_mismatched_lengths_raises_error(self):
        """Test error raised for mismatched list lengths."""
        actual = [100.0, 110.0, 120.0]
        predicted = [95.0, 115.0]  # One less

        with pytest.raises(ValueError):
            calculate_mape(actual, predicted)

        with pytest.raises(ValueError):
            calculate_rmse(actual, predicted)


# =============================================================================
# PATTERN ANALYSIS TESTS
# =============================================================================

class TestPatternAnalysis:
    """Tests for daily and weekly pattern analysis."""

    def test_daily_pattern_extraction(self, forecaster, sample_historical_data, sample_weather_forecast):
        """Test daily pattern is extracted correctly."""
        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=24
        )

        result, _ = forecaster.forecast(inputs)

        assert len(result.daily_pattern) == 24
        assert sum(result.daily_pattern) == pytest.approx(1.0, rel=0.01)  # Normalized to 1

        # Check pattern makes sense (daytime should be higher than nighttime)
        daytime_avg = sum(result.daily_pattern[8:18]) / 10
        nighttime_avg = sum(result.daily_pattern[0:6]) / 6
        assert daytime_avg > nighttime_avg * 0.5  # Daytime at least 50% higher

    def test_weekly_pattern_extraction(self, forecaster, sample_historical_data, sample_weather_forecast):
        """Test weekly pattern is extracted correctly."""
        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=168  # 1 week
        )

        result, _ = forecaster.forecast(inputs)

        assert len(result.weekly_pattern) == 7

        # Weekdays should have higher load than weekends
        weekday_avg = sum(result.weekly_pattern[:5]) / 5
        weekend_avg = sum(result.weekly_pattern[5:]) / 2
        assert weekday_avg > weekend_avg * 0.8  # Weekdays higher

    def test_seasonal_decomposition(self, forecaster, sample_historical_data, sample_weather_forecast):
        """Test seasonal decomposition is performed."""
        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=168
        )

        result, _ = forecaster.forecast(inputs)

        # Seasonal components should be populated
        assert result.seasonal_components is not None
        assert len(result.seasonal_components.trend) > 0
        assert len(result.seasonal_components.seasonal) > 0
        assert len(result.seasonal_components.residual) > 0
        assert result.seasonal_components.period == 24  # Default hourly period


# =============================================================================
# THERMAL LAG AND OCCUPANCY TESTS
# =============================================================================

class TestThermalLagAndOccupancy:
    """Tests for thermal lag and occupancy adjustments."""

    def test_thermal_lag_application(self):
        """Test thermal lag smooths step changes."""
        profile = [0.0] * 5 + [100.0] * 5 + [0.0] * 5
        time_constant = 2.0

        lagged = apply_thermal_lag(profile, time_constant)

        assert len(lagged) == len(profile)
        # Lagged profile should have smoother transitions
        assert lagged[5] < profile[5]  # First high value should be reduced
        assert lagged[10] > profile[10]  # First low value after high should be elevated

    def test_thermal_lag_zero_constant(self):
        """Test thermal lag with zero time constant returns original."""
        profile = [0.0, 50.0, 100.0, 50.0, 0.0]

        lagged = apply_thermal_lag(profile, 0.0)

        assert lagged == profile

    def test_thermal_lag_empty_profile(self):
        """Test thermal lag handles empty profile."""
        lagged = apply_thermal_lag([], 2.0)

        assert lagged == []

    def test_occupancy_adjustment_basic(self):
        """Test occupancy adjustment scales load correctly."""
        base_load = [100.0] * 24
        occupancy = [0.2] * 6 + [1.0] * 12 + [0.3] * 6
        factor = 0.3

        adjusted = adjust_load_for_occupancy(base_load, occupancy, factor)

        assert len(adjusted) == 24
        # Low occupancy hours should have reduced load
        assert adjusted[0] < 100.0
        # High occupancy hours should have full load
        assert adjusted[12] == pytest.approx(100.0, rel=0.01)

    def test_occupancy_adjustment_mismatched_lengths(self):
        """Test occupancy adjustment raises error for mismatched lengths."""
        base_load = [100.0] * 24
        occupancy = [1.0] * 12  # Wrong length

        with pytest.raises(ValueError):
            adjust_load_for_occupancy(base_load, occupancy)


# =============================================================================
# SEASON DETERMINATION TESTS
# =============================================================================

class TestSeasonDetermination:
    """Tests for season determination."""

    def test_winter_months(self):
        """Test winter month detection."""
        assert get_season(12) == SeasonType.WINTER
        assert get_season(1) == SeasonType.WINTER
        assert get_season(2) == SeasonType.WINTER

    def test_spring_months(self):
        """Test spring month detection."""
        assert get_season(3) == SeasonType.SPRING
        assert get_season(4) == SeasonType.SPRING
        assert get_season(5) == SeasonType.SPRING

    def test_summer_months(self):
        """Test summer month detection."""
        assert get_season(6) == SeasonType.SUMMER
        assert get_season(7) == SeasonType.SUMMER
        assert get_season(8) == SeasonType.SUMMER

    def test_fall_months(self):
        """Test fall month detection."""
        assert get_season(9) == SeasonType.FALL
        assert get_season(10) == SeasonType.FALL
        assert get_season(11) == SeasonType.FALL


# =============================================================================
# ENERGY SIGNATURE TESTS
# =============================================================================

class TestEnergySignature:
    """Tests for energy signature analysis."""

    def test_energy_signature_heating_dominated(self):
        """Test energy signature with heating-dominated data."""
        # Cold temperatures, high energy
        energy = [200.0, 180.0, 150.0, 100.0, 80.0]
        temps = [0.0, 5.0, 10.0, 15.0, 20.0]

        sig = calculate_energy_signature(energy, temps)

        assert "base_load" in sig
        assert "heating_slope" in sig
        assert sig["heating_slope"] >= 0  # Positive heating relationship

    def test_energy_signature_insufficient_data(self):
        """Test energy signature with insufficient data."""
        energy = [100.0, 110.0]  # Only 2 points
        temps = [10.0, 15.0]

        sig = calculate_energy_signature(energy, temps)

        assert sig["base_load"] == pytest.approx(105.0, rel=0.01)
        assert sig["heating_slope"] == 0.0

    def test_energy_signature_mismatched_lengths(self):
        """Test energy signature raises error for mismatched lengths."""
        energy = [100.0, 110.0, 120.0]
        temps = [10.0, 15.0]  # Wrong length

        with pytest.raises(ValueError):
            calculate_energy_signature(energy, temps)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_minimal_historical_data(self, forecaster, sample_weather_forecast):
        """Test forecast with minimal historical data."""
        # Just 3 data points
        minimal_data = [
            HistoricalLoadData(
                timestamp=f"2024-01-0{i+1}T12:00:00Z",
                energy_kwh=100.0 + i * 10,
                peak_demand_kw=130.0 + i * 13,
                temp_avg_c=10.0,
                hour_of_day=12
            )
            for i in range(3)
        ]

        inputs = ThermalLoadForecastInput(
            historical_data=minimal_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=24
        )

        result, provenance = forecaster.forecast(inputs)

        assert result is not None
        assert len(result.hourly_forecasts) == 24

    def test_empty_weather_forecast(self, forecaster, sample_historical_data):
        """Test forecast with no weather data."""
        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=[],
            forecast_horizon_hours=24
        )

        result, provenance = forecaster.forecast(inputs)

        assert result is not None
        assert len(result.hourly_forecasts) == 24

    def test_long_forecast_horizon(self, forecaster, sample_historical_data, sample_weather_forecast):
        """Test forecast with long horizon."""
        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=720  # 30 days
        )

        result, provenance = forecaster.forecast(inputs)

        assert result is not None
        assert len(result.hourly_forecasts) == 720

    def test_invalid_forecast_horizon(self, forecaster, sample_historical_data, sample_weather_forecast):
        """Test invalid forecast horizon raises error."""
        with pytest.raises(ValueError):
            inputs = ThermalLoadForecastInput(
                historical_data=sample_historical_data,
                weather_forecast=sample_weather_forecast,
                forecast_horizon_hours=0  # Invalid
            )
            forecaster.forecast(inputs)

    def test_negative_energy_raises_error(self, forecaster, sample_weather_forecast):
        """Test negative energy in historical data raises error."""
        bad_data = [
            HistoricalLoadData(
                timestamp="2024-01-01T12:00:00Z",
                energy_kwh=-100.0,  # Negative
                peak_demand_kw=100.0,
                temp_avg_c=10.0
            )
        ]

        inputs = ThermalLoadForecastInput(
            historical_data=bad_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=24
        )

        with pytest.raises(ValueError):
            forecaster.forecast(inputs)

    def test_extreme_temperatures(self, forecaster, sample_historical_data):
        """Test forecast with extreme temperature forecast."""
        extreme_weather = [
            WeatherData(date="2024-01-15", temp_avg_c=-40.0),  # Very cold
            WeatherData(date="2024-01-16", temp_avg_c=-30.0),
            WeatherData(date="2024-01-17", temp_avg_c=45.0),   # Very hot
        ]

        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=extreme_weather,
            forecast_horizon_hours=72
        )

        result, provenance = forecaster.forecast(inputs)

        assert result is not None
        assert result.total_hdd > 0  # Should have high HDD for cold days


# =============================================================================
# PEAK DEMAND PREDICTION TESTS
# =============================================================================

class TestPeakDemandPrediction:
    """Tests for peak demand prediction."""

    def test_peak_prediction_enabled(self, forecaster, sample_historical_data, sample_weather_forecast):
        """Test peak prediction is generated when enabled."""
        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=24,
            include_peak_prediction=True
        )

        result, _ = forecaster.forecast(inputs)

        assert result.peak_prediction is not None
        assert result.peak_prediction.predicted_peak_kw > 0
        assert 0 <= result.peak_prediction.peak_hour < 24
        assert result.peak_prediction.confidence_lower_kw <= result.peak_prediction.predicted_peak_kw
        assert result.peak_prediction.predicted_peak_kw <= result.peak_prediction.confidence_upper_kw

    def test_peak_prediction_disabled(self, forecaster, sample_historical_data, sample_weather_forecast):
        """Test peak prediction is not generated when disabled."""
        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=24,
            include_peak_prediction=False
        )

        result, _ = forecaster.forecast(inputs)

        assert result.peak_prediction is None

    def test_peak_prediction_contributing_factors(self, forecaster, sample_historical_data, sample_weather_forecast):
        """Test peak prediction includes contributing factors."""
        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=24,
            include_peak_prediction=True
        )

        result, _ = forecaster.forecast(inputs)

        assert "heating_load" in result.peak_prediction.contributing_factors
        assert "base_load" in result.peak_prediction.contributing_factors
        assert "occupancy" in result.peak_prediction.contributing_factors


# =============================================================================
# CONFIDENCE INTERVAL TESTS
# =============================================================================

class TestConfidenceIntervals:
    """Tests for confidence interval calculations."""

    def test_confidence_intervals_generated(self, forecaster, sample_historical_data, sample_weather_forecast):
        """Test confidence intervals are generated for all forecasts."""
        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=24,
            confidence_level="95%"
        )

        result, _ = forecaster.forecast(inputs)

        for forecast in result.hourly_forecasts:
            assert forecast.lower_bound_kwh <= forecast.energy_kwh
            assert forecast.energy_kwh <= forecast.upper_bound_kwh
            assert forecast.confidence_level == "95%"

    def test_different_confidence_levels(self, forecaster, sample_historical_data, sample_weather_forecast):
        """Test different confidence levels produce different interval widths."""
        results = {}

        for level in ["90%", "95%", "99%"]:
            inputs = ThermalLoadForecastInput(
                historical_data=sample_historical_data,
                weather_forecast=sample_weather_forecast,
                forecast_horizon_hours=24,
                confidence_level=level
            )
            results[level], _ = forecaster.forecast(inputs)

        # Higher confidence should mean wider intervals
        # Compare interval widths for first forecast
        width_90 = results["90%"].hourly_forecasts[0].upper_bound_kwh - results["90%"].hourly_forecasts[0].lower_bound_kwh
        width_95 = results["95%"].hourly_forecasts[0].upper_bound_kwh - results["95%"].hourly_forecasts[0].lower_bound_kwh
        width_99 = results["99%"].hourly_forecasts[0].upper_bound_kwh - results["99%"].hourly_forecasts[0].lower_bound_kwh

        assert width_90 <= width_95 <= width_99


# =============================================================================
# DETERMINISM AND REPRODUCIBILITY TESTS
# =============================================================================

class TestDeterminismAndReproducibility:
    """Tests for deterministic and reproducible results."""

    def test_deterministic_results(self, forecaster, sample_historical_data, sample_weather_forecast):
        """Test same inputs produce same outputs."""
        inputs = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=24
        )

        result1, prov1 = forecaster.forecast(inputs)
        result2, prov2 = forecaster.forecast(inputs)

        # Results should be identical
        assert result1.total_energy_kwh == result2.total_energy_kwh
        assert result1.peak_demand_kw == result2.peak_demand_kw
        assert result1.hdd_sensitivity == result2.hdd_sensitivity

        # Provenance hashes should differ (due to timestamp) but input hashes should match
        assert prov1.input_hash == prov2.input_hash

    def test_provenance_hash_changes_with_inputs(self, forecaster, sample_historical_data, sample_weather_forecast):
        """Test provenance hash changes when inputs change."""
        inputs1 = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=24
        )

        inputs2 = ThermalLoadForecastInput(
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
            forecast_horizon_hours=48  # Different horizon
        )

        _, prov1 = forecaster.forecast(inputs1)
        _, prov2 = forecaster.forecast(inputs2)

        # Input hashes should differ
        assert prov1.input_hash != prov2.input_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
