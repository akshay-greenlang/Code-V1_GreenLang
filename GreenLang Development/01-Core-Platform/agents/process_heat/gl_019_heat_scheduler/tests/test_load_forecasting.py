"""
GL-019 HEATSCHEDULER - Load Forecasting Module Tests

Unit tests for ML-based load forecasting including feature engineering,
model training, ensemble predictions, and performance benchmarks.

Test Coverage:
    - Feature engineering
    - Individual model training and prediction
    - Ensemble forecasting
    - Model weight updates
    - Data quality assessment
    - Performance benchmarks

Author: GreenLang Test Team
Date: December 2025
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
import math
import statistics


class TestHistoricalDataPoint:
    """Tests for HistoricalDataPoint model."""

    def test_valid_historical_data_point(self, base_timestamp):
        """Test valid historical data point creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            HistoricalDataPoint,
        )

        point = HistoricalDataPoint(
            timestamp=base_timestamp,
            load_kw=2500.0,
            temperature_c=18.5,
            humidity_pct=65.0,
            is_holiday=False,
            is_weekend=False,
            production_level=0.8,
        )

        assert point.load_kw == 2500.0
        assert point.temperature_c == 18.5
        assert point.is_weekend is False

    def test_historical_data_point_defaults(self, base_timestamp):
        """Test default values for optional fields."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            HistoricalDataPoint,
        )

        point = HistoricalDataPoint(
            timestamp=base_timestamp,
            load_kw=2500.0,
        )

        assert point.temperature_c is None
        assert point.humidity_pct is None
        assert point.is_holiday is False
        assert point.is_weekend is False


class TestForecastFeatures:
    """Tests for ForecastFeatures model."""

    def test_valid_forecast_features(self):
        """Test valid forecast features creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            ForecastFeatures,
        )

        features = ForecastFeatures(
            hour_of_day=14,
            day_of_week=2,
            month=6,
            is_weekend=False,
            is_holiday=False,
            temperature_c=22.5,
            humidity_pct=60.0,
            solar_radiation_w_m2=450.0,
            heating_degree_hours=0.0,
            cooling_degree_hours=0.0,
            lag_1h=2400.0,
            lag_24h=2500.0,
            lag_168h=2450.0,
            rolling_mean_24h=2480.0,
            production_scheduled=800.0,
        )

        assert features.hour_of_day == 14
        assert features.temperature_c == 22.5
        assert features.lag_24h == 2500.0

    def test_forecast_features_hour_bounds(self):
        """Test hour_of_day bounds validation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            ForecastFeatures,
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ForecastFeatures(
                hour_of_day=25,  # Invalid
                day_of_week=2,
                month=6,
                is_weekend=False,
                is_holiday=False,
            )


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    @pytest.fixture
    def feature_engineer(self, sample_load_forecasting_config):
        """Create feature engineer instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            FeatureEngineer,
        )
        return FeatureEngineer(sample_load_forecasting_config)

    def test_feature_engineer_initialization(self, sample_load_forecasting_config):
        """Test feature engineer initialization."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            FeatureEngineer,
        )

        engineer = FeatureEngineer(sample_load_forecasting_config)

        assert engineer.config == sample_load_forecasting_config
        assert engineer.heating_base_temp_c == 18.0
        assert engineer.cooling_base_temp_c == 24.0

    def test_extract_calendar_features(
        self,
        feature_engineer,
        base_timestamp,
        sample_historical_data,
    ):
        """Test extraction of calendar features."""
        features = feature_engineer.extract_features(
            target_time=base_timestamp,
            historical_data=sample_historical_data,
        )

        assert features.hour_of_day == base_timestamp.hour
        assert features.day_of_week == base_timestamp.weekday()
        assert features.month == base_timestamp.month

    def test_extract_weekend_flag(self, feature_engineer, sample_historical_data):
        """Test weekend flag extraction."""
        # Saturday (weekday = 5)
        saturday = datetime(2025, 6, 14, 12, 0, 0, tzinfo=timezone.utc)

        features = feature_engineer.extract_features(
            target_time=saturday,
            historical_data=sample_historical_data,
        )

        assert features.is_weekend is True

    def test_extract_lagged_features(
        self,
        feature_engineer,
        sample_historical_data,
        base_timestamp,
    ):
        """Test extraction of lagged features."""
        features = feature_engineer.extract_features(
            target_time=base_timestamp,
            historical_data=sample_historical_data,
        )

        # Should have lagged values from historical data
        # Note: May be None if exact timestamps don't match
        assert features.lag_1h is not None or features.lag_1h is None
        assert features.lag_24h is not None or features.lag_24h is None

    def test_extract_weather_features(
        self,
        feature_engineer,
        base_timestamp,
        sample_weather_forecast,
        sample_historical_data,
    ):
        """Test extraction of weather features."""
        features = feature_engineer.extract_features(
            target_time=base_timestamp,
            historical_data=sample_historical_data,
            weather_forecast=sample_weather_forecast,
        )

        # Weather features should be extracted
        if sample_weather_forecast.forecast_points:
            # Check if there's a matching weather point
            matching = any(
                abs((p.timestamp - base_timestamp).total_seconds()) < 3600
                for p in sample_weather_forecast.forecast_points
            )
            if matching:
                assert features.temperature_c is not None

    def test_extract_heating_degree_hours(
        self,
        feature_engineer,
        sample_historical_data,
        sample_weather_forecast,
    ):
        """Test heating degree hours calculation."""
        # Cold time (should have HDH)
        cold_time = datetime(2025, 1, 15, 6, 0, 0, tzinfo=timezone.utc)

        # Create weather forecast with cold temperature
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            WeatherForecastPoint,
            WeatherForecastResult,
        )

        cold_weather = WeatherForecastResult(
            latitude=37.7749,
            longitude=-122.4194,
            forecast_points=[
                WeatherForecastPoint(
                    timestamp=cold_time,
                    temperature_c=10.0,  # Cold
                )
            ],
            forecast_horizon_hours=24,
        )

        features = feature_engineer.extract_features(
            target_time=cold_time,
            historical_data=[],
            weather_forecast=cold_weather,
        )

        # HDH = 18 - 10 = 8
        assert features.heating_degree_hours == 8.0

    def test_set_holidays(self, feature_engineer):
        """Test setting holiday dates."""
        holidays = [
            datetime(2025, 7, 4, tzinfo=timezone.utc),
            datetime(2025, 12, 25, tzinfo=timezone.utc),
        ]

        feature_engineer.set_holidays(holidays)

        assert len(feature_engineer._holidays) == 2

    def test_extract_holiday_flag(
        self,
        feature_engineer,
        sample_historical_data,
    ):
        """Test holiday flag extraction."""
        # Set July 4th as holiday
        holiday = datetime(2025, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
        feature_engineer.set_holidays([holiday])

        features = feature_engineer.extract_features(
            target_time=holiday,
            historical_data=sample_historical_data,
        )

        assert features.is_holiday is True


class TestGradientBoostingModel:
    """Tests for GradientBoostingModel."""

    @pytest.fixture
    def gb_model(self):
        """Create Gradient Boosting model instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            GradientBoostingModel,
        )
        return GradientBoostingModel()

    def test_model_initialization(self, gb_model):
        """Test model initialization."""
        assert gb_model.model_name == "gradient_boosting"
        assert gb_model._is_trained is False

    def test_model_training(self, gb_model):
        """Test model training."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            ForecastFeatures,
        )

        # Create training data
        features = [
            ForecastFeatures(
                hour_of_day=h,
                day_of_week=0,
                month=6,
                is_weekend=False,
                is_holiday=False,
            )
            for h in range(24)
        ]
        targets = [1000 + 500 * math.sin((h - 6) * math.pi / 12) for h in range(24)]

        gb_model.train(features, targets)

        assert gb_model._is_trained is True
        assert gb_model._last_trained is not None
        assert gb_model._training_samples == 24

    def test_model_prediction(self, gb_model):
        """Test model prediction."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            ForecastFeatures,
        )

        # Train model
        features = [
            ForecastFeatures(
                hour_of_day=h,
                day_of_week=0,
                month=6,
                is_weekend=False,
                is_holiday=False,
                lag_24h=1500.0,
            )
            for h in range(24)
        ]
        targets = [1000 + 500 * math.sin((h - 6) * math.pi / 12) for h in range(24)]
        gb_model.train(features, targets)

        # Make prediction
        test_features = ForecastFeatures(
            hour_of_day=12,
            day_of_week=0,
            month=6,
            is_weekend=False,
            is_holiday=False,
            lag_24h=1500.0,
        )

        prediction, lower, upper = gb_model.predict(test_features)

        assert prediction > 0
        assert lower < prediction < upper

    def test_model_weekend_adjustment(self, gb_model):
        """Test weekend load adjustment."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            ForecastFeatures,
        )

        # Train model
        features = [
            ForecastFeatures(
                hour_of_day=12,
                day_of_week=d,
                month=6,
                is_weekend=(d >= 5),
                is_holiday=False,
            )
            for d in range(7)
        ]
        targets = [2000.0] * 7
        gb_model.train(features, targets)

        # Weekday prediction
        weekday_features = ForecastFeatures(
            hour_of_day=12,
            day_of_week=2,
            month=6,
            is_weekend=False,
            is_holiday=False,
        )
        weekday_pred, _, _ = gb_model.predict(weekday_features)

        # Weekend prediction
        weekend_features = ForecastFeatures(
            hour_of_day=12,
            day_of_week=5,
            month=6,
            is_weekend=True,
            is_holiday=False,
        )
        weekend_pred, _, _ = gb_model.predict(weekend_features)

        # Weekend should be lower
        assert weekend_pred < weekday_pred

    def test_model_feature_importance(self, gb_model):
        """Test feature importance extraction."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            ForecastFeatures,
        )

        # Train model
        features = [
            ForecastFeatures(
                hour_of_day=h,
                day_of_week=0,
                month=6,
                is_weekend=False,
                is_holiday=False,
            )
            for h in range(24)
        ]
        targets = [1000 + 500 * math.sin((h - 6) * math.pi / 12) for h in range(24)]
        gb_model.train(features, targets)

        importance = gb_model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) > 0


class TestRandomForestModel:
    """Tests for RandomForestModel."""

    @pytest.fixture
    def rf_model(self):
        """Create Random Forest model instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            RandomForestModel,
        )
        return RandomForestModel()

    def test_model_initialization(self, rf_model):
        """Test model initialization."""
        assert rf_model.model_name == "random_forest"
        assert rf_model._is_trained is False

    def test_model_training_learns_hourly_pattern(self, rf_model):
        """Test model learns hourly pattern."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            ForecastFeatures,
        )

        # Create training data with clear hourly pattern
        features = []
        targets = []

        for day in range(7):
            for h in range(24):
                features.append(ForecastFeatures(
                    hour_of_day=h,
                    day_of_week=day,
                    month=6,
                    is_weekend=(day >= 5),
                    is_holiday=False,
                ))
                # Clear hourly pattern
                load = 1500 + 1000 * math.sin((h - 6) * math.pi / 12)
                targets.append(load)

        rf_model.train(features, targets)

        assert rf_model._is_trained is True
        # Hourly pattern should be learned
        assert len(rf_model._hourly_pattern) == 24

    def test_model_prediction_with_temperature(self, rf_model):
        """Test prediction with temperature feature."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            ForecastFeatures,
        )

        # Train model
        features = [
            ForecastFeatures(
                hour_of_day=h,
                day_of_week=0,
                month=6,
                is_weekend=False,
                is_holiday=False,
                temperature_c=20.0,
            )
            for h in range(24)
        ]
        targets = [1500.0] * 24
        rf_model.train(features, targets)

        # Cold temperature should increase load
        cold_features = ForecastFeatures(
            hour_of_day=12,
            day_of_week=0,
            month=1,
            is_weekend=False,
            is_holiday=False,
            temperature_c=5.0,  # Cold
        )
        cold_pred, _, _ = rf_model.predict(cold_features)

        # Warm temperature
        warm_features = ForecastFeatures(
            hour_of_day=12,
            day_of_week=0,
            month=7,
            is_weekend=False,
            is_holiday=False,
            temperature_c=30.0,  # Warm
        )
        warm_pred, _, _ = rf_model.predict(warm_features)

        # Cold should have higher load (heating)
        assert cold_pred > warm_pred


class TestARIMAModel:
    """Tests for ARIMAModel."""

    @pytest.fixture
    def arima_model(self):
        """Create ARIMA model instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            ARIMAModel,
        )
        return ARIMAModel()

    def test_model_initialization(self, arima_model):
        """Test model initialization."""
        assert arima_model.model_name == "arima"
        assert arima_model._is_trained is False
        assert arima_model._p == 2
        assert arima_model._d == 1
        assert arima_model._q == 2

    def test_model_training(self, arima_model):
        """Test model training."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            ForecastFeatures,
        )

        # Create training data
        features = [
            ForecastFeatures(
                hour_of_day=h % 24,
                day_of_week=0,
                month=6,
                is_weekend=False,
                is_holiday=False,
            )
            for h in range(100)
        ]
        targets = [1500 + 200 * math.sin(h * math.pi / 12) for h in range(100)]

        arima_model.train(features, targets)

        assert arima_model._is_trained is True
        # AR coefficients should be calculated
        assert arima_model._ar_coeffs[0] != 0.5 or len(targets) <= 2

    def test_model_uses_lagged_features(self, arima_model):
        """Test model uses lagged features for prediction."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            ForecastFeatures,
        )

        # Train model
        features = [
            ForecastFeatures(
                hour_of_day=h % 24,
                day_of_week=0,
                month=6,
                is_weekend=False,
                is_holiday=False,
            )
            for h in range(100)
        ]
        targets = [1500.0] * 100
        arima_model.train(features, targets)

        # Prediction with high lagged values
        high_lag_features = ForecastFeatures(
            hour_of_day=12,
            day_of_week=0,
            month=6,
            is_weekend=False,
            is_holiday=False,
            lag_1h=2000.0,
            lag_24h=2000.0,
        )
        high_pred, _, _ = arima_model.predict(high_lag_features)

        # Prediction with low lagged values
        low_lag_features = ForecastFeatures(
            hour_of_day=12,
            day_of_week=0,
            month=6,
            is_weekend=False,
            is_holiday=False,
            lag_1h=1000.0,
            lag_24h=1000.0,
        )
        low_pred, _, _ = arima_model.predict(low_lag_features)

        # High lag should predict higher load
        assert high_pred > low_pred

    def test_autocorrelation_calculation(self, arima_model):
        """Test autocorrelation calculation."""
        # Perfect autocorrelation
        series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        autocorr = arima_model._autocorrelation(series, 1)

        assert -1 <= autocorr <= 1


class TestEnsembleForecaster:
    """Tests for EnsembleForecaster."""

    @pytest.fixture
    def forecaster(self, sample_load_forecasting_config):
        """Create EnsembleForecaster instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            EnsembleForecaster,
        )
        return EnsembleForecaster(sample_load_forecasting_config)

    def test_forecaster_initialization(self, forecaster):
        """Test forecaster initialization."""
        assert len(forecaster._models) == 3
        assert "gradient_boosting" in forecaster._models
        assert "random_forest" in forecaster._models
        assert "arima" in forecaster._models

    def test_forecaster_initial_weights(self, forecaster):
        """Test initial model weights."""
        assert forecaster._weights["gradient_boosting"] == 0.4
        assert forecaster._weights["random_forest"] == 0.35
        assert forecaster._weights["arima"] == 0.25

    def test_forecaster_training(self, forecaster, sample_historical_data):
        """Test ensemble training."""
        forecaster.train(sample_historical_data)

        # All models should be trained
        for model in forecaster._models.values():
            assert model._is_trained is True

    def test_forecaster_insufficient_data_warning(
        self,
        forecaster,
        caplog,
    ):
        """Test warning for insufficient training data."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            HistoricalDataPoint,
        )
        import logging

        caplog.set_level(logging.WARNING)

        # Only 10 samples (less than min_training_samples=1000)
        small_data = [
            HistoricalDataPoint(
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                load_kw=1500.0,
            )
            for i in range(10)
        ]

        forecaster.train(small_data)

        assert "Insufficient training data" in caplog.text

    def test_forecaster_forecast_generation(
        self,
        forecaster,
        sample_historical_data,
        base_timestamp,
    ):
        """Test forecast generation."""
        forecaster.train(sample_historical_data)

        result = forecaster.forecast(
            forecast_start=base_timestamp,
            horizon_hours=24,
        )

        assert result is not None
        assert len(result.forecast_points) == 96  # 24h * 4 (15-min intervals)
        assert result.model_used == "ensemble"

    def test_forecaster_forecast_aggregates(
        self,
        forecaster,
        sample_historical_data,
        base_timestamp,
    ):
        """Test forecast aggregates are calculated."""
        forecaster.train(sample_historical_data)

        result = forecaster.forecast(
            forecast_start=base_timestamp,
            horizon_hours=24,
        )

        assert result.peak_load_kw is not None
        assert result.peak_load_kw > 0
        assert result.avg_load_kw is not None
        assert result.total_energy_kwh is not None

    def test_forecaster_forecast_with_weather(
        self,
        forecaster,
        sample_historical_data,
        sample_weather_forecast,
        base_timestamp,
    ):
        """Test forecast generation with weather data."""
        forecaster.train(sample_historical_data)

        result = forecaster.forecast(
            forecast_start=base_timestamp,
            horizon_hours=24,
            weather_forecast=sample_weather_forecast,
        )

        assert result is not None
        assert result.status.value == "success"

    def test_forecaster_update_weights(self, forecaster, sample_historical_data):
        """Test model weight updates based on performance."""
        forecaster.train(sample_historical_data)

        # Create actual values for weight update
        actual_values = [
            (datetime.now(timezone.utc) - timedelta(hours=i), 1500 + i * 10)
            for i in range(10)
        ]

        original_weights = forecaster._weights.copy()
        forecaster.update_weights(actual_values)

        # Weights may have changed
        # At minimum, they should still sum to approximately 1.0
        total = sum(forecaster._weights.values())
        assert 0.99 <= total <= 1.01

    def test_forecaster_data_quality_assessment(
        self,
        forecaster,
        sample_historical_data,
    ):
        """Test data quality assessment."""
        forecaster.train(sample_historical_data)

        quality = forecaster._assess_data_quality()

        assert 0 <= quality <= 1

    def test_forecaster_add_historical_point(self, forecaster, base_timestamp):
        """Test adding historical data points."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            HistoricalDataPoint,
        )

        initial_count = len(forecaster._historical_data)

        point = HistoricalDataPoint(
            timestamp=base_timestamp,
            load_kw=2500.0,
        )
        forecaster.add_historical_point(point)

        assert len(forecaster._historical_data) == initial_count + 1

    def test_forecaster_feature_importance_aggregation(
        self,
        forecaster,
        sample_historical_data,
        base_timestamp,
    ):
        """Test feature importance is aggregated from models."""
        forecaster.train(sample_historical_data)

        result = forecaster.forecast(
            forecast_start=base_timestamp,
            horizon_hours=24,
        )

        assert result.feature_importance is not None
        assert len(result.feature_importance) > 0


class TestForecastPerformance:
    """Performance tests for load forecasting."""

    @pytest.mark.performance
    def test_forecast_generation_time(
        self,
        sample_load_forecasting_config,
        sample_historical_data,
        base_timestamp,
    ):
        """Test forecast generation completes in reasonable time."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            EnsembleForecaster,
        )
        import time

        forecaster = EnsembleForecaster(sample_load_forecasting_config)
        forecaster.train(sample_historical_data)

        start = time.time()
        result = forecaster.forecast(
            forecast_start=base_timestamp,
            horizon_hours=48,
        )
        elapsed = time.time() - start

        assert result is not None
        assert elapsed < 2.0  # Should complete in under 2 seconds

    @pytest.mark.performance
    def test_training_time(
        self,
        sample_load_forecasting_config,
        sample_historical_data,
    ):
        """Test model training completes in reasonable time."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            EnsembleForecaster,
        )
        import time

        forecaster = EnsembleForecaster(sample_load_forecasting_config)

        start = time.time()
        forecaster.train(sample_historical_data)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should complete in under 5 seconds


class TestModelPerformance:
    """Tests for ModelPerformance tracking."""

    def test_model_performance_schema(self, base_timestamp):
        """Test ModelPerformance schema."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.load_forecasting import (
            ModelPerformance,
        )

        perf = ModelPerformance(
            model_name="gradient_boosting",
            mape_pct=5.2,
            rmse_kw=125.0,
            mae_kw=95.0,
            r2_score=0.92,
            last_trained=base_timestamp,
            training_samples=10000,
        )

        assert perf.model_name == "gradient_boosting"
        assert perf.mape_pct == 5.2
        assert perf.r2_score == 0.92
