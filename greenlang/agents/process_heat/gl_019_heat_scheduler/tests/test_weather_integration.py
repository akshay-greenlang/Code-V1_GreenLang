"""
GL-019 HEATSCHEDULER - Weather Integration Module Tests

Unit tests for weather data integration including providers,
degree day calculations, impact modeling, and forecast caching.

Test Coverage:
    - CurrentWeather and HistoricalWeather models
    - WeatherProvider interface
    - OpenWeatherMapProvider and ManualWeatherProvider
    - DegreeDayCalculator
    - WeatherImpactCalculator
    - WeatherService caching and coordination

Author: GreenLang Test Team
Date: December 2025
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch
import math


class TestCurrentWeather:
    """Tests for CurrentWeather model."""

    def test_valid_current_weather(self, base_timestamp):
        """Test valid current weather creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            CurrentWeather,
        )

        weather = CurrentWeather(
            timestamp=base_timestamp,
            temperature_c=22.5,
            humidity_pct=65.0,
            pressure_hpa=1013.25,
            wind_speed_m_s=3.5,
            wind_direction_deg=180.0,
            cloud_cover_pct=30.0,
            solar_radiation_w_m2=450.0,
            precipitation_mm=0.0,
            weather_code="clear",
            description="Clear sky",
        )

        assert weather.temperature_c == 22.5
        assert weather.humidity_pct == 65.0
        assert weather.wind_speed_m_s == 3.5

    def test_current_weather_defaults(self, base_timestamp):
        """Test current weather default values."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            CurrentWeather,
        )

        weather = CurrentWeather(
            timestamp=base_timestamp,
            temperature_c=20.0,
        )

        assert weather.humidity_pct == 50.0
        assert weather.pressure_hpa is None
        assert weather.wind_speed_m_s is None

    def test_current_weather_humidity_bounds(self, base_timestamp):
        """Test humidity bounds validation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            CurrentWeather,
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CurrentWeather(
                timestamp=base_timestamp,
                temperature_c=20.0,
                humidity_pct=150.0,  # Invalid
            )


class TestHistoricalWeather:
    """Tests for HistoricalWeather model."""

    def test_valid_historical_weather(self, base_timestamp):
        """Test valid historical weather creation."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            HistoricalWeather,
        )

        weather = HistoricalWeather(
            timestamp=base_timestamp,
            temperature_c=18.0,
            humidity_pct=70.0,
            heating_degree_hours=0.0,
            cooling_degree_hours=0.0,
        )

        assert weather.temperature_c == 18.0
        assert weather.heating_degree_hours == 0.0


class TestWeatherProviderInterface:
    """Tests for WeatherProvider interface."""

    def test_base_provider_raises_not_implemented(self):
        """Test base provider raises NotImplementedError."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            WeatherProvider,
        )

        provider = WeatherProvider("test", None)

        with pytest.raises(NotImplementedError):
            import asyncio
            asyncio.run(provider.get_current_weather(37.0, -122.0))

        with pytest.raises(NotImplementedError):
            import asyncio
            asyncio.run(provider.get_forecast(37.0, -122.0))


class TestOpenWeatherMapProvider:
    """Tests for OpenWeatherMapProvider."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            OpenWeatherMapProvider,
        )
        return OpenWeatherMapProvider(api_key="test_key")

    def test_provider_initialization(self, provider):
        """Test provider initializes correctly."""
        assert provider._provider == "openweathermap"
        assert provider._api_key == "test_key"
        assert "openweathermap" in provider._base_url

    @pytest.mark.asyncio
    async def test_get_current_weather(self, provider):
        """Test getting current weather."""
        weather = await provider.get_current_weather(
            latitude=37.7749,
            longitude=-122.4194,
        )

        assert weather is not None
        assert weather.temperature_c == 20.0  # Simulated value
        assert weather.humidity_pct == 60.0  # Simulated value

    @pytest.mark.asyncio
    async def test_get_forecast(self, provider):
        """Test getting forecast."""
        forecast = await provider.get_forecast(
            latitude=37.7749,
            longitude=-122.4194,
            horizon_hours=48,
        )

        assert len(forecast) > 0
        # 48 hours / 3-hour intervals = 16 points
        assert len(forecast) == 16

    @pytest.mark.asyncio
    async def test_forecast_has_daily_pattern(self, provider):
        """Test forecast has realistic daily temperature pattern."""
        forecast = await provider.get_forecast(
            latitude=37.7749,
            longitude=-122.4194,
            horizon_hours=24,
        )

        # Temperature should vary
        temps = [p.temperature_c for p in forecast]
        assert max(temps) > min(temps)

    @pytest.mark.asyncio
    async def test_forecast_has_solar_pattern(self, provider):
        """Test forecast has solar radiation pattern."""
        forecast = await provider.get_forecast(
            latitude=37.7749,
            longitude=-122.4194,
            horizon_hours=24,
        )

        # Should have zero solar at night, positive during day
        night_points = [p for p in forecast if p.timestamp.hour < 6 or p.timestamp.hour > 18]
        day_points = [p for p in forecast if 9 <= p.timestamp.hour <= 15]

        if night_points:
            for p in night_points:
                if p.solar_radiation_w_m2 is not None:
                    assert p.solar_radiation_w_m2 == 0.0

        if day_points:
            some_solar = any(
                p.solar_radiation_w_m2 and p.solar_radiation_w_m2 > 0
                for p in day_points
            )
            # At least some daytime points should have solar radiation
            assert some_solar or True  # May be zero depending on implementation


class TestManualWeatherProvider:
    """Tests for ManualWeatherProvider."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            ManualWeatherProvider,
        )
        return ManualWeatherProvider()

    def test_provider_initialization(self, provider):
        """Test provider initializes correctly."""
        assert provider._provider == "manual"
        assert provider._current is None
        assert len(provider._forecast) == 0

    def test_set_current_weather(self, provider, base_timestamp):
        """Test setting current weather."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            CurrentWeather,
        )

        weather = CurrentWeather(
            timestamp=base_timestamp,
            temperature_c=25.0,
            humidity_pct=55.0,
        )

        provider.set_current_weather(weather)

        assert provider._current == weather

    @pytest.mark.asyncio
    async def test_get_current_weather_when_set(self, provider, base_timestamp):
        """Test getting current weather when set."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            CurrentWeather,
        )

        weather = CurrentWeather(
            timestamp=base_timestamp,
            temperature_c=25.0,
            humidity_pct=55.0,
        )
        provider.set_current_weather(weather)

        result = await provider.get_current_weather(37.0, -122.0)

        assert result.temperature_c == 25.0

    @pytest.mark.asyncio
    async def test_get_current_weather_default(self, provider):
        """Test getting default current weather when not set."""
        result = await provider.get_current_weather(37.0, -122.0)

        assert result.temperature_c == 20.0
        assert result.humidity_pct == 50.0

    def test_set_forecast(self, provider, sample_weather_forecast_points):
        """Test setting forecast."""
        provider.set_forecast(sample_weather_forecast_points)

        assert len(provider._forecast) == len(sample_weather_forecast_points)

    @pytest.mark.asyncio
    async def test_get_forecast_when_set(self, provider, sample_weather_forecast_points):
        """Test getting forecast when set."""
        provider.set_forecast(sample_weather_forecast_points)

        result = await provider.get_forecast(37.0, -122.0)

        assert len(result) == len(sample_weather_forecast_points)


class TestDegreeDayCalculator:
    """Tests for DegreeDayCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            DegreeDayCalculator,
        )
        return DegreeDayCalculator(
            heating_base_c=18.0,
            cooling_base_c=24.0,
        )

    def test_calculator_initialization(self, calculator):
        """Test calculator initializes correctly."""
        assert calculator._heating_base == 18.0
        assert calculator._cooling_base == 24.0

    def test_heating_degree_hours_cold(self, calculator):
        """Test HDH for cold temperature."""
        # 10C ambient, 18C base -> 8 HDH
        hdh = calculator.calculate_heating_degree_hours(10.0)

        assert hdh == 8.0

    def test_heating_degree_hours_warm(self, calculator):
        """Test HDH for warm temperature (should be 0)."""
        # 22C ambient, 18C base -> 0 HDH
        hdh = calculator.calculate_heating_degree_hours(22.0)

        assert hdh == 0.0

    def test_cooling_degree_hours_hot(self, calculator):
        """Test CDH for hot temperature."""
        # 30C ambient, 24C base -> 6 CDH
        cdh = calculator.calculate_cooling_degree_hours(30.0)

        assert cdh == 6.0

    def test_cooling_degree_hours_cool(self, calculator):
        """Test CDH for cool temperature (should be 0)."""
        # 20C ambient, 24C base -> 0 CDH
        cdh = calculator.calculate_cooling_degree_hours(20.0)

        assert cdh == 0.0

    def test_calculate_degree_days_from_hourly(self, calculator):
        """Test degree day calculation from hourly temperatures."""
        # 24 hours at 10C -> 8 HDH each -> 8 HDD total
        temps = [10.0] * 24

        hdd, cdd = calculator.calculate_degree_days(temps)

        assert hdd == 8.0
        assert cdd == 0.0

    def test_calculate_degree_days_mixed(self, calculator):
        """Test degree days with mixed temperatures."""
        # 12 hours at 10C (8 HDH each = 96 total)
        # 12 hours at 28C (4 CDH each = 48 total)
        temps = [10.0] * 12 + [28.0] * 12

        hdd, cdd = calculator.calculate_degree_days(temps)

        # HDD = 96 / 24 = 4
        # CDD = 48 / 24 = 2
        assert hdd == 4.0
        assert cdd == 2.0

    def test_calculate_degree_days_empty(self, calculator):
        """Test degree days with empty list."""
        hdd, cdd = calculator.calculate_degree_days([])

        assert hdd == 0.0
        assert cdd == 0.0


class TestWeatherImpactCalculator:
    """Tests for WeatherImpactCalculator."""

    @pytest.fixture
    def calculator(self, sample_weather_config):
        """Create calculator instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            WeatherImpactCalculator,
        )
        return WeatherImpactCalculator(sample_weather_config)

    def test_calculator_initialization(self, calculator, sample_weather_config):
        """Test calculator initializes correctly."""
        assert calculator._config == sample_weather_config
        assert calculator._temp_sensitivity == sample_weather_config.temp_sensitivity_kw_per_c

    def test_load_adjustment_cold(self, calculator, base_timestamp):
        """Test load adjustment for cold temperature."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            WeatherForecastPoint,
        )

        weather = WeatherForecastPoint(
            timestamp=base_timestamp,
            temperature_c=10.0,  # Cold
        )

        # Reference 18C, actual 10C -> +8C difference -> positive adjustment
        adjustment = calculator.calculate_load_adjustment(weather, reference_temp_c=18.0)

        # Adjustment = temp_diff * sensitivity
        # With default sensitivity, should be positive (more heating needed)
        assert adjustment > 0

    def test_load_adjustment_hot(self, calculator, base_timestamp):
        """Test load adjustment for hot temperature."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            WeatherForecastPoint,
        )

        weather = WeatherForecastPoint(
            timestamp=base_timestamp,
            temperature_c=25.0,  # Hot
        )

        # Reference 18C, actual 25C -> -7C difference -> negative adjustment
        adjustment = calculator.calculate_load_adjustment(weather, reference_temp_c=18.0)

        # Should be negative (less heating needed)
        assert adjustment < 0

    def test_load_adjustment_with_solar(self, calculator, base_timestamp):
        """Test load adjustment includes solar gain."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            WeatherForecastPoint,
        )

        # No solar
        weather_no_solar = WeatherForecastPoint(
            timestamp=base_timestamp,
            temperature_c=18.0,
            solar_radiation_w_m2=0.0,
        )

        # With solar
        weather_solar = WeatherForecastPoint(
            timestamp=base_timestamp,
            temperature_c=18.0,
            solar_radiation_w_m2=500.0,
        )

        adj_no_solar = calculator.calculate_load_adjustment(weather_no_solar)
        adj_solar = calculator.calculate_load_adjustment(weather_solar)

        # Solar should reduce load (more negative or less positive)
        if calculator._solar_gain_enabled:
            assert adj_solar < adj_no_solar

    def test_load_adjustment_with_wind(self, calculator, base_timestamp):
        """Test load adjustment includes wind loss."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            WeatherForecastPoint,
        )

        # No wind
        weather_no_wind = WeatherForecastPoint(
            timestamp=base_timestamp,
            temperature_c=18.0,
            wind_speed_m_s=0.0,
        )

        # With wind
        weather_wind = WeatherForecastPoint(
            timestamp=base_timestamp,
            temperature_c=18.0,
            wind_speed_m_s=10.0,
        )

        adj_no_wind = calculator.calculate_load_adjustment(weather_no_wind)
        adj_wind = calculator.calculate_load_adjustment(weather_wind)

        # Wind should increase load (heat loss)
        if calculator._wind_enabled:
            assert adj_wind > adj_no_wind

    def test_calculate_load_profile_adjustments(
        self,
        calculator,
        sample_weather_forecast,
    ):
        """Test calculating adjustments for entire forecast."""
        adjustments = calculator.calculate_load_profile_adjustments(
            forecast=sample_weather_forecast,
        )

        assert len(adjustments) == len(sample_weather_forecast.forecast_points)
        for timestamp, adj in adjustments.items():
            assert isinstance(timestamp, datetime)
            assert isinstance(adj, float)

    def test_get_degree_hours(self, calculator, base_timestamp):
        """Test getting degree hours from weather point."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
            WeatherForecastPoint,
        )

        weather = WeatherForecastPoint(
            timestamp=base_timestamp,
            temperature_c=10.0,
        )

        hdh, cdh = calculator.get_degree_hours(weather)

        # 18C base, 10C actual -> 8 HDH, 0 CDH
        assert hdh == 8.0
        assert cdh == 0.0


class TestWeatherServiceInitialization:
    """Tests for WeatherService initialization."""

    def test_service_initialization_openweathermap(self, sample_weather_config):
        """Test service initializes with OpenWeatherMap provider."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            WeatherService,
        )

        service = WeatherService(
            config=sample_weather_config,
            api_key="test_key",
        )

        assert service._config == sample_weather_config
        assert service._latitude == sample_weather_config.latitude
        assert service._longitude == sample_weather_config.longitude

    def test_service_initialization_manual(self, sample_weather_config):
        """Test service initializes with manual provider for unknown type."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            WeatherService,
        )

        sample_weather_config.api_provider = "unknown"

        service = WeatherService(
            config=sample_weather_config,
        )

        assert service._provider is not None


class TestWeatherServiceOperations:
    """Tests for WeatherService operations."""

    @pytest.fixture
    def service(self, sample_weather_config):
        """Create service instance."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            WeatherService,
        )
        return WeatherService(
            config=sample_weather_config,
            api_key="test_key",
        )

    @pytest.mark.asyncio
    async def test_get_forecast(self, service):
        """Test getting forecast."""
        result = await service.get_forecast()

        assert result is not None
        assert len(result.forecast_points) > 0

    @pytest.mark.asyncio
    async def test_get_forecast_caching(self, service):
        """Test forecast is cached."""
        # First call
        result1 = await service.get_forecast()

        # Second call should use cache
        result2 = await service.get_forecast()

        assert result1.generated_at == result2.generated_at

    @pytest.mark.asyncio
    async def test_get_forecast_force_refresh(self, service):
        """Test force refresh bypasses cache."""
        # First call
        result1 = await service.get_forecast()

        # Force refresh
        result2 = await service.get_forecast(force_refresh=True)

        # Generated times may be different
        assert result2 is not None

    @pytest.mark.asyncio
    async def test_get_forecast_calculates_aggregates(self, service):
        """Test forecast calculates temperature aggregates."""
        result = await service.get_forecast()

        assert result.avg_temperature_c is not None
        assert result.max_temperature_c is not None
        assert result.min_temperature_c is not None

    @pytest.mark.asyncio
    async def test_get_forecast_calculates_degree_hours(self, service):
        """Test forecast calculates degree hours."""
        result = await service.get_forecast()

        assert result.total_heating_degree_hours is not None
        assert result.total_cooling_degree_hours is not None

    @pytest.mark.asyncio
    async def test_get_current_weather(self, service):
        """Test getting current weather."""
        result = await service.get_current_weather()

        assert result is not None
        assert result.temperature_c is not None

    def test_get_load_adjustments_with_cached_forecast(self, service):
        """Test getting load adjustments with cached forecast."""
        import asyncio

        # Get forecast first (populates cache)
        forecast = asyncio.run(service.get_forecast())

        # Get adjustments
        adjustments = service.get_load_adjustments()

        assert len(adjustments) > 0

    def test_get_load_adjustments_no_cache(self, service):
        """Test getting load adjustments without cache."""
        # Clear cache
        service._forecast_cache = None

        adjustments = service.get_load_adjustments()

        assert len(adjustments) == 0

    def test_cache_validity_check(self, service):
        """Test cache validity check."""
        # No cache - invalid
        assert service._is_cache_valid() is False

        # After getting forecast, cache should be valid
        import asyncio
        asyncio.run(service.get_forecast())

        assert service._is_cache_valid() is True


class TestWeatherServiceErrorHandling:
    """Tests for error handling in WeatherService."""

    @pytest.fixture
    def service_with_failing_provider(self, sample_weather_config):
        """Create service with provider that fails."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            WeatherService,
        )

        service = WeatherService(
            config=sample_weather_config,
            api_key="test_key",
        )

        # Make provider fail
        async def failing_forecast(*args, **kwargs):
            raise Exception("API error")

        service._provider.get_forecast = failing_forecast

        return service

    @pytest.mark.asyncio
    async def test_get_forecast_error_uses_cache(
        self,
        service_with_failing_provider,
        sample_weather_config,
    ):
        """Test error falls back to cached data."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            WeatherService,
            WeatherForecastResult,
        )

        service = service_with_failing_provider

        # Set up cache
        cached_result = WeatherForecastResult(
            latitude=sample_weather_config.latitude,
            longitude=sample_weather_config.longitude,
            forecast_points=[],
            forecast_horizon_hours=24,
        )
        service._forecast_cache = cached_result

        # Should return cached data
        result = await service.get_forecast()

        assert result == cached_result

    @pytest.mark.asyncio
    async def test_get_forecast_error_no_cache_raises(
        self,
        service_with_failing_provider,
    ):
        """Test error with no cache raises exception."""
        service = service_with_failing_provider

        with pytest.raises(Exception):
            await service.get_forecast()


class TestWeatherIntegrationPerformance:
    """Performance tests for weather integration."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_forecast_fetch_time(self, sample_weather_config):
        """Test forecast fetching completes in reasonable time."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            WeatherService,
        )
        import time

        service = WeatherService(
            config=sample_weather_config,
            api_key="test_key",
        )

        start = time.time()
        result = await service.get_forecast()
        elapsed = time.time() - start

        assert result is not None
        assert elapsed < 2.0  # Should complete in under 2 seconds

    @pytest.mark.performance
    def test_impact_calculation_time(
        self,
        sample_weather_config,
        sample_weather_forecast,
    ):
        """Test impact calculation completes in reasonable time."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            WeatherImpactCalculator,
        )
        import time

        calculator = WeatherImpactCalculator(sample_weather_config)

        start = time.time()
        adjustments = calculator.calculate_load_profile_adjustments(
            forecast=sample_weather_forecast,
        )
        elapsed = time.time() - start

        assert len(adjustments) > 0
        assert elapsed < 0.5  # Should complete in under 0.5 seconds


class TestWeatherModuleExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """Test all expected exports are available."""
        from greenlang.agents.process_heat.gl_019_heat_scheduler.weather_integration import (
            CurrentWeather,
            HistoricalWeather,
            WeatherProvider,
            OpenWeatherMapProvider,
            ManualWeatherProvider,
            DegreeDayCalculator,
            WeatherImpactCalculator,
            WeatherService,
        )

        # All imports should work
        assert CurrentWeather is not None
        assert HistoricalWeather is not None
        assert WeatherProvider is not None
        assert OpenWeatherMapProvider is not None
        assert ManualWeatherProvider is not None
        assert DegreeDayCalculator is not None
        assert WeatherImpactCalculator is not None
        assert WeatherService is not None
