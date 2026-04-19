"""GL-040 Load Forecaster - Golden Tests"""
import pytest
from .agent import LoadForecasterAgent, LoadForecasterInput, WeatherForecast

class TestLoadForecaster:
    def test_basic_run(self):
        agent = LoadForecasterAgent()
        result = agent.run(LoadForecasterInput(
            system_id="HVAC-001",
            base_load_kw=500,
            weather_forecast=[WeatherForecast(hour=h, temperature_celsius=25) for h in range(24)]
        ))
        assert result.validation_status == "PASS"
        assert len(result.load_forecast_24h) == 24

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
