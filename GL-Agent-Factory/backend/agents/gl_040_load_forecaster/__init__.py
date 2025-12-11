"""GL-040: Load Forecaster Agent"""
from .agent import LoadForecasterAgent, LoadForecasterInput, LoadForecasterOutput, HistoricalLoad, WeatherForecast, HourlyForecast, PACK_SPEC

__all__ = ["LoadForecasterAgent", "LoadForecasterInput", "LoadForecasterOutput", "HistoricalLoad", "WeatherForecast", "HourlyForecast", "PACK_SPEC"]
__version__ = "1.0.0"
__agent_id__ = "GL-040"
