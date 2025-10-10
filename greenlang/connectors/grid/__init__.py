"""
Grid Connectors - Carbon Intensity Data Sources
================================================

Connectors for accessing grid carbon intensity data from various sources.

Available Connectors:
- GridIntensityMockConnector: Deterministic mock for testing/development

Future (W3+):
- ElectricityMapsConnector: Real-time data from Electricity Maps API
- WattTimeConnector: Marginal emissions from WattTime API
- NRELConnector: Historical data from NREL API
"""

from greenlang.connectors.grid.mock import GridIntensityMockConnector

__all__ = ["GridIntensityMockConnector"]
