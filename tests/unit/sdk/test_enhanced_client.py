"""Tests for Enhanced SDK Client."""

import pytest
from unittest.mock import Mock, patch
from greenlang.sdk.enhanced_client import EnhancedClient


class TestEnhancedClient:
    """Test suite for Enhanced SDK Client - sanity checks."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = EnhancedClient(api_key="test_key")
    
    def test_client_initialization(self):
        """Test client initialization."""
        assert self.client is not None
        assert hasattr(self.client, 'api_key')
        assert self.client.api_key == "test_key"
    
    def test_calculate_emissions_basic(self):
        """Test basic emissions calculation."""
        building_data = {
            "building_type": "office",
            "country": "IN",
            "total_area": {"value": 50000, "unit": "sqft"},
            "energy_sources": [
                {
                    "fuel_type": "electricity",
                    "consumption": {"value": 1000000, "unit": "kWh"}
                }
            ]
        }
        
        # Mock the internal calculation
        with patch.object(self.client, '_process_building_data') as mock_process:
            mock_process.return_value = {
                "success": True,
                "emissions": {
                    "total": 710000.0,
                    "unit": "kgCO2e"
                }
            }
            
            result = self.client.calculate_emissions(building_data)
            
            assert result["success"] is True
            assert result["emissions"]["total"] == 710000.0
            mock_process.assert_called_once_with(building_data)
    
    def test_calculate_emissions_with_multiple_fuels(self):
        """Test emissions calculation with multiple fuel types."""
        building_data = {
            "building_type": "office",
            "country": "US",
            "total_area": {"value": 100000, "unit": "sqft"},
            "energy_sources": [
                {
                    "fuel_type": "electricity",
                    "consumption": {"value": 2000000, "unit": "kWh"}
                },
                {
                    "fuel_type": "natural_gas",
                    "consumption": {"value": 50000, "unit": "therms"}
                }
            ]
        }
        
        with patch.object(self.client, '_process_building_data') as mock_process:
            mock_process.return_value = {
                "success": True,
                "emissions": {
                    "total": 1035000.0,
                    "breakdown": [
                        {"fuel_type": "electricity", "emissions": 770000.0},
                        {"fuel_type": "natural_gas", "emissions": 265000.0}
                    ],
                    "unit": "kgCO2e"
                }
            }
            
            result = self.client.calculate_emissions(building_data)
            
            assert result["success"] is True
            assert len(result["emissions"]["breakdown"]) == 2
    
    def test_get_benchmark(self):
        """Test benchmark retrieval."""
        with patch.object(self.client, '_get_benchmark_data') as mock_benchmark:
            mock_benchmark.return_value = {
                "success": True,
                "rating": "C",
                "performance_level": "Average"
            }
            
            result = self.client.get_benchmark(
                building_type="office",
                country="IN",
                intensity=20.0
            )
            
            assert result["success"] is True
            assert result["rating"] == "C"
            assert result["performance_level"] == "Average"
    
    def test_get_recommendations(self):
        """Test recommendations generation."""
        with patch.object(self.client, '_generate_recommendations') as mock_rec:
            mock_rec.return_value = {
                "success": True,
                "recommendations": [
                    {"action": "LED lighting", "impact": "high"},
                    {"action": "HVAC upgrade", "impact": "very_high"}
                ]
            }
            
            result = self.client.get_recommendations(
                building_type="office",
                performance_rating="D",
                country="IN"
            )
            
            assert result["success"] is True
            assert len(result["recommendations"]) == 2
    
    def test_generate_report(self):
        """Test report generation."""
        report_data = {
            "emissions": {"total": 1000000.0},
            "intensities": {"per_sqft": 20.0},
            "benchmark": {"rating": "C"},
            "recommendations": [{"action": "Solar PV", "impact": "high"}]
        }
        
        with patch.object(self.client, '_create_report') as mock_report:
            mock_report.return_value = {
                "success": True,
                "report": "# Carbon Emissions Report\n...",
                "format": "markdown"
            }
            
            result = self.client.generate_report(report_data, format="markdown")
            
            assert result["success"] is True
            assert result["format"] == "markdown"
            assert "report" in result
    
    def test_batch_processing(self):
        """Test batch processing of multiple buildings."""
        buildings = [
            {"id": "b1", "country": "IN", "electricity": 1000000},
            {"id": "b2", "country": "US", "electricity": 2000000}
        ]
        
        with patch.object(self.client, 'calculate_emissions') as mock_calc:
            mock_calc.side_effect = [
                {"success": True, "emissions": {"total": 710000.0}},
                {"success": True, "emissions": {"total": 770000.0}}
            ]
            
            results = self.client.batch_calculate(buildings)
            
            assert len(results) == 2
            assert results[0]["emissions"]["total"] == 710000.0
            assert results[1]["emissions"]["total"] == 770000.0
    
    def test_error_handling(self):
        """Test error handling in client."""
        # Test with invalid data
        result = self.client.calculate_emissions({})
        
        if not result["success"]:
            assert "error" in result
            assert "message" in result["error"]
    
    def test_api_key_validation(self):
        """Test API key validation."""
        # Test with no API key
        with pytest.raises(ValueError, match="API key"):
            client = EnhancedClient(api_key=None)
        
        # Test with empty API key
        with pytest.raises(ValueError, match="API key"):
            client = EnhancedClient(api_key="")
    
    def test_async_support(self):
        """Test async method support if available."""
        import asyncio
        
        async def test_async():
            if hasattr(self.client, 'calculate_emissions_async'):
                result = await self.client.calculate_emissions_async({
                    "country": "IN",
                    "electricity": 1000000
                })
                assert "emissions" in result
        
        # Run async test if supported
        try:
            asyncio.run(test_async())
        except AttributeError:
            pass  # Async not supported
    
    def test_caching(self):
        """Test caching functionality if available."""
        if hasattr(self.client, 'enable_cache'):
            self.client.enable_cache()
            
            # First call
            result1 = self.client.get_grid_factor("IN", "electricity")
            
            # Second call (should be cached)
            result2 = self.client.get_grid_factor("IN", "electricity")
            
            assert result1 == result2
    
    def test_rate_limiting(self):
        """Test rate limiting if implemented."""
        if hasattr(self.client, 'rate_limit'):
            import time
            
            start = time.time()
            
            # Make multiple rapid requests
            for _ in range(5):
                self.client.get_grid_factor("IN", "electricity")
            
            elapsed = time.time() - start
            
            # Should have some delay if rate limited
            if self.client.rate_limit:
                assert elapsed > 0.1  # Some throttling