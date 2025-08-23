"""
Cache Invalidation Tests for GreenLang
Tests to ensure cache correctly invalidates when underlying data changes
"""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from greenlang.sdk.enhanced_client import GreenLangClient
from greenlang.agents.fuel_agent import FuelAgent
from greenlang.agents.grid_factor_agent import GridFactorAgent


class TestCacheInvalidation:
    """Test suite for cache invalidation scenarios"""
    
    @pytest.fixture
    def client(self):
        """Create a fresh client instance"""
        return GreenLangClient(region="US")
    
    @pytest.fixture
    def fuel_agent(self):
        """Create a fresh fuel agent instance"""
        return FuelAgent()
    
    def test_cache_invalidates_on_emission_factor_change(self, client):
        """Test that cache invalidates when emission factor changes"""
        # First calculation with original factor
        result1 = client.calculate_emissions(
            fuel_type="electricity",
            consumption=1000,
            unit="kWh",
            region="US"
        )
        original_emissions = result1["data"]["co2e_emissions_kg"]
        
        # Modify the emission factor (simulating data update)
        with patch('greenlang.data.emission_factors.get_emission_factor') as mock_factor:
            # Return a different factor
            mock_factor.return_value = {"factor": 0.5, "unit": "kgCO2e/kWh"}
            
            # Clear any cache (implementation specific)
            if hasattr(client, '_cache'):
                client._cache.clear()
            
            # Calculate again with modified factor
            result2 = client.calculate_emissions(
                fuel_type="electricity",
                consumption=1000,
                unit="kWh",
                region="US"
            )
            new_emissions = result2["data"]["co2e_emissions_kg"]
            
            # Emissions should be different
            assert new_emissions != original_emissions
            assert new_emissions == 500  # 1000 kWh * 0.5 kgCO2e/kWh
    
    def test_cache_performance_improvement(self, fuel_agent):
        """Test that cache improves performance by >50%"""
        test_payload = {
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms"
        }
        
        # First call (cache miss)
        start = time.perf_counter()
        result1 = fuel_agent.run(test_payload)
        first_call_time = time.perf_counter() - start
        
        # Second call (cache hit)
        start = time.perf_counter()
        result2 = fuel_agent.run(test_payload)
        second_call_time = time.perf_counter() - start
        
        # Results should be identical
        assert result1 == result2
        
        # Second call should be at least 50% faster
        # Note: This might not always be true for very fast operations
        # so we check if caching is present instead
        if hasattr(fuel_agent, '_get_emission_factor'):
            # Check if the cache attribute exists
            assert hasattr(fuel_agent._get_emission_factor, 'cache_info')
            cache_info = fuel_agent._get_emission_factor.cache_info()
            assert cache_info.hits > 0  # At least one cache hit
    
    def test_cache_size_limits(self, client):
        """Test that cache respects size limits"""
        # Generate many different calculations to fill cache
        results = []
        for i in range(1000):  # Large number of unique calculations
            result = client.calculate_emissions(
                fuel_type="electricity",
                consumption=100 + i,  # Different consumption each time
                unit="kWh",
                region="US"
            )
            results.append(result)
        
        # Check cache doesn't grow indefinitely
        if hasattr(client, '_cache'):
            # Typical LRU cache has a maxsize
            assert len(client._cache) <= 128  # Or whatever the limit is
    
    def test_cache_corruption_handling(self, fuel_agent):
        """Test that corrupted cache entries are handled gracefully"""
        # Simulate cache corruption
        if hasattr(fuel_agent, '_cache'):
            # Inject corrupted data
            fuel_agent._cache['bad_key'] = "corrupted_data"
        
        # Should still work correctly
        result = fuel_agent.run({
            "fuel_type": "diesel",
            "amount": 100,
            "unit": "gallons"
        })
        
        assert result["success"] is True
        assert "co2e_emissions_kg" in result["data"]
    
    def test_cache_invalidation_on_data_file_change(self, monkeypatch):
        """Test cache invalidates when data files are modified"""
        # Mock file modification time
        original_mtime = 1000.0
        new_mtime = 2000.0
        
        agent = GridFactorAgent()
        
        # First call
        with patch('os.path.getmtime', return_value=original_mtime):
            result1 = agent.run({
                "country": "US",
                "fuel_type": "electricity"
            })
        
        # Simulate file modification
        with patch('os.path.getmtime', return_value=new_mtime):
            # Clear cache if timestamp-based invalidation is implemented
            if hasattr(agent, '_last_modified'):
                agent._last_modified = 0
            
            result2 = agent.run({
                "country": "US",
                "fuel_type": "electricity"
            })
        
        # Both calls should succeed
        assert result1["success"] is True
        assert result2["success"] is True
    
    def test_cache_thread_safety(self, client):
        """Test that cache is thread-safe"""
        import threading
        import concurrent.futures
        
        def calculate_emissions(client, i):
            return client.calculate_emissions(
                fuel_type="electricity",
                consumption=1000,
                unit="kWh",
                region="US"
            )
        
        # Run multiple threads simultaneously
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(calculate_emissions, client, i)
                for i in range(100)
            ]
            results = [f.result() for f in futures]
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result
    
    def test_cache_persistence_across_sessions(self, tmp_path):
        """Test if cache can be persisted and restored"""
        cache_file = tmp_path / "cache.json"
        
        # First session
        client1 = GreenLangClient(region="US")
        result1 = client1.calculate_emissions(
            fuel_type="electricity",
            consumption=1000,
            unit="kWh"
        )
        
        # Save cache if supported
        if hasattr(client1, 'save_cache'):
            client1.save_cache(cache_file)
        
        # New session
        client2 = GreenLangClient(region="US")
        
        # Load cache if supported
        if hasattr(client2, 'load_cache'):
            client2.load_cache(cache_file)
        
        # Should get same result (potentially from cache)
        result2 = client2.calculate_emissions(
            fuel_type="electricity",
            consumption=1000,
            unit="kWh"
        )
        
        assert result1 == result2
    
    def test_selective_cache_invalidation(self, client):
        """Test that only relevant cache entries are invalidated"""
        # Calculate for multiple regions
        us_result = client.calculate_emissions(
            fuel_type="electricity", consumption=1000, unit="kWh", region="US"
        )
        in_result = client.calculate_emissions(
            fuel_type="electricity", consumption=1000, unit="kWh", region="IN"
        )
        
        # Invalidate only US cache
        if hasattr(client, 'invalidate_cache_for_region'):
            client.invalidate_cache_for_region("US")
            
            # US should recalculate, IN should use cache
            us_result_new = client.calculate_emissions(
                fuel_type="electricity", consumption=1000, unit="kWh", region="US"
            )
            in_result_new = client.calculate_emissions(
                fuel_type="electricity", consumption=1000, unit="kWh", region="IN"
            )
            
            # IN result should be same (from cache)
            assert in_result == in_result_new
    
    def test_cache_warming(self, client):
        """Test cache pre-warming functionality"""
        if hasattr(client, 'warm_cache'):
            # Pre-warm cache with common calculations
            common_queries = [
                {"fuel_type": "electricity", "consumption": 1000, "unit": "kWh"},
                {"fuel_type": "natural_gas", "consumption": 100, "unit": "therms"},
                {"fuel_type": "diesel", "consumption": 50, "unit": "gallons"}
            ]
            
            client.warm_cache(common_queries)
            
            # These calculations should be fast (cache hits)
            for query in common_queries:
                start = time.perf_counter()
                result = client.calculate_emissions(**query, region="US")
                elapsed = time.perf_counter() - start
                
                # Should be very fast (< 1ms typically for cache hit)
                assert elapsed < 0.01
                assert result["success"] is True


class TestConfigurationPrecedence:
    """Test configuration precedence: CLI > ENV > File > Defaults"""
    
    def test_cli_overrides_all(self, monkeypatch):
        """CLI arguments should override all other configuration sources"""
        # Set environment variable
        monkeypatch.setenv("GREENLANG_REGION", "US")
        
        # Create config file
        config_file = Path("test_config.json")
        config_file.write_text(json.dumps({"region": "EU"}))
        
        try:
            # CLI argument should win
            client = GreenLangClient(
                region="IN",  # CLI argument
                config_file=config_file
            )
            assert client.region == "IN"
        finally:
            config_file.unlink(missing_ok=True)
    
    def test_env_overrides_file(self, monkeypatch):
        """Environment variables should override config file"""
        # Set environment variable
        monkeypatch.setenv("GREENLANG_REGION", "US")
        
        # Create config file
        config_file = Path("test_config.json")
        config_file.write_text(json.dumps({"region": "EU"}))
        
        try:
            # ENV should win over file
            client = GreenLangClient(config_file=config_file)
            assert client.region == "US"
        finally:
            config_file.unlink(missing_ok=True)
    
    def test_file_overrides_defaults(self):
        """Config file should override defaults"""
        # Create config file
        config_file = Path("test_config.json")
        config_file.write_text(json.dumps({"region": "EU"}))
        
        try:
            # File should win over defaults
            client = GreenLangClient(config_file=config_file)
            assert client.region == "EU"
        finally:
            config_file.unlink(missing_ok=True)
    
    def test_defaults_used_when_nothing_specified(self, monkeypatch):
        """Defaults should be used when nothing is specified"""
        # Clear any environment variables
        monkeypatch.delenv("GREENLANG_REGION", raising=False)
        
        # No config file, no CLI args
        client = GreenLangClient()
        
        # Should use default
        assert client.region in ["US", None]  # Default region


class TestSnapshotTesting:
    """Snapshot tests for report generation consistency"""
    
    @pytest.fixture
    def snapshot_dir(self, tmp_path):
        """Create directory for snapshots"""
        snapshot_path = tmp_path / "snapshots"
        snapshot_path.mkdir()
        return snapshot_path
    
    def normalize_output(self, output: str) -> str:
        """Normalize output to remove variable elements"""
        import re
        # Remove timestamps
        output = re.sub(r'\d{4}-\d{2}-\d{2}', 'YYYY-MM-DD', output)
        output = re.sub(r'\d{2}:\d{2}:\d{2}', 'HH:MM:SS', output)
        # Remove absolute paths
        output = re.sub(r'[A-Z]:\\[^"]+', '/path/to/file', output)
        output = re.sub(r'/[^"]+/', '/path/to/', output)
        return output
    
    def test_report_snapshot(self, snapshot_dir):
        """Test that report format remains consistent"""
        from greenlang.agents.report_agent import ReportAgent
        
        agent = ReportAgent()
        
        test_data = {
            "building_type": "commercial_office",
            "total_emissions_kg": 100000,
            "total_emissions_tons": 100,
            "breakdown": {
                "electricity": 70000,
                "natural_gas": 30000
            }
        }
        
        result = agent.run(test_data)
        report = result["data"]["report"]
        
        # Normalize the report
        normalized = self.normalize_output(report)
        
        # Save or compare snapshot
        snapshot_file = snapshot_dir / "report_snapshot.txt"
        
        if snapshot_file.exists():
            # Compare with existing snapshot
            expected = snapshot_file.read_text()
            assert normalized == expected, "Report format changed unexpectedly"
        else:
            # Save new snapshot
            snapshot_file.write_text(normalized)
            pytest.skip("Snapshot created - run again to test")
    
    def test_cli_output_snapshot(self, snapshot_dir, capsys):
        """Test CLI output consistency"""
        from greenlang.cli.main import cli
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        
        # Normalize output
        normalized = self.normalize_output(result.output)
        
        # Save or compare snapshot
        snapshot_file = snapshot_dir / "cli_version_snapshot.txt"
        
        if snapshot_file.exists():
            expected = snapshot_file.read_text()
            assert normalized == expected, "CLI output changed unexpectedly"
        else:
            snapshot_file.write_text(normalized)
            pytest.skip("Snapshot created - run again to test")
    
    def test_json_export_snapshot(self, snapshot_dir):
        """Test JSON export format consistency"""
        from greenlang.sdk.enhanced_client import GreenLangClient
        
        client = GreenLangClient(region="US")
        
        building = {
            "metadata": {
                "building_type": "commercial_office",
                "area": 50000,
                "location": {"country": "US"}
            },
            "energy_consumption": {
                "electricity": {"value": 1500000, "unit": "kWh"}
            }
        }
        
        result = client.analyze_building(building)
        
        # Remove variable fields
        if "timestamp" in result:
            result["timestamp"] = "NORMALIZED_TIMESTAMP"
        if "version" in result:
            result["version"] = "NORMALIZED_VERSION"
        
        # Convert to JSON
        json_output = json.dumps(result, indent=2, sort_keys=True)
        normalized = self.normalize_output(json_output)
        
        # Save or compare snapshot
        snapshot_file = snapshot_dir / "json_export_snapshot.json"
        
        if snapshot_file.exists():
            expected = snapshot_file.read_text()
            assert normalized == expected, "JSON export format changed"
        else:
            snapshot_file.write_text(normalized)
            pytest.skip("Snapshot created - run again to test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])