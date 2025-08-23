"""Example 15: Plugin discovery contract (illustrative)."""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.core.plugin_manager import PluginManager
except Exception:
    PluginManager = None

@pytest.mark.example
def test_plugin_discovery_contract():
    """Test plugin discovery and registration."""
    if PluginManager is None:
        pytest.skip("PluginManager not importable")
    
    manager = PluginManager()
    
    # Discover built-in plugins
    plugins = manager.discover_plugins()
    
    # Should find at least basic agents
    assert len(plugins) > 0
    
    # Check plugin contract
    for plugin in plugins:
        assert hasattr(plugin, "name")
        assert hasattr(plugin, "version")
        assert hasattr(plugin, "run") or hasattr(plugin, "execute")