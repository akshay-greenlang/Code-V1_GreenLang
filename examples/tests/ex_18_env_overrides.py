"""Example 18: Environment overrides (region/format)."""

import os
import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.config import Config
except Exception:
    Config = None

@pytest.mark.example
def test_environment_overrides(monkeypatch):
    """Environment variables override defaults."""
    if Config is None:
        pytest.skip("Config not importable")
    
    # Set environment overrides
    monkeypatch.setenv("GREENLANG_DEFAULT_COUNTRY", "EU")
    monkeypatch.setenv("GREENLANG_OUTPUT_FORMAT", "json")
    monkeypatch.setenv("GREENLANG_DECIMAL_PLACES", "3")
    
    config = Config()
    
    assert config.default_country == "EU"
    assert config.output_format == "json"
    assert config.decimal_places == 3
    
    # Clear and test defaults
    monkeypatch.delenv("GREENLANG_DEFAULT_COUNTRY", raising=False)
    monkeypatch.delenv("GREENLANG_OUTPUT_FORMAT", raising=False)
    monkeypatch.delenv("GREENLANG_DECIMAL_PLACES", raising=False)
    
    config2 = Config()
    assert config2.default_country in ["US", "IN", "EU"]  # Some default
    assert config2.output_format in ["json", "markdown", "text"]