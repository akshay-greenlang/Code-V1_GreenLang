"""Example 12: CLI run to JSON export."""

import json
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.cli.main import cli
except Exception:
    cli = None

from click.testing import CliRunner

@pytest.mark.example
def test_cli_run_to_json_export(tmp_path):
    """CLI run → JSON export."""
    if cli is None:
        pytest.skip("CLI not importable")
    
    output_file = tmp_path / "emissions.json"
    
    runner = CliRunner()
    result = runner.invoke(cli, [
        "calc",
        "--building",
        "--input", "examples/fixtures/building_india_office.json",
        "--output", str(output_file),
        "--format", "json"
    ])
    
    assert result.exit_code == 0, result.output
    assert output_file.exists()
    
    with open(output_file, "r") as f:
        data = json.load(f)
    
    assert "total_emissions_kg" in data or "emissions" in data
    assert data.get("total_emissions_kg", 0) > 0 or data.get("emissions", {}).get("total_co2e_kg", 0) > 0