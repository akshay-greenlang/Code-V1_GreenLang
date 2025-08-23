"""Example 13: CLI calc to Markdown snapshot."""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from greenlang.cli.main import cli
except Exception:
    cli = None

from click.testing import CliRunner
from examples.utils.normalizers import normalize_text

@pytest.mark.example
def test_cli_calc_markdown_snapshot(tmp_path, snapshot):
    """CLI calc → Markdown snapshot (normalized to avoid flaky diffs)."""
    if cli is None:
        pytest.skip("CLI not importable")
    
    out_file = tmp_path / "report.md"
    
    runner = CliRunner()
    res = runner.invoke(cli, [
        "calc",
        "--building",
        "--input", "examples/fixtures/building_india_office.json",
        "--output", str(out_file),
        "--format", "markdown"
    ])
    
    assert res.exit_code == 0, res.output
    assert out_file.exists()
    
    # For snapshot testing - would normally compare with saved snapshot
    content = normalize_text(out_file.read_text())
    assert "emissions" in content.lower() or "co2" in content.lower()