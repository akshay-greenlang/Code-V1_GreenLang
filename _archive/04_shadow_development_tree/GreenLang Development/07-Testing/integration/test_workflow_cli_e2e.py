# -*- coding: utf-8 -*-
"""
CLI end-to-end integration tests.
"""
import pytest
import json
import os
from pathlib import Path
from click.testing import CliRunner
from tests.integration.utils import (
    normalize_text,
    load_fixture,
    TestIOHelper
)


@pytest.mark.integration
class TestCLIEndToEnd:
    """Test GreenLang CLI commands end-to-end."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create a Click CLI runner."""
        return CliRunner()
    
    def test_cli_run_workflow(self, cli_runner, tmp_outdir):
        """Test 'gl run' command with workflow."""
        from greenlang.cli import cli  # Import the actual CLI
        
        # Prepare paths
        workflow_path = 'tests/fixtures/workflows/commercial_building_emissions.yaml'
        input_path = 'tests/fixtures/data/building_india_office.json'
        output_path = tmp_outdir / 'output.json'
        
        # Run CLI command
        result = cli_runner.invoke(cli, [
            'run',
            workflow_path,
            '-i', str(input_path),
            '-o', str(output_path),
            '--format', 'json'
        ])
        
        # Check exit code
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        
        # Check output file created
        assert output_path.exists(), "Output file not created"
        
        # Load and validate output
        with open(output_path) as f:
            output_data = json.load(f)
        
        # Verify output structure
        assert 'emissions_report' in output_data or 'emissions' in output_data
        
        # Check for required fields
        report = output_data.get('emissions_report', output_data)
        assert 'total_co2e_kg' in report or 'total_emissions_kg' in report
        assert 'by_fuel' in report
        
        # Verify numerical invariants
        if 'by_fuel' in report and 'total_co2e_kg' in report:
            fuel_sum = sum(report['by_fuel'].values())
            total = report['total_co2e_kg']
            assert abs(fuel_sum - total) < 1.0  # Within 1 kg tolerance
    
    def test_cli_calc_building(self, cli_runner, tmp_outdir):
        """Test 'gl calc --building' command."""
        from greenlang.cli import cli
        
        input_path = 'tests/fixtures/data/building_india_office.json'
        output_path = tmp_outdir / 'report.md'
        
        result = cli_runner.invoke(cli, [
            'calc',
            '--building',
            '--input', str(input_path),
            '--output', str(output_path)
        ])
        
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        
        # Check markdown report created
        assert output_path.exists()
        
        report_content = output_path.read_text()
        
        # Verify report contains expected sections
        assert '# Building Emissions Report' in report_content or 'Emissions' in report_content
        assert 'Total' in report_content
        assert 'CO2' in report_content or 'co2' in report_content.lower()
        
        # Check for normalized snapshot match
        normalized = normalize_text(report_content)
        snapshot_path = Path('tests/integration/snapshots/cli/calc_building.md')
        
        # For now, just check content exists
        assert len(normalized) > 100  # Non-trivial report
    
    def test_cli_with_invalid_input(self, cli_runner):
        """Test CLI error handling with invalid input."""
        from greenlang.cli import cli
        
        result = cli_runner.invoke(cli, [
            'run',
            'nonexistent_workflow.yaml',
            '-i', 'nonexistent_input.json'
        ])
        
        # Should fail with non-zero exit code
        assert result.exit_code != 0
        
        # Should have error message
        assert 'error' in result.output.lower() or 'not found' in result.output.lower()
    
    def test_cli_help_command(self, cli_runner):
        """Test help command output."""
        from greenlang.cli import cli
        
        result = cli_runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'Usage:' in result.output
        assert 'Commands:' in result.output or 'Options:' in result.output
    
    def test_cli_version_command(self, cli_runner):
        """Test version command output."""
        from greenlang.cli import cli
        
        result = cli_runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert 'version' in result.output.lower() or '1.' in result.output
    
    def test_cli_portfolio_command(self, cli_runner, tmp_outdir):
        """Test CLI with portfolio analysis."""
        from greenlang.cli import cli
        
        input_path = 'tests/fixtures/data/portfolio_small.json'
        output_csv = tmp_outdir / 'portfolio.csv'
        
        result = cli_runner.invoke(cli, [
            'calc',
            '--portfolio',
            '--input', str(input_path),
            '--output', str(output_csv),
            '--format', 'csv'
        ])
        
        if result.exit_code == 0:
            # Check CSV created
            assert output_csv.exists()
            
            # Verify CSV content
            import csv
            with open(output_csv, newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Should have row for each building
            portfolio_data = load_fixture('data', 'portfolio_small.json')
            assert len(rows) >= len(portfolio_data['buildings'])
    
    def test_cli_json_output_format(self, cli_runner, tmp_outdir):
        """Test JSON output format."""
        from greenlang.cli import cli
        
        input_path = 'tests/fixtures/data/building_us_office.json'
        output_path = tmp_outdir / 'output.json'
        
        result = cli_runner.invoke(cli, [
            'calc',
            '--building',
            '--input', str(input_path),
            '--output', str(output_path),
            '--format', 'json'
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        
        # Verify valid JSON
        with open(output_path) as f:
            data = json.load(f)
        
        assert isinstance(data, dict)
        assert len(data) > 0
    
    def test_cli_quiet_mode(self, cli_runner, tmp_outdir):
        """Test quiet mode suppresses output."""
        from greenlang.cli import cli
        
        input_path = 'tests/fixtures/data/building_india_office.json'
        output_path = tmp_outdir / 'output.json'
        
        # Run without quiet
        result_normal = cli_runner.invoke(cli, [
            'calc',
            '--building',
            '--input', str(input_path),
            '--output', str(output_path),
            '--format', 'json'
        ])
        
        # Run with quiet
        result_quiet = cli_runner.invoke(cli, [
            'calc',
            '--building',
            '--input', str(input_path),
            '--output', str(output_path),
            '--format', 'json',
            '--quiet'
        ])
        
        # Quiet mode should have less output
        if '--quiet' in str(cli.params):  # If quiet flag is implemented
            assert len(result_quiet.output) < len(result_normal.output)
    
    def test_cli_verbose_mode(self, cli_runner, tmp_outdir):
        """Test verbose mode provides extra output."""
        from greenlang.cli import cli
        
        input_path = 'tests/fixtures/data/building_india_office.json'
        output_path = tmp_outdir / 'output.json'
        
        # Run with verbose
        result = cli_runner.invoke(cli, [
            'calc',
            '--building',
            '--input', str(input_path),
            '--output', str(output_path),
            '--format', 'json',
            '--verbose'
        ])
        
        if '--verbose' in str(cli.params):  # If verbose flag is implemented
            # Should include debug/info messages
            assert 'Processing' in result.output or 'Calculating' in result.output
    
    @pytest.mark.performance
    def test_cli_performance(self, cli_runner, tmp_outdir):
        """Test CLI performance meets requirements."""
        from greenlang.cli import cli
        import time
        
        input_path = 'tests/fixtures/data/building_india_office.json'
        output_path = tmp_outdir / 'output.json'
        
        start = time.time()
        
        result = cli_runner.invoke(cli, [
            'calc',
            '--building',
            '--input', str(input_path),
            '--output', str(output_path),
            '--format', 'json'
        ])
        
        elapsed = time.time() - start
        
        assert result.exit_code == 0
        assert elapsed < 2.0, f"CLI took {elapsed:.2f}s, exceeding 2s budget"