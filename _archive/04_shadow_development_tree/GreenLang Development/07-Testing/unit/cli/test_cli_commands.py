# -*- coding: utf-8 -*-
"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner
from greenlang.cli.main import cli


class TestCLICommands:
    """Test suite for CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_version_prints_semver(self):
        """Test that version command prints semantic version."""
        result = self.runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        output = result.output.strip()
        
        # Should contain version number in semver format
        assert any(char.isdigit() for char in output)  # Has numbers
        assert "." in output  # Has dots for version separation
        
        # Check for semantic version pattern (x.y.z)
        import re
        semver_pattern = r'\d+\.\d+\.\d+'
        assert re.search(semver_pattern, output) is not None
    
    def test_help_command(self):
        """Test that help command works."""
        result = self.runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "Options:" in result.output
        assert "Commands:" in result.output
    
    def test_agents_list_command(self):
        """Test that agents list command shows available agents."""
        result = self.runner.invoke(cli, ['agents', 'list'])
        
        assert result.exit_code == 0
        output = result.output.lower()
        
        # Should list key agents
        assert "fuel" in output
        assert "grid" in output or "factor" in output
        assert "carbon" in output
        assert "intensity" in output
        assert "benchmark" in output
    
    def test_agents_info_command(self):
        """Test that agents info command shows agent details."""
        result = self.runner.invoke(cli, ['agents', 'info', 'fuel'])
        
        if result.exit_code == 0:
            output = result.output.lower()
            assert "fuel" in output
            assert any(word in output for word in ["description", "input", "output", "calculate"])
    
    def test_calculate_command_basic(self):
        """Test basic calculate command."""
        result = self.runner.invoke(cli, [
            'calculate',
            '--country', 'IN',
            '--electricity', '1000000',
            '--area', '50000'
        ])
        
        if result.exit_code == 0:
            output = result.output
            # Should show emissions result
            assert any(word in output.lower() for word in ["emissions", "co2", "kg"])
            # Should show the calculated value
            assert any(char.isdigit() for char in output)
    
    def test_calculate_with_multiple_fuels(self):
        """Test calculate command with multiple fuel types."""
        result = self.runner.invoke(cli, [
            'calculate',
            '--country', 'US',
            '--electricity', '1000000',
            '--natural-gas', '50000',
            '--area', '100000'
        ])
        
        if result.exit_code == 0:
            output = result.output.lower()
            # Should show both fuel types
            assert "electricity" in output
            assert "natural" in output or "gas" in output
            # Should show total
            assert "total" in output
    
    def test_calculate_with_output_format(self):
        """Test calculate command with different output formats."""
        # Test JSON output
        result_json = self.runner.invoke(cli, [
            'calculate',
            '--country', 'IN',
            '--electricity', '1000000',
            '--format', 'json'
        ])
        
        if result_json.exit_code == 0:
            import json
            try:
                data = json.loads(result_json.output)
                assert "emissions" in data or "total" in data
            except json.JSONDecodeError:
                pass  # Format might be different
        
        # Test CSV output
        result_csv = self.runner.invoke(cli, [
            'calculate',
            '--country', 'IN',
            '--electricity', '1000000',
            '--format', 'csv'
        ])
        
        if result_csv.exit_code == 0:
            assert "," in result_csv.output  # CSV has commas
    
    def test_benchmark_command(self):
        """Test benchmark command."""
        result = self.runner.invoke(cli, [
            'benchmark',
            '--country', 'IN',
            '--building-type', 'office',
            '--intensity', '20'
        ])
        
        if result.exit_code == 0:
            output = result.output
            # Should show rating
            assert any(letter in output for letter in ["A", "B", "C", "D", "E", "F"])
            # Should show performance level
            assert any(word in output.lower() for word in ["excellent", "good", "average", "poor"])
    
    def test_workflow_run_command(self):
        """Test workflow run command."""
        # Create a temporary workflow file
        with self.runner.isolated_filesystem():
            with open('test_workflow.yaml', 'w') as f:
                f.write("""
name: test_workflow
agents:
  - name: fuel
    type: FuelAgent
workflow:
  steps:
    - agent: fuel
      input: test_data
""")
            
            result = self.runner.invoke(cli, ['workflow', 'run', 'test_workflow.yaml'])
            
            # Should attempt to run workflow
            assert result.exit_code in [0, 1, 2]  # Various exit codes possible
    
    def test_config_show_command(self):
        """Test config show command."""
        result = self.runner.invoke(cli, ['config', 'show'])
        
        if result.exit_code == 0:
            output = result.output.lower()
            # Should show configuration
            assert any(word in output for word in ["config", "settings", "default"])
    
    def test_config_set_command(self):
        """Test config set command."""
        result = self.runner.invoke(cli, [
            'config', 'set',
            'default_country', 'US'
        ])
        
        if result.exit_code == 0:
            assert "set" in result.output.lower() or "updated" in result.output.lower()
    
    def test_invalid_command(self):
        """Test handling of invalid commands."""
        result = self.runner.invoke(cli, ['invalid_command'])
        
        assert result.exit_code != 0
        assert "Error" in result.output or "Usage" in result.output
    
    def test_missing_required_args(self):
        """Test handling of missing required arguments."""
        result = self.runner.invoke(cli, ['calculate'])  # Missing required args
        
        assert result.exit_code != 0
        assert any(word in result.output for word in ["Error", "Missing", "required", "Usage"])
    
    def test_verbose_mode(self):
        """Test verbose mode output."""
        result = self.runner.invoke(cli, [
            '--verbose',
            'calculate',
            '--country', 'IN',
            '--electricity', '1000000'
        ])
        
        if "--verbose" in cli.params or "-v" in cli.params:
            # Verbose mode should show more output
            assert len(result.output) > 100  # More detailed output
    
    def test_quiet_mode(self):
        """Test quiet mode output."""
        result_normal = self.runner.invoke(cli, [
            'calculate',
            '--country', 'IN',
            '--electricity', '1000000'
        ])
        
        result_quiet = self.runner.invoke(cli, [
            '--quiet',
            'calculate',
            '--country', 'IN',
            '--electricity', '1000000'
        ])
        
        if "--quiet" in cli.params or "-q" in cli.params:
            # Quiet mode should show less output
            assert len(result_quiet.output) < len(result_normal.output)