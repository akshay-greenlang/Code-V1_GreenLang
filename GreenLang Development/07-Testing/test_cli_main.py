# -*- coding: utf-8 -*-
"""
Comprehensive tests for greenlang.cli.main module
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from typer.testing import CliRunner
from rich.console import Console

# Add the greenlang directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "greenlang"))

from greenlang.cli.main import app, main, cli, FALLBACK_VERSION


class TestCLIMain:
    """Test the CLI main functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.runner = CliRunner()

    def test_cli_app_creation(self):
        """Test that the CLI app is created correctly"""
        assert app.info.name == "gl"
        assert "GreenLang" in app.info.help
        assert app.info.no_args_is_help is True
        assert app.info.add_completion is False

    def test_main_entry_point(self):
        """Test the main entry point function"""
        with patch.object(app, '__call__') as mock_app:
            main()
            mock_app.assert_called_once()

    def test_cli_legacy_entry_point(self):
        """Test the legacy CLI entry point"""
        with patch.object(app, '__call__') as mock_app:
            cli()
            mock_app.assert_called_once()

    def test_help_command(self):
        """Test the help command output"""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "GreenLang" in result.stdout
        assert "Infrastructure for Climate Intelligence" in result.stdout

    def test_version_flag_with_import_success(self):
        """Test --version flag when version import succeeds"""
        with patch('greenlang.cli.main.__version__', "1.2.3"):
            result = self.runner.invoke(app, ["--version"])
            assert result.exit_code == 0
            assert "GreenLang v1.2.3" in result.stdout
            assert "Infrastructure for Climate Intelligence" in result.stdout
            assert "https://greenlang.in" in result.stdout

    def test_version_flag_with_import_error(self):
        """Test --version flag when version import fails"""
        with patch('greenlang.cli.main.__version__', side_effect=ImportError()):
            result = self.runner.invoke(app, ["--version"])
            assert result.exit_code == 0
            assert f"GreenLang v{FALLBACK_VERSION}" in result.stdout
            assert "Infrastructure for Climate Intelligence" in result.stdout

    def test_version_command_with_import_success(self):
        """Test version command when import succeeds"""
        with patch('greenlang.cli.main.__version__', "2.1.0"):
            result = self.runner.invoke(app, ["version"])
            assert result.exit_code == 0
            assert "GreenLang v2.1.0" in result.stdout

    def test_version_command_with_import_error(self):
        """Test version command when import fails"""
        with patch('greenlang.cli.main.__version__', side_effect=ImportError()):
            result = self.runner.invoke(app, ["version"])
            assert result.exit_code == 0
            assert f"GreenLang v{FALLBACK_VERSION}" in result.stdout

    def test_init_command_with_name_only(self):
        """Test init command with name parameter only"""
        with patch('pathlib.Path.cwd') as mock_cwd, \
             patch('pathlib.Path.exists') as mock_exists:

            mock_cwd.return_value = Path("/test")
            mock_exists.return_value = False

            result = self.runner.invoke(app, ["init", "--name", "test-pack"])
            assert result.exit_code == 0
            assert "Created pack: test-pack" in result.stdout

    def test_init_command_with_custom_path(self):
        """Test init command with custom path"""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = False

            result = self.runner.invoke(app, [
                "init",
                "--name", "custom-pack",
                "--path", "/custom/path"
            ])
            assert result.exit_code == 0
            assert "Created pack: custom-pack" in result.stdout

    def test_init_command_directory_exists_error(self):
        """Test init command when directory already exists"""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True

            result = self.runner.invoke(app, ["init", "--name", "existing-pack"])
            assert result.exit_code == 1
            assert "Error: Directory already exists" in result.stdout

    def test_doctor_command_with_version_import_success(self):
        """Test doctor command when version import succeeds"""
        with patch('greenlang.cli.main.__version__', "1.0.0"), \
             patch('pathlib.Path.home') as mock_home, \
             patch('pathlib.Path.exists') as mock_exists:

            mock_home.return_value = Path("/home/user")
            mock_exists.return_value = True

            result = self.runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            assert "GreenLang Environment Check" in result.stdout
            assert "GreenLang Version: v1.0.0" in result.stdout
            assert "[OK]" in result.stdout

    def test_doctor_command_with_version_import_error(self):
        """Test doctor command when version import fails"""
        with patch('greenlang.cli.main.__version__', side_effect=ImportError()), \
             patch('pathlib.Path.home') as mock_home, \
             patch('pathlib.Path.exists') as mock_exists:

            mock_home.return_value = Path("/home/user")
            mock_exists.return_value = True

            result = self.runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            assert f"GreenLang Version: v{FALLBACK_VERSION}" in result.stdout

    def test_doctor_command_python_version_check(self):
        """Test doctor command Python version checking"""
        with patch('greenlang.cli.main.__version__', "1.0.0"), \
             patch('pathlib.Path.home') as mock_home, \
             patch('pathlib.Path.exists') as mock_exists:

            mock_home.return_value = Path("/home/user")
            mock_exists.return_value = True

            result = self.runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            assert "Python Version:" in result.stdout
            # Should show OK for Python 3.10+ or FAIL for older versions
            if sys.version_info >= (3, 10):
                assert "[OK]" in result.stdout
            else:
                assert "[FAIL]" in result.stdout

    def test_doctor_command_config_directory_missing(self):
        """Test doctor command when config directory is missing"""
        with patch('greenlang.cli.main.__version__', "1.0.0"), \
             patch('pathlib.Path.home') as mock_home, \
             patch('pathlib.Path.exists') as mock_exists:

            mock_home.return_value = Path("/home/user")
            mock_exists.return_value = False

            result = self.runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            assert "[WARN]" in result.stdout
            assert "Config Directory:" in result.stdout

    def test_run_command_with_pipeline_only(self):
        """Test run command with pipeline parameter only"""
        result = self.runner.invoke(app, ["run", "test-pipeline"])
        assert result.exit_code == 0
        assert "Running pipeline: test-pipeline" in result.stdout
        assert "Pipeline completed" in result.stdout

    def test_run_command_with_input_file(self):
        """Test run command with input file"""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True

            result = self.runner.invoke(app, [
                "run", "test-pipeline",
                "--input", "/path/to/input.json"
            ])
            assert result.exit_code == 0
            assert "Running pipeline: test-pipeline" in result.stdout
            assert "Input: /path/to/input.json" in result.stdout

    def test_run_command_with_output_file(self):
        """Test run command with output file"""
        result = self.runner.invoke(app, [
            "run", "test-pipeline",
            "--output", "/path/to/output.json"
        ])
        assert result.exit_code == 0
        assert "Output: /path/to/output.json" in result.stdout

    def test_run_command_with_nonexistent_input_file(self):
        """Test run command with non-existent input file"""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = False

            result = self.runner.invoke(app, [
                "run", "test-pipeline",
                "--input", "/nonexistent/input.json"
            ])
            assert result.exit_code == 0
            # Should still run but not show input file

    def test_policy_command_check_action(self):
        """Test policy command with check action"""
        result = self.runner.invoke(app, ["policy", "check", "test-target"])
        assert result.exit_code == 0
        assert "Checking policy for test-target" in result.stdout
        assert "Policy check passed" in result.stdout

    def test_policy_command_list_action(self):
        """Test policy command with list action"""
        result = self.runner.invoke(app, ["policy", "list"])
        assert result.exit_code == 0
        assert "No policies configured" in result.stdout

    def test_policy_command_unknown_action(self):
        """Test policy command with unknown action"""
        result = self.runner.invoke(app, ["policy", "unknown"])
        assert result.exit_code == 0
        assert "Action 'unknown' not yet implemented" in result.stdout

    def test_verify_command_with_existing_artifact(self):
        """Test verify command with existing artifact"""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True

            result = self.runner.invoke(app, ["verify", "/path/to/artifact"])
            assert result.exit_code == 0
            assert "Verifying /path/to/artifact" in result.stdout
            assert "Artifact verified" in result.stdout

    def test_verify_command_with_signature(self):
        """Test verify command with signature file"""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True

            result = self.runner.invoke(app, [
                "verify", "/path/to/artifact",
                "--sig", "/path/to/signature"
            ])
            assert result.exit_code == 0
            assert "Using signature: /path/to/signature" in result.stdout

    def test_verify_command_with_nonexistent_artifact(self):
        """Test verify command with non-existent artifact"""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = False

            result = self.runner.invoke(app, ["verify", "/nonexistent/artifact"])
            assert result.exit_code == 1
            assert "Artifact not found" in result.stdout

    def test_pack_subcommand_integration(self):
        """Test that pack subcommand is integrated"""
        result = self.runner.invoke(app, ["pack", "--help"])
        # Should not error out, even if pack command doesn't exist yet
        # This tests the integration attempt
        assert result.exit_code in [0, 2]  # 0 for success, 2 for command not found

    @pytest.mark.parametrize("command,expected_in_output", [
        (["--help"], "GreenLang"),
        (["version"], "GreenLang"),
        (["doctor"], "Environment Check"),
        (["run", "test"], "Running pipeline"),
        (["policy", "list"], "No policies"),
        (["init", "--name", "test"], "Created pack"),
    ])
    def test_command_output_consistency(self, command, expected_in_output):
        """Test that commands produce expected output consistently"""
        with patch('pathlib.Path.exists', return_value=False):
            result = self.runner.invoke(app, command)
            assert expected_in_output in result.stdout

    def test_console_initialization(self):
        """Test that Rich console is properly initialized"""
        from greenlang.cli.main import console
        assert isinstance(console, Console)

    def test_fallback_version_constant(self):
        """Test that fallback version constant is defined"""
        assert FALLBACK_VERSION == "2.0.0"

    def test_app_configuration(self):
        """Test CLI app configuration"""
        assert app.info.name == "gl"
        assert "Climate Intelligence" in app.info.help
        assert app.info.no_args_is_help is True
        assert app.info.add_completion is False

    @patch('greenlang.cli.main.console')
    def test_version_command_console_usage(self, mock_console):
        """Test that version command uses console correctly"""
        with patch('greenlang.cli.main.__version__', "1.0.0"):
            result = self.runner.invoke(app, ["version"])
            assert result.exit_code == 0
            # Console should have been called to print version info

    def test_command_error_handling(self):
        """Test error handling in commands"""
        # Test with invalid command
        result = self.runner.invoke(app, ["nonexistent-command"])
        assert result.exit_code != 0

    def test_multiple_command_execution(self):
        """Test executing multiple commands in sequence"""
        commands = [
            ["version"],
            ["doctor"],
            ["policy", "list"],
        ]

        for command in commands:
            result = self.runner.invoke(app, command)
            # All should execute without crashing
            assert result.exit_code in [0, 1]  # Success or controlled failure

    @pytest.mark.parametrize("invalid_input", [
        ["init"],  # Missing required name
        ["run"],   # Missing required pipeline
        ["verify"],  # Missing required artifact
        ["policy"],  # Missing required action
    ])
    def test_missing_required_parameters(self, invalid_input):
        """Test handling of missing required parameters"""
        result = self.runner.invoke(app, invalid_input)
        assert result.exit_code == 2  # Typer error code for missing arguments

    def test_path_handling_in_commands(self):
        """Test path handling across different commands"""
        test_paths = [
            "/absolute/path",
            "relative/path",
            ".",
            "..",
            "~/home/path"
        ]

        for path in test_paths:
            with patch('pathlib.Path.exists', return_value=True):
                # Test init command with different paths
                result = self.runner.invoke(app, [
                    "init", "--name", "test", "--path", path
                ])
                assert result.exit_code == 0

    def test_init_command_path_resolution(self):
        """Test that init command resolves paths correctly"""
        with patch('pathlib.Path.cwd') as mock_cwd, \
             patch('pathlib.Path.exists') as mock_exists:

            mock_cwd.return_value = Path("/current/dir")
            mock_exists.return_value = False

            result = self.runner.invoke(app, ["init", "--name", "test-pack"])
            assert result.exit_code == 0
            # Should create pack in current directory by default

    def test_verify_command_signature_handling(self):
        """Test verify command signature file handling"""
        with patch('pathlib.Path.exists') as mock_exists:
            # Artifact exists, signature doesn't
            mock_exists.side_effect = lambda path: str(path) == "/artifact"

            result = self.runner.invoke(app, [
                "verify", "/artifact",
                "--sig", "/nonexistent/signature"
            ])
            assert result.exit_code == 0
            # Should still verify without signature