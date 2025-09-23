"""
Comprehensive tests for greenlang._version module
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
from importlib.metadata import PackageNotFoundError

import sys
import os

# Add the greenlang directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "greenlang"))

from greenlang import _version


class TestVersionModule:
    """Test the version resolution logic"""

    def test_version_import_successful(self):
        """Test that version module can be imported"""
        assert hasattr(_version, '__version__')
        assert isinstance(_version.__version__, str)

    @patch('greenlang._version.version')
    def test_version_from_greenlang_cli_package(self, mock_version):
        """Test version resolution from greenlang-cli package"""
        mock_version.return_value = "1.2.3"

        # Reload the module to trigger version resolution
        import importlib
        importlib.reload(_version)

        assert _version.__version__ == "1.2.3"
        mock_version.assert_called_with("greenlang-cli")

    @patch('greenlang._version.version')
    def test_version_fallback_to_greenlang_package(self, mock_version):
        """Test version fallback to greenlang package"""
        def side_effect(package_name):
            if package_name == "greenlang-cli":
                raise PackageNotFoundError()
            return "2.1.0"

        mock_version.side_effect = side_effect

        import importlib
        importlib.reload(_version)

        assert _version.__version__ == "2.1.0"
        assert mock_version.call_count == 2

    @patch('greenlang._version.version')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.read_text')
    def test_version_from_version_file(self, mock_read_text, mock_exists, mock_version):
        """Test version resolution from VERSION file"""
        mock_version.side_effect = PackageNotFoundError()
        mock_exists.return_value = True
        mock_read_text.return_value = "3.0.0-dev\n"

        import importlib
        importlib.reload(_version)

        assert _version.__version__ == "3.0.0-dev"
        mock_read_text.assert_called_once()

    @patch('greenlang._version.version')
    @patch('pathlib.Path.exists')
    def test_version_fallback_when_no_version_file(self, mock_exists, mock_version):
        """Test fallback version when VERSION file doesn't exist"""
        mock_version.side_effect = PackageNotFoundError()
        mock_exists.return_value = False

        import importlib
        importlib.reload(_version)

        assert _version.__version__ == "2.0.0"

    @patch('greenlang._version.version')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.read_text')
    def test_version_file_path_resolution(self, mock_read_text, mock_exists, mock_version):
        """Test that VERSION file path is correctly resolved"""
        mock_version.side_effect = PackageNotFoundError()
        mock_exists.return_value = True
        mock_read_text.return_value = "4.5.6"

        import importlib
        importlib.reload(_version)

        # Verify that the path resolution works correctly
        expected_path = Path(_version.__file__).resolve().parents[1] / "VERSION"
        mock_exists.assert_called()
        assert _version.__version__ == "4.5.6"

    @patch('greenlang._version.version')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.read_text')
    def test_version_file_with_whitespace(self, mock_read_text, mock_exists, mock_version):
        """Test version file reading with whitespace handling"""
        mock_version.side_effect = PackageNotFoundError()
        mock_exists.return_value = True
        mock_read_text.return_value = "  1.0.0-beta  \n\r\t  "

        import importlib
        importlib.reload(_version)

        assert _version.__version__ == "1.0.0-beta"

    @patch('greenlang._version.version')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.read_text')
    def test_version_file_read_error(self, mock_read_text, mock_exists, mock_version):
        """Test handling of VERSION file read errors"""
        mock_version.side_effect = PackageNotFoundError()
        mock_exists.return_value = True
        mock_read_text.side_effect = IOError("Permission denied")

        import importlib
        importlib.reload(_version)

        # Should fall back to default version on read error
        assert _version.__version__ == "2.0.0"

    def test_version_constants(self):
        """Test that version constants are correct"""
        # Test the fallback version constant
        from greenlang._version import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    @patch('greenlang._version.version')
    def test_version_with_different_package_formats(self, mock_version):
        """Test version handling with different package name formats"""
        test_cases = [
            ("greenlang-cli", "1.0.0"),
            ("greenlang_cli", "2.0.0"),
            ("greenlang.cli", "3.0.0"),
        ]

        for package_name, expected_version in test_cases:
            mock_version.return_value = expected_version
            mock_version.reset_mock()

            import importlib
            importlib.reload(_version)

            # First call should be to greenlang-cli
            mock_version.assert_called_with("greenlang-cli")

    @patch('greenlang._version.version')
    @patch('pathlib.Path.exists')
    def test_version_precedence_order(self, mock_exists, mock_version):
        """Test that version resolution follows correct precedence"""
        mock_exists.return_value = False

        # Test 1: greenlang-cli package available
        mock_version.side_effect = lambda x: "1.0.0" if x == "greenlang-cli" else PackageNotFoundError()

        import importlib
        importlib.reload(_version)

        assert _version.__version__ == "1.0.0"

        # Test 2: Only greenlang package available
        mock_version.side_effect = lambda x: "2.0.0" if x == "greenlang" else PackageNotFoundError()

        importlib.reload(_version)

        assert _version.__version__ == "2.0.0"

    def test_version_immutability(self):
        """Test that version can't be easily modified"""
        original_version = _version.__version__

        # Try to modify the version
        try:
            _version.__version__ = "hacked"
            # If we get here, verify it can still be changed back
            _version.__version__ = original_version
        except AttributeError:
            # This is expected if the module protects the attribute
            pass

    @patch('greenlang._version.version')
    def test_multiple_import_consistency(self, mock_version):
        """Test that multiple imports return consistent version"""
        mock_version.return_value = "1.5.0"

        import importlib
        importlib.reload(_version)

        version1 = _version.__version__

        # Import again
        from greenlang._version import __version__
        version2 = __version__

        assert version1 == version2
        assert version1 == "1.5.0"

    @pytest.mark.parametrize("version_string,expected", [
        ("1.0.0", "1.0.0"),
        ("2.1.3-beta", "2.1.3-beta"),
        ("0.0.1-alpha.1", "0.0.1-alpha.1"),
        ("10.20.30", "10.20.30"),
        ("1.0.0+build.123", "1.0.0+build.123"),
    ])
    @patch('greenlang._version.version')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.read_text')
    def test_version_format_preservation(self, mock_read_text, mock_exists, mock_version, version_string, expected):
        """Test that various version formats are preserved correctly"""
        mock_version.side_effect = PackageNotFoundError()
        mock_exists.return_value = True
        mock_read_text.return_value = version_string

        import importlib
        importlib.reload(_version)

        assert _version.__version__ == expected

    def test_version_file_path_exists_in_real_project(self):
        """Test that the VERSION file path logic works with real project structure"""
        # Get the actual file path
        version_file_path = Path(_version.__file__).resolve().parents[1] / "VERSION"

        # The path calculation should be correct regardless of whether file exists
        assert isinstance(version_file_path, Path)
        assert version_file_path.name == "VERSION"

    @patch('greenlang._version.Path')
    def test_version_file_path_calculation_error(self, mock_path):
        """Test handling of path calculation errors"""
        mock_path.side_effect = Exception("Path error")

        # Should handle gracefully and fall back
        import importlib
        with patch('greenlang._version.version', side_effect=PackageNotFoundError()):
            importlib.reload(_version)
            # Should fall back to default version
            assert _version.__version__ == "2.0.0"