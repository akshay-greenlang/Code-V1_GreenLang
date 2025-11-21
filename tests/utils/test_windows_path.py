# -*- coding: utf-8 -*-
"""
Comprehensive test suite for Windows PATH utilities.

Tests PATH manipulation, backup/restore, and gl.exe discovery.
"""

import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the module to test
from greenlang.utils import windows_path


class TestPythonScriptsDetection:
    """Test detection of Python Scripts directories."""

    def test_get_python_scripts_directories_finds_current(self, tmp_path):
        """Test that current Python Scripts directory is found."""
        with patch('sys.executable', str(tmp_path / 'python.exe')):
            scripts_dir = tmp_path / 'Scripts'
            scripts_dir.mkdir(parents=True, exist_ok=True)

            result = windows_path.get_python_scripts_directories()

            assert len(result) > 0
            assert scripts_dir in result

    def test_get_python_scripts_directories_handles_missing(self, tmp_path):
        """Test behavior when Scripts directory doesn't exist."""
        with patch('sys.executable', str(tmp_path / 'python.exe')):
            # Don't create Scripts directory
            result = windows_path.get_python_scripts_directories()

            # Should return empty list or list without the missing directory
            assert isinstance(result, list)


class TestUserInstallDetection:
    """Test user installation detection."""

    def test_is_user_install_detection(self):
        """Test user install detection logic."""
        result = windows_path.is_user_install()
        assert isinstance(result, bool)

    def test_is_user_install_with_user_site(self, monkeypatch):
        """Test detection when user site is in sys.path."""
        mock_site = Mock()
        mock_site.getusersitepackages.return_value = "/home/user/.local/lib/python3.10/site-packages"

        with patch('site.getusersitepackages', mock_site.getusersitepackages):
            with patch('sys.path', ["/home/user/.local/lib/python3.10/site-packages"]):
                result = windows_path.is_user_install()
                assert isinstance(result, bool)


class TestPathChecks:
    """Test PATH checking utilities."""

    def test_is_in_path_positive(self, tmp_path):
        """Test detection of directory in PATH."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        with patch.dict(os.environ, {'PATH': str(test_dir)}):
            assert windows_path.is_in_path(test_dir)

    def test_is_in_path_negative(self, tmp_path):
        """Test detection of directory not in PATH."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        other_dir = tmp_path / "other"

        with patch.dict(os.environ, {'PATH': str(other_dir)}):
            assert not windows_path.is_in_path(test_dir)

    def test_is_in_path_empty_path(self, tmp_path):
        """Test behavior with empty PATH."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        with patch.dict(os.environ, {'PATH': ''}):
            assert not windows_path.is_in_path(test_dir)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific tests")
class TestWindowsPathManipulation:
    """Test Windows PATH manipulation (registry operations)."""

    def test_get_user_path_reads_registry(self):
        """Test reading PATH from registry."""
        result = windows_path.get_user_path()
        assert isinstance(result, str)

    def test_set_user_path_updates_registry(self, tmp_path):
        """Test setting PATH in registry."""
        # This is a potentially dangerous test, so we'll mock the registry operations
        with patch('greenlang.utils.windows_path.winreg') as mock_winreg:
            mock_key = MagicMock()
            mock_winreg.OpenKey.return_value.__enter__.return_value = mock_key

            result = windows_path.set_user_path("C:\\test\\path")

            # Should have attempted to set the value
            mock_winreg.SetValueEx.assert_called_once()

    def test_add_to_user_path_new_directory(self, tmp_path):
        """Test adding new directory to PATH."""
        test_dir = tmp_path / "new_dir"
        test_dir.mkdir()

        with patch('greenlang.utils.windows_path.is_in_path', return_value=False):
            with patch('greenlang.utils.windows_path.get_user_path', return_value="C:\\existing"):
                with patch('greenlang.utils.windows_path.set_user_path', return_value=True) as mock_set:
                    with patch('greenlang.utils.windows_path.backup_user_path'):
                        result = windows_path.add_to_user_path(test_dir)

                        assert result is True
                        # Should have called set_user_path with directory at start
                        call_args = mock_set.call_args[0][0]
                        assert str(test_dir) in call_args

    def test_add_to_user_path_already_exists(self, tmp_path):
        """Test adding directory already in PATH."""
        test_dir = tmp_path / "existing"
        test_dir.mkdir()

        with patch('greenlang.utils.windows_path.is_in_path', return_value=True):
            result = windows_path.add_to_user_path(test_dir)

            # Should return True without modifying PATH
            assert result is True

    def test_remove_from_user_path(self, tmp_path):
        """Test removing directory from PATH."""
        test_dir = tmp_path / "to_remove"
        test_dir.mkdir()

        mock_path = f"C:\\other{os.pathsep}{test_dir}{os.pathsep}C:\\another"

        with patch('greenlang.utils.windows_path.is_in_path', return_value=True):
            with patch('greenlang.utils.windows_path.get_user_path', return_value=mock_path):
                with patch('greenlang.utils.windows_path.set_user_path', return_value=True) as mock_set:
                    with patch('greenlang.utils.windows_path.backup_user_path'):
                        result = windows_path.remove_from_user_path(test_dir)

                        assert result is True
                        # Should have called set_user_path without the removed directory
                        call_args = mock_set.call_args[0][0]
                        assert str(test_dir) not in call_args

    def test_remove_from_user_path_not_in_path(self, tmp_path):
        """Test removing directory not in PATH."""
        test_dir = tmp_path / "not_there"
        test_dir.mkdir()

        with patch('greenlang.utils.windows_path.is_in_path', return_value=False):
            result = windows_path.remove_from_user_path(test_dir)

            # Should return True without attempting removal
            assert result is True


class TestPathBackup:
    """Test PATH backup and restore functionality."""

    def test_backup_user_path_creates_file(self, tmp_path):
        """Test that backup creates a JSON file."""
        backup_dir = tmp_path / "backup"

        with patch('greenlang.utils.windows_path.BACKUP_DIR', backup_dir):
            with patch('greenlang.utils.windows_path.get_user_path', return_value="C:\\test\\path"):
                if sys.platform == "win32":
                    backup_file = windows_path.backup_user_path()

                    assert backup_file is not None
                    assert backup_file.exists()
                    assert backup_file.suffix == '.json'

                    # Verify backup content
                    with open(backup_file, 'r') as f:
                        data = json.load(f)
                        assert 'timestamp' in data
                        assert 'path' in data
                        assert 'entries' in data

    def test_cleanup_old_backups(self, tmp_path):
        """Test that old backups are removed."""
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir(parents=True)

        # Create 15 fake backup files
        for i in range(15):
            backup_file = backup_dir / f"path_2024010{i:02d}_120000.json"
            backup_file.write_text(json.dumps({"timestamp": f"2024-01-{i:02d}", "path": "", "entries": []}))

        with patch('greenlang.utils.windows_path.BACKUP_DIR', backup_dir):
            windows_path.cleanup_old_backups(max_backups=10)

            remaining = list(backup_dir.glob("path_*.json"))
            assert len(remaining) == 10

    def test_list_path_backups(self, tmp_path):
        """Test listing available backups."""
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir(parents=True)

        # Create a few backup files
        for i in range(3):
            backup_file = backup_dir / f"path_2024010{i}_120000.json"
            backup_data = {
                "timestamp": f"2024-01-0{i}T12:00:00",
                "path": f"C:\\test{i}",
                "entries": [f"C:\\test{i}"]
            }
            backup_file.write_text(json.dumps(backup_data))

        with patch('greenlang.utils.windows_path.BACKUP_DIR', backup_dir):
            backups = windows_path.list_path_backups()

            assert len(backups) == 3
            assert all('timestamp' in b for b in backups)
            assert all('entries_count' in b for b in backups)
            assert all('file' in b for b in backups)

    def test_restore_path_from_backup(self, tmp_path):
        """Test restoring PATH from backup."""
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir(parents=True)

        backup_file = backup_dir / "path_20240101_120000.json"
        backup_data = {
            "timestamp": "2024-01-01T12:00:00",
            "path": "C:\\restored\\path",
            "entries": ["C:\\restored\\path"]
        }
        backup_file.write_text(json.dumps(backup_data))

        with patch('greenlang.utils.windows_path.BACKUP_DIR', backup_dir):
            with patch('greenlang.utils.windows_path.set_user_path', return_value=True) as mock_set:
                if sys.platform == "win32":
                    success, message = windows_path.restore_path_from_backup(backup_file)

                    assert success is True
                    assert "Successfully restored" in message
                    mock_set.assert_called_once_with("C:\\restored\\path")

    def test_restore_path_from_most_recent_backup(self, tmp_path):
        """Test restoring from most recent backup when none specified."""
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir(parents=True)

        # Create multiple backups
        for i in range(3):
            backup_file = backup_dir / f"path_2024010{i}_120000.json"
            backup_data = {
                "timestamp": f"2024-01-0{i}T12:00:00",
                "path": f"C:\\test{i}",
                "entries": [f"C:\\test{i}"]
            }
            backup_file.write_text(json.dumps(backup_data))

        with patch('greenlang.utils.windows_path.BACKUP_DIR', backup_dir):
            with patch('greenlang.utils.windows_path.set_user_path', return_value=True):
                if sys.platform == "win32":
                    success, message = windows_path.restore_path_from_backup()

                    assert success is True

    def test_restore_path_no_backups(self, tmp_path):
        """Test restore behavior when no backups exist."""
        backup_dir = tmp_path / "empty_backup"
        backup_dir.mkdir(parents=True)

        with patch('greenlang.utils.windows_path.BACKUP_DIR', backup_dir):
            if sys.platform == "win32":
                success, message = windows_path.restore_path_from_backup()

                assert success is False
                assert "No PATH backups" in message


class TestGLExecutableDiscovery:
    """Test gl.exe discovery functionality."""

    def test_find_gl_executable_found(self, tmp_path):
        """Test finding gl.exe when it exists."""
        scripts_dir = tmp_path / "Scripts"
        scripts_dir.mkdir(parents=True)
        gl_exe = scripts_dir / "gl.exe"
        gl_exe.touch()

        with patch('greenlang.utils.windows_path.get_python_scripts_directories', return_value=[scripts_dir]):
            result = windows_path.find_gl_executable()

            assert result is not None
            assert result == gl_exe

    def test_find_gl_executable_not_found(self, tmp_path):
        """Test behavior when gl.exe doesn't exist."""
        scripts_dir = tmp_path / "Scripts"
        scripts_dir.mkdir(parents=True)
        # Don't create gl.exe

        with patch('greenlang.utils.windows_path.get_python_scripts_directories', return_value=[scripts_dir]):
            result = windows_path.find_gl_executable()

            assert result is None


class TestSetupWindowsPath:
    """Test main Windows PATH setup function."""

    def test_setup_windows_path_success(self, tmp_path):
        """Test successful PATH setup."""
        scripts_dir = tmp_path / "Scripts"
        scripts_dir.mkdir(parents=True)
        gl_exe = scripts_dir / "gl.exe"
        gl_exe.touch()

        with patch('sys.platform', 'win32'):
            with patch('greenlang.utils.windows_path.get_python_scripts_directories', return_value=[scripts_dir]):
                with patch('greenlang.utils.windows_path.find_gl_executable', return_value=gl_exe):
                    with patch('greenlang.utils.windows_path.is_in_path', return_value=False):
                        with patch('greenlang.utils.windows_path.add_to_user_path', return_value=True):
                            success, message = windows_path.setup_windows_path()

                            assert success is True
                            assert "Successfully added" in message

    def test_setup_windows_path_already_in_path(self, tmp_path):
        """Test setup when already in PATH."""
        scripts_dir = tmp_path / "Scripts"
        scripts_dir.mkdir(parents=True)
        gl_exe = scripts_dir / "gl.exe"
        gl_exe.touch()

        with patch('sys.platform', 'win32'):
            with patch('greenlang.utils.windows_path.get_python_scripts_directories', return_value=[scripts_dir]):
                with patch('greenlang.utils.windows_path.find_gl_executable', return_value=gl_exe):
                    with patch('greenlang.utils.windows_path.is_in_path', return_value=True):
                        success, message = windows_path.setup_windows_path()

                        assert success is True
                        assert "already accessible" in message

    def test_setup_windows_path_gl_not_found(self):
        """Test setup when gl.exe doesn't exist."""
        with patch('sys.platform', 'win32'):
            with patch('greenlang.utils.windows_path.get_python_scripts_directories', return_value=[Path("/test")]):
                with patch('greenlang.utils.windows_path.find_gl_executable', return_value=None):
                    success, message = windows_path.setup_windows_path()

                    assert success is False
                    assert "not found" in message

    def test_setup_windows_path_non_windows(self):
        """Test behavior on non-Windows platforms."""
        with patch('sys.platform', 'linux'):
            success, message = windows_path.setup_windows_path()

            assert success is False
            assert "only for Windows" in message


class TestDiagnosePathIssues:
    """Test PATH diagnostics."""

    def test_diagnose_path_issues_structure(self):
        """Test diagnostic output structure."""
        diagnosis = windows_path.diagnose_path_issues()

        assert isinstance(diagnosis, dict)
        assert "platform" in diagnosis
        assert "python_executable" in diagnosis
        assert "scripts_directories" in diagnosis
        assert "gl_executable_found" in diagnosis
        assert "gl_executable_path" in diagnosis
        assert "in_path" in diagnosis
        assert "path_entries" in diagnosis
        assert "recommendations" in diagnosis

    def test_diagnose_path_issues_with_gl_found(self, tmp_path):
        """Test diagnostics when gl.exe is found."""
        scripts_dir = tmp_path / "Scripts"
        scripts_dir.mkdir(parents=True)
        gl_exe = scripts_dir / "gl.exe"
        gl_exe.touch()

        with patch('greenlang.utils.windows_path.get_python_scripts_directories', return_value=[scripts_dir]):
            with patch('greenlang.utils.windows_path.find_gl_executable', return_value=gl_exe):
                with patch('greenlang.utils.windows_path.is_in_path', return_value=True):
                    diagnosis = windows_path.diagnose_path_issues()

                    assert diagnosis["gl_executable_found"] is True
                    assert diagnosis["gl_executable_path"] == str(gl_exe)
                    assert diagnosis["in_path"] is True
                    assert "should be working correctly" in diagnosis["recommendations"][0].lower()

    def test_diagnose_path_issues_gl_not_in_path(self, tmp_path):
        """Test diagnostics when gl.exe exists but not in PATH."""
        scripts_dir = tmp_path / "Scripts"
        scripts_dir.mkdir(parents=True)
        gl_exe = scripts_dir / "gl.exe"
        gl_exe.touch()

        with patch('greenlang.utils.windows_path.get_python_scripts_directories', return_value=[scripts_dir]):
            with patch('greenlang.utils.windows_path.find_gl_executable', return_value=gl_exe):
                with patch('greenlang.utils.windows_path.is_in_path', return_value=False):
                    diagnosis = windows_path.diagnose_path_issues()

                    assert diagnosis["gl_executable_found"] is True
                    assert diagnosis["in_path"] is False
                    assert any("Add" in rec or "gl doctor --setup-path" in rec for rec in diagnosis["recommendations"])
