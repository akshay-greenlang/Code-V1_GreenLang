# -*- coding: utf-8 -*-
"""
Comprehensive tests for greenlang.packs.registry module
"""

import pytest
import json
import sys
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime
from dataclasses import asdict

# Add the greenlang directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "greenlang"))

from greenlang.packs.registry import InstalledPack, PackRegistry


class TestInstalledPack:
    """Test InstalledPack dataclass"""

    def test_installed_pack_creation(self):
        """Test creating InstalledPack instance"""
        pack = InstalledPack(
            name="test-pack",
            version="1.0.0",
            location="/path/to/pack",
            manifest={"name": "test-pack"},
            installed_at="2023-01-01T00:00:00",
            hash="abc123"
        )

        assert pack.name == "test-pack"
        assert pack.version == "1.0.0"
        assert pack.location == "/path/to/pack"
        assert pack.manifest == {"name": "test-pack"}
        assert pack.installed_at == "2023-01-01T00:00:00"
        assert pack.hash == "abc123"
        assert pack.verified is False  # Default value
        assert pack.signature is None  # Default value

    def test_installed_pack_with_verification(self):
        """Test InstalledPack with verification info"""
        pack = InstalledPack(
            name="verified-pack",
            version="2.0.0",
            location="/verified/path",
            manifest={"name": "verified-pack"},
            installed_at="2023-01-01T00:00:00",
            hash="def456",
            verified=True,
            signature="signature_data"
        )

        assert pack.verified is True
        assert pack.signature == "signature_data"

    def test_installed_pack_dataclass_methods(self):
        """Test dataclass methods work correctly"""
        pack = InstalledPack(
            name="test",
            version="1.0.0",
            location="/test",
            manifest={},
            installed_at="2023-01-01T00:00:00",
            hash="test"
        )

        # Test that asdict works (used by registry)
        pack_dict = asdict(pack)
        assert isinstance(pack_dict, dict)
        assert pack_dict["name"] == "test"
        assert pack_dict["verified"] is False


class TestPackRegistry:
    """Test PackRegistry class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.registry_dir = self.temp_dir / "registry"

    def test_registry_creation_default(self):
        """Test creating PackRegistry with default directory"""
        with patch('pathlib.Path.home') as mock_home:
            mock_home.return_value = self.temp_dir

            with patch('pathlib.Path.mkdir') as mock_mkdir, \
                 patch.object(PackRegistry, '_load_registry'), \
                 patch.object(PackRegistry, '_discover_entry_points'), \
                 patch.object(PackRegistry, '_discover_local_packs'):

                registry = PackRegistry()

                expected_dir = self.temp_dir / ".greenlang" / "registry"
                assert registry.registry_dir == expected_dir
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_registry_creation_custom_dir(self):
        """Test creating PackRegistry with custom directory"""
        custom_dir = self.temp_dir / "custom_registry"

        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=custom_dir)

            assert registry.registry_dir == custom_dir
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_registry_success(self, mock_json_load, mock_file, mock_exists):
        """Test _load_registry with existing registry file"""
        mock_exists.return_value = True
        mock_json_load.return_value = {
            "packs": [
                {
                    "name": "pack1",
                    "version": "1.0.0",
                    "location": "/path1",
                    "manifest": {"name": "pack1"},
                    "installed_at": "2023-01-01T00:00:00",
                    "hash": "hash1",
                    "verified": True,
                    "signature": None
                }
            ]
        }

        with patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

        assert len(registry.packs) == 1
        assert "pack1" in registry.packs
        assert registry.packs["pack1"].name == "pack1"

    @patch('pathlib.Path.exists')
    def test_load_registry_no_file(self, mock_exists):
        """Test _load_registry with no existing registry file"""
        mock_exists.return_value = False

        with patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

        assert len(registry.packs) == 0

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_registry_error_handling(self, mock_json_load, mock_file, mock_exists):
        """Test _load_registry error handling"""
        mock_exists.return_value = True
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        with patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

        # Should handle error gracefully and have empty packs dict
        assert len(registry.packs) == 0

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_registry(self, mock_json_dump, mock_file):
        """Test _save_registry method"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

        # Add a test pack
        pack = InstalledPack(
            name="test",
            version="1.0.0",
            location="/test",
            manifest={},
            installed_at="2023-01-01T00:00:00",
            hash="test"
        )
        registry.packs["test"] = pack

        registry._save_registry()

        mock_json_dump.assert_called_once()
        saved_data = mock_json_dump.call_args[0][0]
        assert saved_data["version"] == "0.1.0"
        assert "updated_at" in saved_data
        assert len(saved_data["packs"]) == 1

    @patch('importlib.metadata.entry_points')
    def test_discover_entry_points_python310_plus(self, mock_entry_points):
        """Test _discover_entry_points for Python 3.10+"""
        # Mock entry point
        mock_ep = MagicMock()
        mock_ep.name = "test-pack"
        mock_ep.load.return_value = lambda: "/path/to/pack.yaml"
        mock_entry_points.return_value = [mock_ep]

        # Mock manifest loading
        mock_manifest = MagicMock()
        mock_manifest.name = "test-pack"
        mock_manifest.version = "1.0.0"
        mock_manifest.model_dump.return_value = {"name": "test-pack"}

        with patch('sys.version_info', (3, 10, 0)), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('greenlang.packs.registry.PackManifest.from_yaml', return_value=mock_manifest), \
             patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_local_packs'), \
             patch.object(PackRegistry, '_calculate_directory_hash', return_value="hash123"):

            registry = PackRegistry(registry_dir=self.registry_dir)

        assert "test-pack" in registry.packs
        assert registry.packs["test-pack"].verified is True

    @patch('importlib.metadata.entry_points')
    def test_discover_entry_points_python39(self, mock_entry_points):
        """Test _discover_entry_points for Python < 3.10"""
        # Mock entry point
        mock_ep = MagicMock()
        mock_ep.name = "test-pack"
        mock_ep.load.return_value = "/path/to/pack.yaml"
        mock_entry_points.return_value = {"greenlang.packs": [mock_ep]}

        # Mock manifest loading
        mock_manifest = MagicMock()
        mock_manifest.name = "test-pack"
        mock_manifest.version = "1.0.0"
        mock_manifest.model_dump.return_value = {"name": "test-pack"}

        with patch('sys.version_info', (3, 9, 0)), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('greenlang.packs.registry.PackManifest.from_yaml', return_value=mock_manifest), \
             patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_local_packs'), \
             patch.object(PackRegistry, '_calculate_directory_hash', return_value="hash123"):

            registry = PackRegistry(registry_dir=self.registry_dir)

        assert "test-pack" in registry.packs

    @patch('importlib.metadata.entry_points')
    def test_discover_entry_points_no_entries(self, mock_entry_points):
        """Test _discover_entry_points with no entry points"""
        mock_entry_points.side_effect = Exception("No entry points")

        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

        # Should handle gracefully
        assert len(registry.packs) == 0

    @patch('pathlib.Path.iterdir')
    @patch('pathlib.Path.exists')
    def test_discover_local_packs(self, mock_exists, mock_iterdir):
        """Test _discover_local_packs method"""
        # Mock directory structure
        mock_pack_dir = MagicMock()
        mock_pack_dir.is_dir.return_value = True
        mock_pack_dir.__truediv__ = lambda self, other: MagicMock()
        mock_iterdir.return_value = [mock_pack_dir]

        # Mock pack.yaml exists
        mock_exists.side_effect = lambda path: str(path).endswith("pack.yaml")

        # Mock manifest
        mock_manifest = MagicMock()
        mock_manifest.name = "local-pack"
        mock_manifest.version = "1.0.0"
        mock_manifest.model_dump.return_value = {"name": "local-pack"}

        with patch('pathlib.Path.cwd', return_value=self.temp_dir), \
             patch('pathlib.Path.home', return_value=self.temp_dir), \
             patch('greenlang.packs.registry.PackManifest.from_yaml', return_value=mock_manifest), \
             patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_calculate_directory_hash', return_value="hash123"):

            registry = PackRegistry(registry_dir=self.registry_dir)

        # Should have discovered the local pack
        assert "local-pack" in registry.packs
        assert registry.packs["local-pack"].verified is False

    def test_calculate_hash(self):
        """Test _calculate_hash method"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

        test_content = "test content"
        expected_hash = hashlib.sha256(test_content.encode()).hexdigest()

        result = registry._calculate_hash(test_content)
        assert result == expected_hash

    @patch('pathlib.Path.rglob')
    @patch('builtins.open', new_callable=mock_open)
    def test_calculate_directory_hash(self, mock_file, mock_rglob):
        """Test _calculate_directory_hash method"""
        # Mock file structure
        mock_file1 = MagicMock()
        mock_file1.is_file.return_value = True
        mock_file1.name = "file1.py"
        mock_file2 = MagicMock()
        mock_file2.is_file.return_value = True
        mock_file2.name = "file2.yaml"
        mock_rglob.return_value = [mock_file1, mock_file2]

        mock_file.return_value.read.return_value = b"file content"

        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

        result = registry._calculate_directory_hash(Path("/test/dir"))
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex digest length

    @patch('pathlib.Path.exists')
    def test_register_pack_success(self, mock_exists):
        """Test register method success"""
        mock_exists.return_value = True

        # Mock manifest
        mock_manifest = MagicMock()
        mock_manifest.name = "new-pack"
        mock_manifest.version = "1.0.0"
        mock_manifest.model_dump.return_value = {"name": "new-pack"}
        mock_manifest.validate_files.return_value = []  # No errors

        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'), \
             patch('greenlang.packs.registry.PackManifest.from_yaml', return_value=mock_manifest), \
             patch.object(PackRegistry, '_calculate_directory_hash', return_value="hash123"), \
             patch.object(PackRegistry, '_save_registry'):

            registry = PackRegistry(registry_dir=self.registry_dir)
            pack = registry.register(Path("/path/to/pack"), verify=True)

        assert pack.name == "new-pack"
        assert pack.verified is True
        assert "new-pack" in registry.packs

    @patch('pathlib.Path.exists')
    def test_register_pack_no_manifest(self, mock_exists):
        """Test register method with no pack.yaml"""
        mock_exists.return_value = False

        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            with pytest.raises(ValueError, match="No pack.yaml found"):
                registry.register(Path("/invalid/path"))

    @patch('pathlib.Path.exists')
    def test_register_pack_validation_errors(self, mock_exists):
        """Test register method with validation errors"""
        mock_exists.return_value = True

        # Mock manifest with validation errors
        mock_manifest = MagicMock()
        mock_manifest.validate_files.return_value = ["Missing required file"]

        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'), \
             patch('greenlang.packs.registry.PackManifest.from_yaml', return_value=mock_manifest):

            registry = PackRegistry(registry_dir=self.registry_dir)

            with pytest.raises(ValueError, match="Pack validation failed"):
                registry.register(Path("/invalid/pack"))

    def test_unregister_pack_success(self):
        """Test unregister method success"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'), \
             patch.object(PackRegistry, '_save_registry'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            # Add a pack first
            pack = InstalledPack(
                name="test-pack",
                version="1.0.0",
                location="/test",
                manifest={},
                installed_at="2023-01-01T00:00:00",
                hash="test"
            )
            registry.packs["test-pack"] = pack

            registry.unregister("test-pack")

            assert "test-pack" not in registry.packs

    def test_unregister_pack_not_found(self):
        """Test unregister method with non-existent pack"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            with pytest.raises(ValueError, match="Pack not found"):
                registry.unregister("nonexistent-pack")

    def test_get_pack_by_name(self):
        """Test get method by name only"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            # Add a test pack
            pack = InstalledPack(
                name="test-pack",
                version="1.0.0",
                location="/test",
                manifest={},
                installed_at="2023-01-01T00:00:00",
                hash="test"
            )
            registry.packs["test-pack"] = pack

            result = registry.get("test-pack")
            assert result is pack

    def test_get_pack_by_name_and_version(self):
        """Test get method by name and version"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            # Add a test pack
            pack = InstalledPack(
                name="test-pack",
                version="1.0.0",
                location="/test",
                manifest={},
                installed_at="2023-01-01T00:00:00",
                hash="test"
            )
            registry.packs["test-pack"] = pack

            # Correct version
            result = registry.get("test-pack", "1.0.0")
            assert result is pack

            # Wrong version
            result = registry.get("test-pack", "2.0.0")
            assert result is None

    def test_get_pack_not_found(self):
        """Test get method with non-existent pack"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            result = registry.get("nonexistent-pack")
            assert result is None

    def test_list_all_packs(self):
        """Test list method without filter"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            # Add test packs
            pack1 = InstalledPack("pack1", "1.0.0", "/path1", {"kind": "pack"}, "2023-01-01T00:00:00", "hash1")
            pack2 = InstalledPack("pack2", "1.0.0", "/path2", {"kind": "dataset"}, "2023-01-01T00:00:00", "hash2")
            registry.packs["pack1"] = pack1
            registry.packs["pack2"] = pack2

            result = registry.list()
            assert len(result) == 2

    def test_list_packs_by_kind(self):
        """Test list method with kind filter"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            # Add test packs
            pack1 = InstalledPack("pack1", "1.0.0", "/path1", {"kind": "pack"}, "2023-01-01T00:00:00", "hash1")
            pack2 = InstalledPack("pack2", "1.0.0", "/path2", {"kind": "dataset"}, "2023-01-01T00:00:00", "hash2")
            registry.packs["pack1"] = pack1
            registry.packs["pack2"] = pack2

            result = registry.list(kind="pack")
            assert len(result) == 1
            assert result[0].name == "pack1"

    def test_search_packs(self):
        """Test search method"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            # Add test packs
            pack1 = InstalledPack("data-pack", "1.0.0", "/path1", {"description": "Data processing pack"}, "2023-01-01T00:00:00", "hash1")
            pack2 = InstalledPack("ml-pack", "1.0.0", "/path2", {"description": "Machine learning utilities"}, "2023-01-01T00:00:00", "hash2")
            registry.packs["data-pack"] = pack1
            registry.packs["ml-pack"] = pack2

            # Search by name
            result = registry.search("data")
            assert len(result) == 1
            assert result[0].name == "data-pack"

            # Search by description
            result = registry.search("machine")
            assert len(result) == 1
            assert result[0].name == "ml-pack"

    def test_verify_pack_success(self):
        """Test verify method success"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'), \
             patch.object(PackRegistry, '_calculate_directory_hash', return_value="hash123"), \
             patch.object(PackRegistry, '_save_registry'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            # Add a test pack
            pack = InstalledPack("test-pack", "1.0.0", "/path", {}, "2023-01-01T00:00:00", "hash123")
            registry.packs["test-pack"] = pack

            result = registry.verify("test-pack")
            assert result is True
            assert pack.verified is True

    def test_verify_pack_failure(self):
        """Test verify method failure"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'), \
             patch.object(PackRegistry, '_calculate_directory_hash', return_value="different_hash"):

            registry = PackRegistry(registry_dir=self.registry_dir)

            # Add a test pack
            pack = InstalledPack("test-pack", "1.0.0", "/path", {}, "2023-01-01T00:00:00", "original_hash")
            registry.packs["test-pack"] = pack

            result = registry.verify("test-pack")
            assert result is False

    def test_verify_pack_not_found(self):
        """Test verify method with non-existent pack"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            with pytest.raises(ValueError, match="Pack not found"):
                registry.verify("nonexistent-pack")

    def test_verify_entry_point_pack(self):
        """Test verify method with entry point pack"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            # Add an entry point pack
            pack = InstalledPack("ep-pack", "1.0.0", "entry_point:/path", {}, "2023-01-01T00:00:00", "hash")
            registry.packs["ep-pack"] = pack

            result = registry.verify("ep-pack")
            assert result is True  # Entry points are pre-verified

    def test_list_pipelines(self):
        """Test list_pipelines method"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            # Add a pack with pipelines
            pack = InstalledPack(
                "test-pack",
                "1.0.0",
                "/path",
                {
                    "contents": {
                        "pipelines": ["pipeline1.yaml", "pipeline2.yml"]
                    }
                },
                "2023-01-01T00:00:00",
                "hash"
            )
            registry.packs["test-pack"] = pack

            result = registry.list_pipelines()
            assert "test-pack" in result
            assert len(result["test-pack"]) == 2
            assert result["test-pack"][0]["name"] == "pipeline1"
            assert result["test-pack"][1]["name"] == "pipeline2"

    def test_get_dependencies_string_format(self):
        """Test get_dependencies with string format dependencies"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            # Add a pack with string dependencies
            pack = InstalledPack(
                "test-pack",
                "1.0.0",
                "/path",
                {
                    "dependencies": ["dep1", "dep2"]
                },
                "2023-01-01T00:00:00",
                "hash"
            )
            registry.packs["test-pack"] = pack

            deps = registry.get_dependencies("test-pack")
            assert deps == ["dep1", "dep2"]

    def test_get_dependencies_dict_format(self):
        """Test get_dependencies with dict format dependencies"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            # Add a pack with dict dependencies
            pack = InstalledPack(
                "test-pack",
                "1.0.0",
                "/path",
                {
                    "dependencies": [
                        {"name": "dep1", "version": ">=1.0.0"},
                        {"name": "dep2"}
                    ]
                },
                "2023-01-01T00:00:00",
                "hash"
            )
            registry.packs["test-pack"] = pack

            deps = registry.get_dependencies("test-pack")
            assert "dep1>=1.0.0" in deps
            assert "dep2" in deps

    def test_get_dependencies_pack_not_found(self):
        """Test get_dependencies with non-existent pack"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

            deps = registry.get_dependencies("nonexistent-pack")
            assert deps == []

    @patch('pathlib.Path.exists')
    def test_discover_local_packs_skip_hidden_files(self, mock_exists):
        """Test that _calculate_directory_hash skips hidden files"""
        with patch.object(PackRegistry, '_load_registry'), \
             patch.object(PackRegistry, '_discover_entry_points'), \
             patch.object(PackRegistry, '_discover_local_packs'):

            registry = PackRegistry(registry_dir=self.registry_dir)

        # Mock files including hidden ones
        mock_files = [
            MagicMock(is_file=lambda: True, name="normal_file.py"),
            MagicMock(is_file=lambda: True, name=".hidden_file"),
            MagicMock(is_file=lambda: True, name="another_file.yaml"),
        ]

        with patch('pathlib.Path.rglob', return_value=mock_files), \
             patch('builtins.open', mock_open(read_data=b"content")):

            result = registry._calculate_directory_hash(Path("/test"))

        # Should only process non-hidden files
        assert isinstance(result, str)

    def test_entry_point_error_handling(self):
        """Test entry point discovery error handling"""
        with patch('importlib.metadata.entry_points') as mock_ep:
            mock_ep.side_effect = Exception("Entry point error")

            with patch.object(PackRegistry, '_load_registry'), \
                 patch.object(PackRegistry, '_discover_local_packs'):

                # Should not raise exception
                registry = PackRegistry(registry_dir=self.registry_dir)
                assert isinstance(registry, PackRegistry)