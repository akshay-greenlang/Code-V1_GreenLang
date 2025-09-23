"""
Unit tests for pack loading functionality.

Tests cover:
- Pack discovery
- Manifest parsing
- Pack validation
- Dependency resolution
- Component loading
- Entry point discovery
- Local pack discovery
- Archive loading
"""

import json
import tempfile
import tarfile
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open, Mock, MagicMock
from typing import Dict, Any, List
import importlib.metadata as md

from greenlang.packs.loader import (
    PackLoader, LoadedPack, discover_installed, discover_local_packs,
    load_from_path, parse_pack_ref, version_matches
)
from greenlang.packs.manifest import PackManifest, Contents
from greenlang.sdk.base import Agent


class TestPackLoader:
    """Test the main PackLoader class."""

    def test_pack_loader_initialization(self):
        """Test PackLoader initialization."""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader()

            assert loader.cache_dir.name == "cache"
            assert loader.loaded_packs == {}
            assert loader.discovered_packs == {}

    def test_pack_loader_custom_cache_dir(self):
        """Test PackLoader with custom cache directory."""
        custom_cache = Path("/tmp/custom_cache")

        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=custom_cache)

            assert loader.cache_dir == custom_cache

    @patch('greenlang.packs.loader.discover_installed')
    @patch('greenlang.packs.loader.discover_local_packs')
    def test_discover_all(self, mock_discover_local, mock_discover_installed):
        """Test discovering all available packs."""
        # Mock discovered packs
        installed_packs = {"pack1": Mock(spec=PackManifest)}
        local_packs = {"pack2": Mock(spec=PackManifest)}

        mock_discover_installed.return_value = installed_packs
        mock_discover_local.return_value = local_packs

        with patch.object(Path, 'exists', return_value=True):
            loader = PackLoader()

            # Should have both installed and local packs
            assert "pack1" in loader.discovered_packs
            assert "pack2" in loader.discovered_packs

    def test_load_already_loaded_pack(self):
        """Test loading a pack that's already loaded."""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader()

            # Mock already loaded pack
            mock_loaded_pack = Mock(spec=LoadedPack)
            loader.loaded_packs["test_pack"] = mock_loaded_pack

            result = loader.load("test_pack")

            assert result == mock_loaded_pack

    def test_load_pack_not_found(self):
        """Test loading a pack that doesn't exist."""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader()

            with pytest.raises(ValueError, match="Pack not found"):
                loader.load("nonexistent_pack")

    @patch('greenlang.packs.loader.load_manifest')
    def test_load_from_path_success(self, mock_load_manifest):
        """Test successfully loading a pack from path."""
        # Create mock manifest
        mock_manifest = Mock(spec=PackManifest)
        mock_manifest.name = "test_pack"
        mock_load_manifest.return_value = mock_manifest

        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader()

            # Mock loaded pack
            with patch.object(LoadedPack, 'load_components'):
                pack_path = Path("/fake/path")
                result = loader._load_from_path(pack_path)

                assert isinstance(result, LoadedPack)
                assert result.manifest == mock_manifest
                assert result.path == pack_path

    def test_resolve_pack_path_discovered(self):
        """Test resolving pack path from discovered packs."""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader()

            # Add mock discovered pack
            mock_manifest = Mock(spec=PackManifest)
            mock_manifest.version = "1.0.0"
            mock_manifest._location = "/discovered/path"
            loader.discovered_packs["test_pack"] = mock_manifest

            result = loader._resolve_pack_path("test_pack")

            assert result == Path("/discovered/path")

    def test_resolve_pack_path_version_mismatch(self):
        """Test resolving pack path with version mismatch."""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader()

            # Add mock discovered pack with different version
            mock_manifest = Mock(spec=PackManifest)
            mock_manifest.version = "1.0.0"
            loader.discovered_packs["test_pack"] = mock_manifest

            # Request different version
            result = loader._resolve_pack_path("test_pack", "2.0.0")

            assert result is None

    def test_resolve_pack_path_direct_path(self):
        """Test resolving pack path using direct path."""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader()

            with patch.object(Path, 'exists', return_value=True):
                result = loader._resolve_pack_path("/direct/path")

                assert result == Path("/direct/path")

    def test_load_from_archive(self):
        """Test loading pack from archive."""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader()

            # Mock tarfile extraction
            with patch('tarfile.open') as mock_tarfile:
                mock_tar = Mock()
                mock_tar.extractall = Mock()
                mock_tarfile.return_value.__enter__.return_value = mock_tar

                # Mock pathlib operations
                with patch('pathlib.Path.glob') as mock_glob:
                    with patch('pathlib.Path.exists', return_value=True):
                        mock_pack_yaml = Mock()
                        mock_pack_yaml.parent = Mock()
                        mock_glob.return_value = [mock_pack_yaml]

                        with patch.object(loader, '_load_from_path') as mock_load:
                            mock_loaded_pack = Mock(spec=LoadedPack)
                            mock_load.return_value = mock_loaded_pack

                            fake_archive = Path("/fake/archive.glpack")
                            result = loader.load_from_archive(fake_archive)

                            assert result == mock_loaded_pack

    def test_list_available(self):
        """Test listing available packs."""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader()
            loader.discovered_packs = {"pack1": Mock(), "pack2": Mock()}

            result = loader.list_available()

            assert set(result) == {"pack1", "pack2"}

    def test_get_manifest(self):
        """Test getting pack manifest."""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader()

            mock_manifest = Mock(spec=PackManifest)
            loader.discovered_packs["test_pack"] = mock_manifest

            result = loader.get_manifest("test_pack")

            assert result == mock_manifest

    def test_get_agent_from_pack(self):
        """Test getting agent from loaded pack."""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader()

            # Mock loaded pack with agent
            mock_loaded_pack = Mock(spec=LoadedPack)
            mock_agent_class = Mock()
            mock_loaded_pack.get_agent.return_value = mock_agent_class
            loader.loaded_packs["test_pack"] = mock_loaded_pack

            result = loader.get_agent("test_pack:test_agent")

            assert result == mock_agent_class
            mock_loaded_pack.get_agent.assert_called_once_with("test_agent")

    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_get_agent_from_file(self, mock_module_from_spec, mock_spec_from_file):
        """Test getting agent from file path."""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader()

            # Mock file exists
            agent_file = Path("/path/to/agent.py")
            with patch.object(agent_file, 'exists', return_value=True):
                # Mock module loading
                mock_spec = Mock()
                mock_spec.loader = Mock()
                mock_spec_from_file.return_value = mock_spec

                mock_module = Mock()
                mock_module_from_spec.return_value = mock_module

                # Mock agent class
                class TestAgent(Agent):
                    pass

                with patch('inspect.getmembers') as mock_getmembers:
                    mock_getmembers.return_value = [("TestAgent", TestAgent)]

                    result = loader.get_agent("/path/to/agent.py")

                    assert result == TestAgent


class TestLoadedPack:
    """Test the LoadedPack class."""

    def test_loaded_pack_initialization(self):
        """Test LoadedPack initialization."""
        mock_manifest = Mock(spec=PackManifest)
        mock_loader = Mock(spec=PackLoader)
        pack_path = Path("/test/path")

        loaded_pack = LoadedPack(mock_manifest, pack_path, mock_loader)

        assert loaded_pack.manifest == mock_manifest
        assert loaded_pack.path == pack_path
        assert loaded_pack.loader == mock_loader
        assert loaded_pack.agents == {}
        assert loaded_pack.pipelines == {}
        assert loaded_pack.datasets == {}

    @patch('sys.path')
    def test_load_components(self, mock_sys_path):
        """Test loading pack components."""
        mock_manifest = Mock(spec=PackManifest)
        mock_loader = Mock(spec=PackLoader)
        pack_path = Path("/test/path")

        loaded_pack = LoadedPack(mock_manifest, pack_path, mock_loader)

        # Mock component loading methods
        with patch.object(loaded_pack, '_load_agents') as mock_load_agents:
            with patch.object(loaded_pack, '_load_pipelines') as mock_load_pipelines:
                with patch.object(loaded_pack, '_load_datasets') as mock_load_datasets:
                    with patch.object(loaded_pack, '_load_reports') as mock_load_reports:
                        loaded_pack.load_components()

                        # Should call all loading methods
                        mock_load_agents.assert_called_once()
                        mock_load_pipelines.assert_called_once()
                        mock_load_datasets.assert_called_once()
                        mock_load_reports.assert_called_once()

    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_load_agents_from_file(self, mock_module_from_spec, mock_spec_from_file):
        """Test loading agents from files."""
        # Create mock manifest with agent files
        mock_contents = Mock(spec=Contents)
        mock_contents.agents = ["agents/test_agent.py"]

        mock_manifest = Mock(spec=PackManifest)
        mock_manifest.name = "test_pack"
        mock_manifest.contents = mock_contents

        mock_loader = Mock(spec=PackLoader)
        pack_path = Path("/test/path")

        loaded_pack = LoadedPack(mock_manifest, pack_path, mock_loader)

        # Mock agent file exists
        agent_file = pack_path / "agents/test_agent.py"
        with patch.object(agent_file, 'exists', return_value=True):
            # Mock module loading
            mock_spec = Mock()
            mock_spec.loader = Mock()
            mock_spec_from_file.return_value = mock_spec

            mock_module = Mock()
            mock_module_from_spec.return_value = mock_module

            # Mock agent class
            class TestAgent(Agent):
                pass

            with patch('inspect.getmembers') as mock_getmembers:
                mock_getmembers.return_value = [("TestAgent", TestAgent)]

                loaded_pack._load_agents()

                assert "TestAgent" in loaded_pack.agents
                assert loaded_pack.agents["TestAgent"] == TestAgent

    def test_load_pipelines(self):
        """Test loading pipeline definitions."""
        # Create mock manifest with pipelines
        mock_contents = Mock(spec=Contents)
        mock_contents.pipelines = ["pipelines/test_pipeline.yaml"]

        mock_manifest = Mock(spec=PackManifest)
        mock_manifest.contents = mock_contents

        mock_loader = Mock(spec=PackLoader)
        pack_path = Path("/test/path")

        loaded_pack = LoadedPack(mock_manifest, pack_path, mock_loader)

        # Mock pipeline file
        pipeline_file = pack_path / "pipelines/test_pipeline.yaml"
        pipeline_data = {"name": "test_pipeline", "steps": []}

        with patch.object(pipeline_file, 'exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=yaml.dump(pipeline_data))):
                loaded_pack._load_pipelines()

                assert "test_pipeline" in loaded_pack.pipelines
                assert loaded_pack.pipelines["test_pipeline"] == pipeline_data

    def test_load_datasets(self):
        """Test loading dataset metadata."""
        # Create mock manifest with datasets
        mock_contents = Mock(spec=Contents)
        mock_contents.datasets = ["data.csv"]

        mock_manifest = Mock(spec=PackManifest)
        mock_manifest.contents = mock_contents

        mock_loader = Mock(spec=PackLoader)
        pack_path = Path("/test/path")

        loaded_pack = LoadedPack(mock_manifest, pack_path, mock_loader)

        # Mock dataset file
        dataset_file = pack_path / "datasets/data.csv"
        with patch.object(dataset_file, 'exists', return_value=True):
            with patch.object(dataset_file, 'stat') as mock_stat:
                mock_stat.return_value.st_size = 1024

                loaded_pack._load_datasets()

                assert "data.csv" in loaded_pack.datasets
                assert loaded_pack.datasets["data.csv"]["size"] == 1024

    def test_load_reports(self):
        """Test loading report templates."""
        # Create mock manifest with reports
        mock_contents = Mock(spec=Contents)
        mock_contents.reports = ["summary.md"]

        mock_manifest = Mock(spec=PackManifest)
        mock_manifest.contents = mock_contents

        mock_loader = Mock(spec=PackLoader)
        pack_path = Path("/test/path")

        loaded_pack = LoadedPack(mock_manifest, pack_path, mock_loader)

        # Mock report file
        report_file = pack_path / "reports/summary.md"
        report_content = "# Test Report\nThis is a test report."

        with patch.object(report_file, 'exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=report_content)):
                loaded_pack._load_reports()

                assert "summary.md" in loaded_pack.reports
                assert loaded_pack.reports["summary.md"]["template"] == report_content

    def test_get_methods(self):
        """Test getter methods for components."""
        mock_manifest = Mock(spec=PackManifest)
        mock_loader = Mock(spec=PackLoader)
        pack_path = Path("/test/path")

        loaded_pack = LoadedPack(mock_manifest, pack_path, mock_loader)

        # Add test components
        test_agent = Mock()
        test_pipeline = {"name": "test"}
        test_dataset = {"path": "/data"}
        test_report = {"template": "content"}

        loaded_pack.agents["test_agent"] = test_agent
        loaded_pack.pipelines["test_pipeline"] = test_pipeline
        loaded_pack.datasets["test_dataset"] = test_dataset
        loaded_pack.reports["test_report"] = test_report

        # Test getters
        assert loaded_pack.get_agent("test_agent") == test_agent
        assert loaded_pack.get_pipeline("test_pipeline") == test_pipeline
        assert loaded_pack.get_dataset("test_dataset") == test_dataset
        assert loaded_pack.get_report("test_report") == test_report

        # Test non-existent components
        assert loaded_pack.get_agent("nonexistent") is None
        assert loaded_pack.get_pipeline("nonexistent") is None


class TestPackDiscovery:
    """Test pack discovery functions."""

    @patch('importlib.metadata.entry_points')
    def test_discover_installed_success(self, mock_entry_points):
        """Test discovering installed packs via entry points."""
        # Mock entry point
        mock_ep = Mock()
        mock_ep.name = "test_pack"
        mock_ep.load.return_value = lambda: "/path/to/pack.yaml"

        mock_entry_points.return_value = [mock_ep]

        # Mock manifest loading
        with patch('greenlang.packs.manifest.PackManifest.from_yaml') as mock_from_yaml:
            mock_manifest = Mock(spec=PackManifest)
            mock_manifest.name = "test_pack"
            mock_manifest.version = "1.0.0"
            mock_from_yaml.return_value = mock_manifest

            with patch.object(Path, 'exists', return_value=True):
                result = discover_installed()

                assert "test_pack" in result
                assert result["test_pack"] == mock_manifest

    @patch('importlib.metadata.entry_points')
    def test_discover_installed_no_entry_points(self, mock_entry_points):
        """Test discovering installed packs when no entry points exist."""
        mock_entry_points.return_value = []

        result = discover_installed()

        assert result == {}

    def test_discover_local_packs_success(self):
        """Test discovering local packs."""
        base_dir = Path("/test/packs")

        # Mock pack directories
        pack1_yaml = base_dir / "pack1" / "pack.yaml"
        pack2_yaml = base_dir / "pack2" / "pack.yaml"

        with patch.object(base_dir, 'exists', return_value=True):
            with patch.object(base_dir, 'glob') as mock_glob:
                mock_glob.return_value = [pack1_yaml, pack2_yaml]

                # Mock manifest loading
                with patch('greenlang.packs.manifest.PackManifest.from_yaml') as mock_from_yaml:
                    mock_manifest1 = Mock(spec=PackManifest)
                    mock_manifest1.name = "pack1"
                    mock_manifest1.version = "1.0.0"

                    mock_manifest2 = Mock(spec=PackManifest)
                    mock_manifest2.name = "pack2"
                    mock_manifest2.version = "2.0.0"

                    mock_from_yaml.side_effect = [mock_manifest1, mock_manifest2]

                    result = discover_local_packs(base_dir)

                    assert len(result) == 2
                    assert "pack1" in result
                    assert "pack2" in result

    def test_discover_local_packs_no_directory(self):
        """Test discovering local packs when directory doesn't exist."""
        base_dir = Path("/nonexistent")

        with patch.object(base_dir, 'exists', return_value=False):
            result = discover_local_packs(base_dir)

            assert result == {}

    def test_load_from_path_pack_yaml(self):
        """Test loading manifest from pack.yaml path."""
        pack_yaml = Path("/test/pack.yaml")

        with patch('greenlang.packs.manifest.PackManifest.from_yaml') as mock_from_yaml:
            mock_manifest = Mock(spec=PackManifest)
            mock_from_yaml.return_value = mock_manifest

            result = load_from_path(str(pack_yaml))

            assert result == mock_manifest
            mock_from_yaml.assert_called_once_with(pack_yaml.parent)

    def test_load_from_path_directory(self):
        """Test loading manifest from directory path."""
        pack_dir = Path("/test/pack")
        pack_yaml = pack_dir / "pack.yaml"

        with patch.object(pack_yaml, 'exists', return_value=True):
            with patch('greenlang.packs.manifest.PackManifest.from_yaml') as mock_from_yaml:
                mock_manifest = Mock(spec=PackManifest)
                mock_from_yaml.return_value = mock_manifest

                result = load_from_path(str(pack_dir))

                assert result == mock_manifest

    def test_load_from_path_not_found(self):
        """Test loading manifest when pack.yaml not found."""
        pack_dir = Path("/test/nonexistent")
        pack_yaml = pack_dir / "pack.yaml"

        with patch.object(pack_yaml, 'exists', return_value=False):
            with pytest.raises(ValueError, match="No pack.yaml found"):
                load_from_path(str(pack_dir))


class TestPackReference:
    """Test pack reference parsing and version matching."""

    def test_parse_pack_ref_name_only(self):
        """Test parsing pack reference with name only."""
        name, version = parse_pack_ref("test_pack")

        assert name == "test_pack"
        assert version is None

    def test_parse_pack_ref_with_version(self):
        """Test parsing pack reference with version."""
        name, version = parse_pack_ref("test_pack@1.0.0")

        assert name == "test_pack"
        assert version == "1.0.0"

    def test_parse_pack_ref_with_constraint(self):
        """Test parsing pack reference with version constraint."""
        test_cases = [
            ("test_pack>=1.0.0", "test_pack", ">=1.0.0"),
            ("test_pack<=2.0.0", "test_pack", "<=2.0.0"),
            ("test_pack==1.5.0", "test_pack", "==1.5.0"),
            ("test_pack>1.0", "test_pack", ">1.0"),
            ("test_pack<2.0", "test_pack", "<2.0"),
        ]

        for pack_ref, expected_name, expected_version in test_cases:
            name, version = parse_pack_ref(pack_ref)
            assert name == expected_name
            assert version == expected_version

    def test_version_matches_no_constraint(self):
        """Test version matching with no constraint."""
        assert version_matches("1.0.0", None) is True
        assert version_matches("2.5.1", "") is True

    def test_version_matches_exact(self):
        """Test exact version matching."""
        assert version_matches("1.0.0", "==1.0.0") is True
        assert version_matches("1.0.0", "==1.0.1") is False

    def test_version_matches_range(self):
        """Test version range matching."""
        assert version_matches("1.5.0", ">=1.0.0") is True
        assert version_matches("0.9.0", ">=1.0.0") is False

        assert version_matches("1.5.0", "<=2.0.0") is True
        assert version_matches("2.1.0", "<=2.0.0") is False

    @patch('greenlang.packs.loader.logger')
    def test_version_matches_fallback(self, mock_logger):
        """Test version matching fallback when packaging is unavailable."""
        with patch('greenlang.packs.loader.specifiers', side_effect=ImportError()):
            # Should use fallback comparison
            assert version_matches("1.5.0", ">=1.0.0") is True
            assert version_matches("0.9.0", ">=1.0.0") is False

            # Should log warning about fallback
            mock_logger.warning.assert_called()

    def test_version_matches_compatible_release(self):
        """Test compatible release version matching."""
        with patch('greenlang.packs.loader.specifiers', side_effect=ImportError()):
            # Test ~= operator (compatible release)
            assert version_matches("1.4.5", "~=1.4.2") is True
            assert version_matches("1.5.0", "~=1.4.2") is False