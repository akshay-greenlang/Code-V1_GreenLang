# -*- coding: utf-8 -*-
"""
Comprehensive tests for greenlang.packs.loader module
"""

import pytest
import sys
import tempfile
import tarfile
import inspect
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, call

# Add the greenlang directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "greenlang"))

from greenlang.packs.loader import (
    PackLoader, LoadedPack, discover_installed, discover_local_packs,
    load_from_path, parse_pack_ref, version_matches, ENTRY_GROUP
)
from greenlang.sdk.base import Agent


class TestPackLoader:
    """Test PackLoader class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "cache"

    def test_pack_loader_creation_default(self):
        """Test creating PackLoader with default cache directory"""
        with patch('pathlib.Path.home') as mock_home, \
             patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch.object(PackLoader, '_discover_all'):

            mock_home.return_value = self.temp_dir
            loader = PackLoader()

            expected_cache = self.temp_dir / ".greenlang" / "cache"
            assert loader.cache_dir == expected_cache
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_pack_loader_creation_custom_cache(self):
        """Test creating PackLoader with custom cache directory"""
        custom_cache = self.temp_dir / "custom_cache"

        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch.object(PackLoader, '_discover_all'):

            loader = PackLoader(cache_dir=custom_cache)

            assert loader.cache_dir == custom_cache
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('greenlang.packs.loader.discover_installed')
    @patch('greenlang.packs.loader.discover_local_packs')
    @patch('pathlib.Path.exists')
    def test_discover_all(self, mock_exists, mock_discover_local, mock_discover_installed):
        """Test _discover_all method"""
        mock_discover_installed.return_value = {"installed_pack": MagicMock()}
        mock_discover_local.return_value = {"local_pack": MagicMock()}
        mock_exists.return_value = True

        with patch('pathlib.Path.cwd', return_value=self.temp_dir), \
             patch('pathlib.Path.home', return_value=self.temp_dir):

            loader = PackLoader(cache_dir=self.cache_dir)

        assert "installed_pack" in loader.discovered_packs
        assert "local_pack" in loader.discovered_packs
        mock_discover_installed.assert_called_once()
        assert mock_discover_local.call_count >= 1

    def test_load_already_loaded(self):
        """Test load method with already loaded pack"""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

        # Add a loaded pack
        mock_loaded_pack = MagicMock()
        loader.loaded_packs["test-pack"] = mock_loaded_pack

        result = loader.load("test-pack")
        assert result is mock_loaded_pack

    @patch.object(PackLoader, '_resolve_pack_path')
    @patch.object(PackLoader, '_load_from_path')
    def test_load_new_pack(self, mock_load_from_path, mock_resolve):
        """Test load method with new pack"""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

        mock_resolve.return_value = Path("/path/to/pack")
        mock_loaded_pack = MagicMock()
        mock_load_from_path.return_value = mock_loaded_pack

        result = loader.load("new-pack", verify=True)

        assert result is mock_loaded_pack
        assert "new-pack" in loader.loaded_packs
        mock_resolve.assert_called_once_with("new-pack", None)
        mock_load_from_path.assert_called_once_with(Path("/path/to/pack"), True)

    def test_load_pack_not_found(self):
        """Test load method with pack not found"""
        with patch.object(PackLoader, '_discover_all'), \
             patch.object(PackLoader, '_resolve_pack_path', return_value=None):

            loader = PackLoader(cache_dir=self.cache_dir)

            with pytest.raises(ValueError, match="Pack not found"):
                loader.load("nonexistent-pack")

    def test_resolve_pack_path_discovered(self):
        """Test _resolve_pack_path with discovered pack"""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

        # Add a discovered pack
        mock_manifest = MagicMock()
        mock_manifest.version = "1.0.0"
        mock_manifest._location = "/discovered/path"
        loader.discovered_packs["discovered-pack"] = mock_manifest

        result = loader._resolve_pack_path("discovered-pack")
        assert result == Path("/discovered/path")

    def test_resolve_pack_path_version_mismatch(self):
        """Test _resolve_pack_path with version mismatch"""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

        # Add a discovered pack with wrong version
        mock_manifest = MagicMock()
        mock_manifest.version = "1.0.0"
        loader.discovered_packs["version-pack"] = mock_manifest

        result = loader._resolve_pack_path("version-pack", "2.0.0")
        assert result is None

    @patch('pathlib.Path.exists')
    def test_resolve_pack_path_direct_path(self, mock_exists):
        """Test _resolve_pack_path with direct path"""
        mock_exists.return_value = True

        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

        result = loader._resolve_pack_path("/direct/path/to/pack")
        assert result == Path("/direct/path/to/pack")

    @patch('pathlib.Path.exists')
    def test_resolve_pack_path_cached(self, mock_exists):
        """Test _resolve_pack_path with cached pack"""
        def exists_side_effect(path):
            return str(path).endswith("cached-pack")

        mock_exists.side_effect = exists_side_effect

        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

        result = loader._resolve_pack_path("cached-pack")
        assert result == loader.cache_dir / "cached-pack"

    @patch('greenlang.packs.loader.load_manifest')
    def test_load_from_path(self, mock_load_manifest):
        """Test _load_from_path method"""
        mock_manifest = MagicMock()
        mock_load_manifest.return_value = mock_manifest

        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

        with patch.object(LoadedPack, 'load_components') as mock_load_components:
            result = loader._load_from_path(Path("/test/pack"), verify=True)

        assert isinstance(result, LoadedPack)
        assert result.manifest is mock_manifest
        assert result.path == Path("/test/pack")
        assert result.loader is loader
        mock_load_components.assert_called_once()

    @patch('pathlib.Path.exists')
    @patch('tarfile.open')
    @patch('pathlib.Path.glob')
    def test_load_from_archive(self, mock_glob, mock_tarfile, mock_exists):
        """Test load_from_archive method"""
        mock_exists.side_effect = lambda path: str(path).endswith(".glpack")
        mock_tar = MagicMock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar

        # Mock finding pack.yaml in extracted directory
        mock_pack_yaml = MagicMock()
        mock_pack_yaml.parent = Path("/extracted/pack")
        mock_glob.return_value = [mock_pack_yaml]

        with patch.object(PackLoader, '_discover_all'), \
             patch.object(PackLoader, '_load_from_path') as mock_load_from_path:

            loader = PackLoader(cache_dir=self.cache_dir)
            mock_loaded_pack = MagicMock()
            mock_load_from_path.return_value = mock_loaded_pack

            result = loader.load_from_archive(Path("/test/pack.glpack"))

        assert result is mock_loaded_pack
        mock_tar.extractall.assert_called_once()

    @patch('pathlib.Path.exists')
    def test_load_from_archive_not_found(self, mock_exists):
        """Test load_from_archive with non-existent archive"""
        mock_exists.return_value = False

        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

            with pytest.raises(ValueError, match="Archive not found"):
                loader.load_from_archive(Path("/nonexistent.glpack"))

    def test_list_available(self):
        """Test list_available method"""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

        loader.discovered_packs = {
            "pack1": MagicMock(),
            "pack2": MagicMock(),
            "pack3": MagicMock()
        }

        result = loader.list_available()
        assert set(result) == {"pack1", "pack2", "pack3"}

    def test_get_manifest(self):
        """Test get_manifest method"""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

        mock_manifest = MagicMock()
        loader.discovered_packs["test-pack"] = mock_manifest

        result = loader.get_manifest("test-pack")
        assert result is mock_manifest

        result = loader.get_manifest("nonexistent")
        assert result is None

    def test_get_agent_pack_format(self):
        """Test get_agent with pack:agent format"""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

        # Mock loaded pack
        mock_loaded_pack = MagicMock()
        mock_agent_class = MagicMock()
        mock_loaded_pack.get_agent.return_value = mock_agent_class
        loader.loaded_packs["test-pack"] = mock_loaded_pack

        result = loader.get_agent("test-pack:TestAgent")
        assert result is mock_agent_class

    def test_get_agent_discover_and_load(self):
        """Test get_agent with pack discovery and loading"""
        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

        # Mock discovered pack
        mock_manifest = MagicMock()
        loader.discovered_packs["discovered-pack"] = mock_manifest

        # Mock loading process
        mock_loaded_pack = MagicMock()
        mock_agent_class = MagicMock()
        mock_loaded_pack.get_agent.return_value = mock_agent_class

        with patch.object(loader, 'load', return_value=mock_loaded_pack):
            result = loader.get_agent("discovered-pack:Agent")

        assert result is mock_agent_class

    @patch('pathlib.Path.exists')
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_get_agent_file_path(self, mock_module_from_spec, mock_spec, mock_exists):
        """Test get_agent with file path"""
        mock_exists.return_value = True

        # Mock module loading
        mock_spec_obj = MagicMock()
        mock_loader = MagicMock()
        mock_spec_obj.loader = mock_loader
        mock_spec.return_value = mock_spec_obj

        mock_module = MagicMock()
        mock_module_from_spec.return_value = mock_module

        # Create a mock Agent class
        class MockAgent(Agent):
            def validate(self, input_data):
                return True
            def process(self, input_data):
                return input_data

        # Mock inspect.getmembers to return our mock agent
        with patch('inspect.getmembers') as mock_getmembers, \
             patch.object(PackLoader, '_discover_all'):

            mock_getmembers.return_value = [("MockAgent", MockAgent)]
            loader = PackLoader(cache_dir=self.cache_dir)

            result = loader.get_agent("/path/to/agent.py")

        assert result is MockAgent

    @patch('pathlib.Path.exists')
    @patch('importlib.util.spec_from_file_location')
    def test_get_agent_file_with_class_name(self, mock_spec, mock_exists):
        """Test get_agent with file path and specific class name"""
        mock_exists.return_value = True

        # Mock module loading
        mock_spec_obj = MagicMock()
        mock_loader = MagicMock()
        mock_spec_obj.loader = mock_loader
        mock_spec.return_value = mock_spec_obj

        mock_module = MagicMock()
        mock_agent_class = MagicMock()
        mock_module.SpecificAgent = mock_agent_class

        with patch('importlib.util.module_from_spec', return_value=mock_module), \
             patch.object(PackLoader, '_discover_all'):

            loader = PackLoader(cache_dir=self.cache_dir)
            result = loader.get_agent("/path/to/agent.py:SpecificAgent")

        assert result is mock_agent_class

    @patch('pathlib.Path.exists')
    def test_get_agent_file_not_found(self, mock_exists):
        """Test get_agent with non-existent file"""
        mock_exists.return_value = False

        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

            with pytest.raises(ValueError, match="Agent file not found"):
                loader.get_agent("/nonexistent/agent.py")

    @patch('pathlib.Path.exists')
    @patch('importlib.util.spec_from_file_location')
    def test_get_agent_invalid_module(self, mock_spec, mock_exists):
        """Test get_agent with invalid module"""
        mock_exists.return_value = True
        mock_spec.return_value = None

        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

            with pytest.raises(ValueError, match="Cannot load module"):
                loader.get_agent("/invalid/agent.py")

    @patch('pathlib.Path.exists')
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_get_agent_class_not_found(self, mock_module_from_spec, mock_spec, mock_exists):
        """Test get_agent with class not found in module"""
        mock_exists.return_value = True

        # Mock module loading
        mock_spec_obj = MagicMock()
        mock_spec_obj.loader = MagicMock()
        mock_spec.return_value = mock_spec_obj

        mock_module = MagicMock()
        mock_module_from_spec.return_value = mock_module
        # Module doesn't have the requested class
        del mock_module.MissingAgent

        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

            with pytest.raises(ValueError, match="Class MissingAgent not found"):
                loader.get_agent("/path/to/agent.py:MissingAgent")

    @patch('pathlib.Path.exists')
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    @patch('inspect.getmembers')
    def test_get_agent_no_agent_subclass(self, mock_getmembers, mock_module_from_spec, mock_spec, mock_exists):
        """Test get_agent with no Agent subclass in module"""
        mock_exists.return_value = True

        # Mock module loading
        mock_spec_obj = MagicMock()
        mock_spec_obj.loader = MagicMock()
        mock_spec.return_value = mock_spec_obj
        mock_module_from_spec.return_value = MagicMock()

        # No Agent subclasses found
        mock_getmembers.return_value = [("SomeClass", str)]

        with patch.object(PackLoader, '_discover_all'):
            loader = PackLoader(cache_dir=self.cache_dir)

            with pytest.raises(ValueError, match="No Agent subclass found"):
                loader.get_agent("/path/to/agent.py")


class TestLoadedPack:
    """Test LoadedPack class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_manifest = MagicMock()
        self.mock_loader = MagicMock()

    def test_loaded_pack_creation(self):
        """Test creating LoadedPack instance"""
        pack = LoadedPack(
            manifest=self.mock_manifest,
            path=self.temp_dir,
            loader=self.mock_loader
        )

        assert pack.manifest is self.mock_manifest
        assert pack.path == self.temp_dir
        assert pack.loader is self.mock_loader
        assert isinstance(pack.agents, dict)
        assert isinstance(pack.pipelines, dict)
        assert isinstance(pack.datasets, dict)
        assert isinstance(pack.reports, dict)
        assert isinstance(pack.connectors, dict)

    @patch('sys.path')
    def test_load_components(self, mock_sys_path):
        """Test load_components method"""
        pack = LoadedPack(
            manifest=self.mock_manifest,
            path=self.temp_dir,
            loader=self.mock_loader
        )

        with patch.object(pack, '_load_agents') as mock_load_agents, \
             patch.object(pack, '_load_pipelines') as mock_load_pipelines, \
             patch.object(pack, '_load_datasets') as mock_load_datasets, \
             patch.object(pack, '_load_reports') as mock_load_reports:

            pack.load_components()

        mock_load_agents.assert_called_once()
        mock_load_pipelines.assert_called_once()
        mock_load_datasets.assert_called_once()
        mock_load_reports.assert_called_once()

    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    @patch('inspect.getmembers')
    @patch('pathlib.Path.exists')
    def test_load_agents_file_path(self, mock_exists, mock_getmembers, mock_module_from_spec, mock_spec):
        """Test _load_agents with file paths"""
        mock_exists.return_value = True

        # Mock manifest contents
        mock_contents = MagicMock()
        mock_contents.agents = ["agents/test_agent.py"]
        self.mock_manifest.contents = mock_contents
        self.mock_manifest.name = "test-pack"

        # Mock module loading
        mock_spec_obj = MagicMock()
        mock_spec_obj.loader = MagicMock()
        mock_spec.return_value = mock_spec_obj
        mock_module_from_spec.return_value = MagicMock()

        # Mock Agent class
        class MockAgent(Agent):
            def validate(self, input_data):
                return True
            def process(self, input_data):
                return input_data

        mock_getmembers.return_value = [("MockAgent", MockAgent)]

        pack = LoadedPack(
            manifest=self.mock_manifest,
            path=self.temp_dir,
            loader=self.mock_loader
        )

        pack._load_agents()

        assert "MockAgent" in pack.agents
        assert pack.agents["MockAgent"] is MockAgent

    def test_load_agents_no_contents(self):
        """Test _load_agents with no contents"""
        self.mock_manifest.contents = None

        pack = LoadedPack(
            manifest=self.mock_manifest,
            path=self.temp_dir,
            loader=self.mock_loader
        )

        pack._load_agents()
        assert len(pack.agents) == 0

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_load_pipelines(self, mock_yaml, mock_file, mock_exists):
        """Test _load_pipelines method"""
        mock_exists.return_value = True
        mock_yaml.return_value = {"name": "test_pipeline", "steps": []}

        # Mock manifest contents
        mock_contents = MagicMock()
        mock_contents.pipelines = ["pipeline1.yaml"]
        self.mock_manifest.contents = mock_contents

        pack = LoadedPack(
            manifest=self.mock_manifest,
            path=self.temp_dir,
            loader=self.mock_loader
        )

        pack._load_pipelines()

        assert "pipeline1" in pack.pipelines
        assert pack.pipelines["pipeline1"]["name"] == "test_pipeline"

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_load_datasets(self, mock_stat, mock_exists):
        """Test _load_datasets method"""
        mock_exists.return_value = True
        mock_stat_obj = MagicMock()
        mock_stat_obj.st_size = 1024
        mock_stat.return_value = mock_stat_obj

        # Mock manifest contents
        mock_contents = MagicMock()
        mock_contents.datasets = ["data.csv"]
        self.mock_manifest.contents = mock_contents

        # Mock dataset card
        with patch('builtins.open', mock_open(read_data="# Dataset Card")):
            pack = LoadedPack(
                manifest=self.mock_manifest,
                path=self.temp_dir,
                loader=self.mock_loader
            )

            pack._load_datasets()

        assert "data.csv" in pack.datasets
        assert pack.datasets["data.csv"]["size"] == 1024
        assert pack.datasets["data.csv"]["format"] == "csv"

    @patch('pathlib.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_reports(self, mock_file, mock_exists):
        """Test _load_reports method"""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = "# Report Template"

        # Mock manifest contents
        mock_contents = MagicMock()
        mock_contents.reports = ["report.md"]
        self.mock_manifest.contents = mock_contents

        pack = LoadedPack(
            manifest=self.mock_manifest,
            path=self.temp_dir,
            loader=self.mock_loader
        )

        pack._load_reports()

        assert "report.md" in pack.reports
        assert pack.reports["report.md"]["template"] == "# Report Template"

    def test_get_agent(self):
        """Test get_agent method"""
        pack = LoadedPack(
            manifest=self.mock_manifest,
            path=self.temp_dir,
            loader=self.mock_loader
        )

        # Add a test agent
        mock_agent = MagicMock()
        pack.agents["TestAgent"] = mock_agent

        result = pack.get_agent("TestAgent")
        assert result is mock_agent

        result = pack.get_agent("NonexistentAgent")
        assert result is None

    def test_get_pipeline(self):
        """Test get_pipeline method"""
        pack = LoadedPack(
            manifest=self.mock_manifest,
            path=self.temp_dir,
            loader=self.mock_loader
        )

        # Add a test pipeline
        test_pipeline = {"name": "test", "steps": []}
        pack.pipelines["test"] = test_pipeline

        result = pack.get_pipeline("test")
        assert result is test_pipeline

        result = pack.get_pipeline("nonexistent")
        assert result is None

    def test_get_dataset(self):
        """Test get_dataset method"""
        pack = LoadedPack(
            manifest=self.mock_manifest,
            path=self.temp_dir,
            loader=self.mock_loader
        )

        # Add a test dataset
        test_dataset = {"name": "test.csv", "size": 1024}
        pack.datasets["test.csv"] = test_dataset

        result = pack.get_dataset("test.csv")
        assert result is test_dataset

        result = pack.get_dataset("nonexistent.csv")
        assert result is None

    def test_get_report(self):
        """Test get_report method"""
        pack = LoadedPack(
            manifest=self.mock_manifest,
            path=self.temp_dir,
            loader=self.mock_loader
        )

        # Add a test report
        test_report = {"name": "test.md", "template": "# Test"}
        pack.reports["test.md"] = test_report

        result = pack.get_report("test.md")
        assert result is test_report

        result = pack.get_report("nonexistent.md")
        assert result is None


class TestModuleFunctions:
    """Test module-level functions"""

    @patch('importlib.metadata.entry_points')
    @patch('pathlib.Path.exists')
    @patch('greenlang.packs.loader.PackManifest.from_yaml')
    def test_discover_installed_python310(self, mock_from_yaml, mock_exists, mock_entry_points):
        """Test discover_installed for Python 3.10+"""
        # Mock entry point
        mock_ep = MagicMock()
        mock_ep.name = "test-pack"
        mock_ep.load.return_value = lambda: "/path/to/pack.yaml"
        mock_entry_points.return_value = [mock_ep]

        mock_exists.return_value = True

        # Mock manifest
        mock_manifest = MagicMock()
        mock_manifest.name = "test-pack"
        mock_manifest.version = "1.0.0"
        mock_from_yaml.return_value = mock_manifest

        with patch('sys.version_info', (3, 10, 0)):
            result = discover_installed()

        assert "test-pack" in result
        assert result["test-pack"] is mock_manifest
        assert hasattr(result["test-pack"], "_location")

    @patch('importlib.metadata.entry_points')
    @patch('pathlib.Path.exists')
    @patch('greenlang.packs.loader.PackManifest.from_yaml')
    def test_discover_installed_python39(self, mock_from_yaml, mock_exists, mock_entry_points):
        """Test discover_installed for Python < 3.10"""
        # Mock entry point
        mock_ep = MagicMock()
        mock_ep.name = "test-pack"
        mock_ep.load.return_value = "/path/to/pack.yaml"
        mock_entry_points.return_value = {ENTRY_GROUP: [mock_ep]}

        mock_exists.return_value = True

        # Mock manifest
        mock_manifest = MagicMock()
        mock_manifest.name = "test-pack"
        mock_manifest.version = "1.0.0"
        mock_from_yaml.return_value = mock_manifest

        with patch('sys.version_info', (3, 9, 0)):
            result = discover_installed()

        assert "test-pack" in result

    @patch('importlib.metadata.entry_points')
    def test_discover_installed_no_entry_points(self, mock_entry_points):
        """Test discover_installed with no entry points"""
        mock_entry_points.side_effect = Exception("No entry points")

        result = discover_installed()
        assert result == {}

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    @patch('greenlang.packs.loader.PackManifest.from_yaml')
    def test_discover_local_packs(self, mock_from_yaml, mock_glob, mock_exists):
        """Test discover_local_packs function"""
        mock_exists.return_value = True

        # Mock pack.yaml files
        mock_pack_yaml1 = MagicMock()
        mock_pack_yaml1.parent = Path("/pack1")
        mock_pack_yaml2 = MagicMock()
        mock_pack_yaml2.parent = Path("/pack2")
        mock_glob.return_value = [mock_pack_yaml1, mock_pack_yaml2]

        # Mock manifests
        mock_manifest1 = MagicMock()
        mock_manifest1.name = "pack1"
        mock_manifest1.version = "1.0.0"
        mock_manifest2 = MagicMock()
        mock_manifest2.name = "pack2"
        mock_manifest2.version = "2.0.0"
        mock_from_yaml.side_effect = [mock_manifest1, mock_manifest2]

        result = discover_local_packs(Path("/base/dir"))

        assert len(result) == 2
        assert "pack1" in result
        assert "pack2" in result

    @patch('pathlib.Path.exists')
    def test_discover_local_packs_no_directory(self, mock_exists):
        """Test discover_local_packs with non-existent directory"""
        mock_exists.return_value = False

        result = discover_local_packs(Path("/nonexistent"))
        assert result == {}

    @patch('pathlib.Path.is_file')
    @patch('pathlib.Path.exists')
    @patch('greenlang.packs.loader.PackManifest.from_yaml')
    def test_load_from_path_file(self, mock_from_yaml, mock_exists, mock_is_file):
        """Test load_from_path with pack.yaml file"""
        mock_is_file.return_value = True
        mock_exists.return_value = True

        mock_manifest = MagicMock()
        mock_from_yaml.return_value = mock_manifest

        result = load_from_path("/path/to/pack.yaml")
        assert result is mock_manifest

    @patch('pathlib.Path.is_file')
    @patch('pathlib.Path.exists')
    @patch('greenlang.packs.loader.PackManifest.from_yaml')
    def test_load_from_path_directory(self, mock_from_yaml, mock_exists, mock_is_file):
        """Test load_from_path with directory"""
        mock_is_file.return_value = False
        mock_exists.return_value = True

        mock_manifest = MagicMock()
        mock_from_yaml.return_value = mock_manifest

        result = load_from_path("/path/to/pack")
        assert result is mock_manifest

    @patch('pathlib.Path.exists')
    def test_load_from_path_not_found(self, mock_exists):
        """Test load_from_path with non-existent path"""
        mock_exists.return_value = False

        with pytest.raises(ValueError, match="No pack.yaml found"):
            load_from_path("/nonexistent/path")

    @pytest.mark.parametrize("pack_ref,expected_name,expected_version", [
        ("simple-pack", "simple-pack", None),
        ("pack@1.0.0", "pack", "1.0.0"),
        ("pack>=1.2.0", "pack", ">=1.2.0"),
        ("pack<=2.0.0", "pack", "<=2.0.0"),
        ("pack==1.5.0", "pack", "==1.5.0"),
        ("pack>1.0", "pack", ">1.0"),
        ("pack<2.0", "pack", "<2.0"),
    ])
    def test_parse_pack_ref(self, pack_ref, expected_name, expected_version):
        """Test parse_pack_ref function"""
        name, version = parse_pack_ref(pack_ref)
        assert name == expected_name
        assert version == expected_version

    @pytest.mark.parametrize("actual,constraint,expected", [
        ("1.0.0", None, True),
        ("1.0.0", ">=1.0.0", True),
        ("1.0.0", ">=1.1.0", False),
        ("2.0.0", "<=2.0.0", True),
        ("2.1.0", "<=2.0.0", False),
        ("1.5.0", "==1.5.0", True),
        ("1.5.0", "==1.6.0", False),
        ("1.0.0", "!=1.0.0", False),
        ("1.1.0", "!=1.0.0", True),
        ("1.5.0", ">1.0.0", True),
        ("1.0.0", ">1.0.0", False),
        ("1.0.0", "<2.0.0", True),
        ("2.0.0", "<2.0.0", False),
    ])
    def test_version_matches_fallback(self, actual, constraint, expected):
        """Test version_matches function with fallback logic"""
        with patch('greenlang.packs.loader.version', side_effect=ImportError()):
            result = version_matches(actual, constraint)
            assert result == expected

    @patch('greenlang.packs.loader.version')
    @patch('greenlang.packs.loader.specifiers')
    def test_version_matches_packaging(self, mock_specifiers, mock_version):
        """Test version_matches with packaging library"""
        mock_actual_version = MagicMock()
        mock_version.parse.return_value = mock_actual_version

        mock_spec_set = MagicMock()
        mock_spec_set.__contains__.return_value = True
        mock_specifiers.SpecifierSet.return_value = mock_spec_set

        result = version_matches("1.0.0", ">=1.0.0")
        assert result is True

        mock_version.parse.assert_called_once_with("1.0.0")
        mock_specifiers.SpecifierSet.assert_called_once_with(">=1.0.0")

    def test_version_matches_compatible_release(self):
        """Test version_matches with compatible release operator"""
        with patch('greenlang.packs.loader.version', side_effect=ImportError()):
            # ~=1.4.2 should match >=1.4.2, <1.5.0
            assert version_matches("1.4.2", "~=1.4.2") is True
            assert version_matches("1.4.5", "~=1.4.2") is True
            assert version_matches("1.5.0", "~=1.4.2") is False
            assert version_matches("1.3.9", "~=1.4.2") is False

    def test_version_matches_error_handling(self):
        """Test version_matches error handling"""
        with patch('greenlang.packs.loader.version', side_effect=ImportError()), \
             patch('greenlang.packs.loader.logger') as mock_logger:

            # Test with invalid constraint format
            result = version_matches("1.0.0", "invalid_constraint")
            assert result is True  # Should default to True on error

    def test_entry_group_constant(self):
        """Test ENTRY_GROUP constant"""
        assert ENTRY_GROUP == "greenlang.packs"