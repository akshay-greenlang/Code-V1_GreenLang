# -*- coding: utf-8 -*-
"""
Unit Tests for Pack Loader

Tests loading, validation, and execution of GreenLang packs.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import json


class TestPackLoader:
    """Test PackLoader functionality"""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test pack loader"""
        try:
            from greenlang.packs.loader import PackLoader
            self.loader = PackLoader()
        except ImportError:
            try:
                from core.greenlang.packs.loader import PackLoader
                self.loader = PackLoader()
            except ImportError:
                pytest.skip("PackLoader not available")

        self.test_pack_dir = tmp_path / "test_pack"
        self.test_pack_dir.mkdir()

    def test_load_valid_pack(self):
        """Test loading a valid pack"""
        # Create valid manifest
        manifest = {
            "name": "test-pack",
            "version": "1.0.0",
            "agents": [
                {
                    "id": "fuel_agent",
                    "type": "deterministic",
                    "module": "greenlang.agents.fuel_agent"
                }
            ]
        }

        manifest_path = self.test_pack_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)

        pack = self.loader.load_pack(str(self.test_pack_dir))

        assert pack is not None
        assert pack.name == "test-pack"
        assert pack.version == "1.0.0"
        assert len(pack.agents) == 1

    def test_load_pack_missing_manifest(self):
        """Test loading pack with missing manifest"""
        with pytest.raises(FileNotFoundError, match="manifest"):
            self.loader.load_pack(str(self.test_pack_dir))

    def test_validate_pack_schema(self):
        """Test pack manifest schema validation"""
        invalid_manifest = {
            "name": "test-pack",
            # Missing version
            "agents": []
        }

        manifest_path = self.test_pack_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(invalid_manifest, f)

        with pytest.raises(ValueError, match="version"):
            self.loader.load_pack(str(self.test_pack_dir))

    def test_pack_agent_discovery(self):
        """Test discovering agents in pack"""
        manifest = {
            "name": "multi-agent-pack",
            "version": "1.0.0",
            "agents": [
                {"id": "fuel_agent", "type": "deterministic"},
                {"id": "grid_agent", "type": "deterministic"},
                {"id": "carbon_agent_ai", "type": "ai"}
            ]
        }

        manifest_path = self.test_pack_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)

        pack = self.loader.load_pack(str(self.test_pack_dir))

        assert len(pack.agents) == 3
        assert pack.get_agent("fuel_agent") is not None
        assert pack.get_agent("grid_agent") is not None


class TestPackValidator:
    """Test pack validation"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup pack validator"""
        try:
            from greenlang.packs.validator import PackValidator
            self.validator = PackValidator()
        except ImportError:
            pytest.skip("PackValidator not available")

    def test_validate_pack_structure(self):
        """Test validating pack directory structure"""
        # Pack should have:
        # - manifest.json
        # - agents/ directory
        # - README.md
        pass

    def test_validate_agent_compatibility(self):
        """Test validating agent compatibility"""
        # Agents should implement required interface
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
