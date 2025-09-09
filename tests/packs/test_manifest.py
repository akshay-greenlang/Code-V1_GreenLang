"""
Tests for pack manifest schema and loader.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from pydantic import ValidationError

from core.greenlang.packs.manifest import PackManifest, Compat, Policy, Security, Contents
from core.greenlang.packs.loader_simple import load_manifest, validate_pack


class TestPackManifest:
    """Test the PackManifest Pydantic model."""
    
    def test_valid_manifest(self):
        """Test that a valid manifest passes validation."""
        manifest_data = {
            "name": "test-pack",
            "version": "1.0.0",
            "kind": "pack",
            "license": "Apache-2.0",
            "compat": {
                "greenlang": ">=0.1.0",
                "python": ">=3.10"
            },
            "contents": {
                "pipelines": ["gl.yaml"],
                "agents": ["agents/calculator.py"],
                "datasets": ["data/emissions.csv"],
                "reports": ["reports/summary.md"]
            },
            "policy": {
                "network": ["api.weather.gov"],
                "data_residency": ["US", "EU"],
                "license_allowlist": ["Apache-2.0", "MIT"]
            },
            "security": {
                "sbom": "sbom.json",
                "signatures": ["signatures/pack.sig"]
            },
            "tests": ["tests/*.py"],
            "card": "CARD.md"
        }
        
        # Should not raise any exceptions
        manifest = PackManifest(**manifest_data)
        assert manifest.name == "test-pack"
        assert manifest.version == "1.0.0"
        assert manifest.kind == "pack"
    
    def test_invalid_version(self):
        """Test that invalid version format fails."""
        manifest_data = {
            "name": "test-pack",
            "version": "invalid",  # No dots
            "kind": "pack",
            "license": "MIT",
            "compat": {
                "greenlang": ">=0.1.0",
                "python": ">=3.10"
            },
            "contents": {
                "pipelines": ["gl.yaml"]
            },
            "card": "CARD.md"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PackManifest(**manifest_data)
        
        assert "Version must be in semantic versioning format" in str(exc_info.value)
    
    def test_invalid_name(self):
        """Test that invalid pack names fail."""
        test_cases = [
            ("Test-Pack", "Pack name must be kebab-case"),  # Uppercase not allowed
            ("test_pack", "Pack name must be kebab-case"),  # Underscore not allowed
            ("test@pack", "Pack name must be kebab-case"),  # Invalid character
        ]
        
        for invalid_name, expected_error in test_cases:
            manifest_data = {
                "name": invalid_name,
                "version": "1.0.0",
                "kind": "pack",
                "license": "MIT",
                "compat": {
                    "greenlang": ">=0.1.0",
                    "python": ">=3.10"
                },
                "contents": {
                    "pipelines": ["gl.yaml"]
                },
                "card": "CARD.md"
            }
            
            with pytest.raises(ValidationError) as exc_info:
                PackManifest(**manifest_data)
            
            assert expected_error in str(exc_info.value)
    
    def test_default_values(self):
        """Test that default values are applied correctly."""
        manifest_data = {
            "name": "minimal-pack",
            "version": "0.1.0",
            "license": "MIT",
            "compat": {
                "greenlang": ">=0.1.0",
                "python": ">=3.10"
            },
            "contents": {
                "pipelines": ["gl.yaml"]
            },
            "card": "CARD.md"
        }
        
        manifest = PackManifest(**manifest_data)
        
        # Check defaults
        assert manifest.kind == "pack"
        assert manifest.policy.license_allowlist == ["Apache-2.0", "MIT", "Commercial"]
        assert manifest.security.sbom is None
        assert manifest.security.signatures == []
        assert manifest.tests == []
        assert manifest.contents.agents == []
        assert manifest.contents.datasets == []


class TestPackLoader:
    """Test the pack loader functionality."""
    
    def test_load_valid_pack(self, tmp_path):
        """Test loading a valid pack with all files present."""
        # Create pack structure
        pack_dir = tmp_path / "test-pack"
        pack_dir.mkdir()
        
        # Create manifest
        manifest_data = {
            "name": "test-pack",
            "version": "1.0.0",
            "license": "MIT",
            "compat": {
                "greenlang": ">=0.1.0",
                "python": ">=3.10"
            },
            "contents": {
                "pipelines": ["gl.yaml"],
                "agents": ["agents/calc.py"],
                "datasets": ["data/test.csv"]
            },
            "tests": ["tests/*.py"],
            "card": "CARD.md"
        }
        
        with open(pack_dir / "pack.yaml", "w") as f:
            yaml.dump(manifest_data, f)
        
        # Create referenced files
        (pack_dir / "gl.yaml").write_text("# Pipeline")
        (pack_dir / "agents").mkdir()
        (pack_dir / "agents" / "calc.py").write_text("# Agent")
        (pack_dir / "data").mkdir()
        (pack_dir / "data" / "test.csv").write_text("col1,col2")
        (pack_dir / "tests").mkdir()
        (pack_dir / "tests" / "test_1.py").write_text("# Test")
        (pack_dir / "CARD.md").write_text("# Card")
        
        # Load should succeed
        manifest = load_manifest(str(pack_dir))
        assert manifest.name == "test-pack"
        assert manifest.version == "1.0.0"
    
    def test_missing_manifest(self, tmp_path):
        """Test that missing pack.yaml raises error."""
        pack_dir = tmp_path / "empty-pack"
        pack_dir.mkdir()
        
        with pytest.raises(FileNotFoundError) as exc_info:
            load_manifest(str(pack_dir))
        
        assert "Missing pack.yaml" in str(exc_info.value)
    
    def test_missing_referenced_files(self, tmp_path):
        """Test that missing referenced files raise errors."""
        pack_dir = tmp_path / "incomplete-pack"
        pack_dir.mkdir()
        
        # Create manifest referencing non-existent files
        manifest_data = {
            "name": "incomplete-pack",
            "version": "1.0.0",
            "license": "MIT",
            "compat": {
                "greenlang": ">=0.1.0",
                "python": ">=3.10"
            },
            "contents": {
                "pipelines": ["missing.yaml"],
                "agents": ["agents/missing.py"]
            },
            "card": "MISSING.md"
        }
        
        with open(pack_dir / "pack.yaml", "w") as f:
            yaml.dump(manifest_data, f)
        
        # Load should fail with clear error message
        with pytest.raises(FileNotFoundError) as exc_info:
            load_manifest(str(pack_dir))
        
        error_msg = str(exc_info.value)
        assert "Missing files" in error_msg
        assert "Pipeline: missing.yaml" in error_msg
        assert "Agent: agents/missing.py" in error_msg
        assert "Card: MISSING.md" in error_msg
    
    def test_validate_pack_success(self, tmp_path):
        """Test validate_pack returns success for valid pack."""
        pack_dir = tmp_path / "valid-pack"
        pack_dir.mkdir()
        
        # Create complete pack
        manifest_data = {
            "name": "valid-pack",
            "version": "1.0.0",
            "license": "MIT",
            "compat": {
                "greenlang": ">=0.1.0",
                "python": ">=3.10"
            },
            "contents": {
                "pipelines": ["gl.yaml"]
            },
            "card": "CARD.md"
        }
        
        with open(pack_dir / "pack.yaml", "w") as f:
            yaml.dump(manifest_data, f)
        
        (pack_dir / "gl.yaml").write_text("# Pipeline")
        (pack_dir / "CARD.md").write_text("# Card")
        
        # Validate should succeed
        is_valid, errors = validate_pack(str(pack_dir))
        assert is_valid is True
        assert errors == []
    
    def test_validate_pack_failure(self, tmp_path):
        """Test validate_pack returns errors for invalid pack."""
        pack_dir = tmp_path / "invalid-pack"
        pack_dir.mkdir()
        
        # Create pack with missing files
        manifest_data = {
            "name": "invalid-pack",
            "version": "1.0.0",
            "license": "MIT",
            "compat": {
                "greenlang": ">=0.1.0",
                "python": ">=3.10"
            },
            "contents": {
                "pipelines": ["missing.yaml"]
            },
            "card": "CARD.md"
        }
        
        with open(pack_dir / "pack.yaml", "w") as f:
            yaml.dump(manifest_data, f)
        
        # Only create card, not pipeline
        (pack_dir / "CARD.md").write_text("# Card")
        
        # Validate should fail
        is_valid, errors = validate_pack(str(pack_dir))
        assert is_valid is False
        assert len(errors) > 0
        assert "Missing files" in errors[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])