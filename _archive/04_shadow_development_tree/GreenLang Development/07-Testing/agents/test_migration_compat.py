"""
Test agent migration compatibility.

This module tests that agent migrations work correctly and that
deprecated agents still function with appropriate warnings.
"""

import pytest
import warnings
from unittest.mock import Mock, patch

from greenlang.agents.registry import (
    AgentRegistry,
    AgentStatus,
    ExecutionMode,
    get_agent_info,
    create_agent,
    list_agents,
    get_migration_path,
)


class TestAgentRegistry:
    """Test the agent registry functionality."""

    def test_list_agents(self):
        """Test listing available agents."""
        agents = list_agents(include_deprecated=False)
        assert "FuelAgent" in agents
        assert "CarbonAgent" in agents
        assert "BoilerReplacementAgent" in agents
        assert len(agents) >= 18  # We have at least 18 canonical agents

    def test_get_agent_info(self):
        """Test getting agent information."""
        info = get_agent_info("FuelAgent")
        assert info is not None
        assert info.name == "FuelAgent"
        assert info.version == "2.0.0"
        assert info.status == AgentStatus.ACTIVE
        assert ExecutionMode.SYNC in info.supported_modes
        assert ExecutionMode.ASYNC in info.supported_modes
        assert info.ai_enabled is True

    def test_canonical_import(self):
        """Test canonical import path."""
        info = get_agent_info("FuelAgent")
        assert info.canonical_import == "from greenlang.agents.fuel_agent_ai_v2 import FuelAgent"

    def test_alias_resolution(self):
        """Test that aliases resolve correctly."""
        info1 = get_agent_info("FuelAgent")
        info2 = get_agent_info("FuelAgentAI")  # Alias
        info3 = get_agent_info("FuelAgentAsync")  # Alias

        # All should resolve to the same agent
        assert info1.name == info2.name == info3.name

    def test_deprecation_check(self):
        """Test deprecation checking."""
        from greenlang.agents.registry import check_deprecation

        # Active agent should not be deprecated
        assert check_deprecation("FuelAgent") is None

        # Legacy agent should be deprecated
        msg = check_deprecation("FuelAgent_legacy")
        assert msg is not None
        assert "deprecated" in msg.lower()

    def test_migration_path(self):
        """Test getting migration paths."""
        # Alias should map to canonical name
        assert get_migration_path("FuelAgentAI") == "FuelAgent"
        assert get_migration_path("BoilerReplacementAgentV3") == "BoilerReplacementAgent"

        # Unknown agent should return None
        assert get_migration_path("NonExistentAgent") is None

    @patch('importlib.import_module')
    def test_create_agent_mock(self, mock_import):
        """Test agent creation with mocking."""
        # Mock the agent class
        mock_module = Mock()
        mock_agent_class = Mock()
        mock_module.FuelAgent = mock_agent_class
        mock_import.return_value = mock_module

        registry = AgentRegistry()
        config = {"test": "config"}

        # Create agent
        agent = registry.create_agent("FuelAgent", config=config, mode=ExecutionMode.ASYNC)

        # Verify agent was created with correct parameters
        mock_agent_class.assert_called_once()
        call_args = mock_agent_class.call_args
        assert call_args[0][0] == config  # Config passed as first arg
        assert call_args[1]["mode"] == "async"
        assert call_args[1]["ai_enabled"] is True

    def test_experimental_agent_blocking(self):
        """Test that experimental agents are blocked by default."""
        registry = AgentRegistry()

        # Mock an experimental agent
        with patch.object(registry, 'get_agent_info') as mock_get_info:
            mock_info = Mock()
            mock_info.status = AgentStatus.EXPERIMENTAL
            mock_info.name = "TestAgent"
            mock_get_info.return_value = mock_info

            with pytest.raises(ValueError, match="experimental"):
                registry.create_agent("TestAgent")

    def test_unsupported_mode_error(self):
        """Test error when requesting unsupported execution mode."""
        registry = AgentRegistry()

        # Mock agent with limited mode support
        with patch.object(registry, 'get_agent_info') as mock_get_info:
            mock_info = Mock()
            mock_info.status = AgentStatus.ACTIVE
            mock_info.supported_modes = [ExecutionMode.SYNC]
            mock_info.name = "TestAgent"
            mock_info.module_path = "test.module"
            mock_info.canonical_class = "TestAgent"
            mock_get_info.return_value = mock_info

            with pytest.raises(ValueError, match="does not support mode"):
                registry.create_agent("TestAgent", mode=ExecutionMode.ASYNC)

    def test_registry_export(self):
        """Test registry export for documentation."""
        from greenlang.agents.registry import export_registry

        registry_data = export_registry()

        assert isinstance(registry_data, dict)
        assert "FuelAgent" in registry_data

        fuel_data = registry_data["FuelAgent"]
        assert fuel_data["version"] == "2.0.0"
        assert fuel_data["status"] == "active"
        assert fuel_data["ai_enabled"] is True
        assert "sync" in fuel_data["modes"]
        assert "async" in fuel_data["modes"]


class TestMigrationUtilities:
    """Test migration utilities."""

    def test_import_migration_detection(self):
        """Test detection of deprecated imports."""
        from greenlang.agents.migration import ImportMigrator

        migrator = ImportMigrator()

        # Test code with deprecated imports
        test_code = """
from greenlang.agents.fuel_agent import FuelAgent
from greenlang.agents.carbon_agent import CarbonAgent
from greenlang.agents.boiler_replacement_agent_ai import BoilerReplacementAgent
"""

        # Check that deprecated imports are detected
        for old_import in [
            "from greenlang.agents.fuel_agent import",
            "from greenlang.agents.carbon_agent import",
            "from greenlang.agents.boiler_replacement_agent_ai import"
        ]:
            assert old_import in test_code
            assert old_import in migrator.IMPORT_MAPPINGS

    def test_class_name_migration(self):
        """Test class name mapping."""
        from greenlang.agents.migration import ImportMigrator

        migrator = ImportMigrator()

        # Test mappings exist for common variants
        assert "FuelAgentAsync" in migrator.CLASS_MAPPINGS
        assert migrator.CLASS_MAPPINGS["FuelAgentAsync"] == "FuelAgent"

        assert "BoilerReplacementAgentV3" in migrator.CLASS_MAPPINGS
        assert migrator.CLASS_MAPPINGS["BoilerReplacementAgentV3"] == "BoilerReplacementAgent"

    def test_migration_dry_run(self, tmp_path):
        """Test migration in dry run mode."""
        from greenlang.agents.migration import ImportMigrator

        # Create test file with deprecated imports
        test_file = tmp_path / "test_code.py"
        test_file.write_text("""
from greenlang.agents.fuel_agent import FuelAgent
from greenlang.agents.carbon_agent import CarbonAgent

def main():
    agent = FuelAgentAsync(config)
    carbon = CarbonAgentBase(config)
""")

        migrator = ImportMigrator()
        changes = migrator.migrate_file(str(test_file), dry_run=True)

        # Should detect changes but not modify file
        assert len(changes) > 0
        assert "from greenlang.agents.fuel_agent import" in test_file.read_text()

    def test_migration_apply(self, tmp_path):
        """Test applying migration changes."""
        from greenlang.agents.migration import ImportMigrator

        # Create test file
        test_file = tmp_path / "test_code.py"
        original_content = """from greenlang.agents.fuel_agent import FuelAgent

agent = FuelAgentAsync(config)
"""
        test_file.write_text(original_content)

        migrator = ImportMigrator()
        changes = migrator.migrate_file(str(test_file), dry_run=False)

        # Should modify file
        assert len(changes) > 0
        new_content = test_file.read_text()
        assert "from greenlang.agents.fuel_agent_ai_v2 import" in new_content
        assert "FuelAgent(" in new_content  # Class name updated
        assert "WARNING: This file has been automatically migrated" in new_content


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_deprecation_warnings(self):
        """Test that deprecated agents show warnings."""
        registry = AgentRegistry()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock deprecated agent info
            with patch.object(registry, 'get_agent_info') as mock_get_info:
                mock_info = Mock()
                mock_info.status = AgentStatus.DEPRECATED
                mock_info.is_deprecated = True
                mock_info.deprecated_in = "1.8.0"
                mock_info.removed_in = "2.0.0"
                mock_info.migration_guide = "Use new version"
                mock_info.name = "OldAgent"
                mock_info.module_path = "test"
                mock_info.canonical_class = "OldAgent"
                mock_info.supported_modes = None
                mock_get_info.return_value = mock_info

                # Mock the import
                with patch('importlib.import_module'):
                    try:
                        registry.create_agent("OldAgent")
                    except:
                        pass  # We expect this to fail, just checking warning

                # Should have deprecation warning
                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "deprecated" in str(w[0].message)

    def test_suppress_warnings_config(self):
        """Test suppressing deprecation warnings."""
        from greenlang.agents.registry import AgentRegistryConfig

        config = AgentRegistryConfig(suppress_deprecation_warnings=True)
        registry = AgentRegistry(config=config)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock deprecated agent
            with patch.object(registry, 'get_agent_info') as mock_get_info:
                mock_info = Mock()
                mock_info.status = AgentStatus.DEPRECATED
                mock_info.is_deprecated = True
                mock_info.deprecated_in = "1.8.0"
                mock_info.removed_in = "2.0.0"
                mock_info.migration_guide = "Use new version"
                mock_info.name = "OldAgent"
                mock_get_info.return_value = mock_info

                # Try to create (will fail but should not warn)
                try:
                    registry.create_agent("OldAgent")
                except:
                    pass

                # Should NOT have warning (suppressed)
                assert len(w) == 0


def test_integration_canonical_agents():
    """Integration test for canonical agents."""
    # Test that we can import and use canonical agents
    canonical_imports = [
        ("greenlang.agents.fuel_agent_ai_v2", "FuelAgent"),
        ("greenlang.agents.carbon_agent_ai", "CarbonAgent"),
        ("greenlang.agents.grid_factor_agent_ai", "GridFactorAgent"),
    ]

    for module_path, class_name in canonical_imports:
        try:
            module = __import__(module_path, fromlist=[class_name])
            agent_class = getattr(module, class_name, None)
            assert agent_class is not None, f"Could not find {class_name} in {module_path}"
        except ImportError as e:
            # Some agents may not be available in test environment
            pytest.skip(f"Could not import {module_path}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])