# -*- coding: utf-8 -*-
"""
Comprehensive tests for greenlang.__init__ module
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the greenlang directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "greenlang"))


class TestGreenLangInit:
    """Test the main GreenLang module initialization"""

    def test_module_metadata(self):
        """Test module metadata is correctly defined"""
        import greenlang

        assert hasattr(greenlang, "__author__")
        assert hasattr(greenlang, "__email__")
        assert hasattr(greenlang, "__license__")
        assert hasattr(greenlang, "__version__")

        assert greenlang.__author__ == "GreenLang Team"
        assert greenlang.__email__ == "team@greenlang.in"
        assert greenlang.__license__ == "MIT"
        assert isinstance(greenlang.__version__, str)

    def test_version_import(self):
        """Test that version is imported correctly"""
        import greenlang

        # Version should be a non-empty string
        assert isinstance(greenlang.__version__, str)
        assert len(greenlang.__version__) > 0

    def test_core_imports_available(self):
        """Test that all core components can be imported"""
        import greenlang

        # Core SDK abstractions
        assert hasattr(greenlang, "Agent")
        assert hasattr(greenlang, "Pipeline")
        assert hasattr(greenlang, "Connector")
        assert hasattr(greenlang, "Dataset")
        assert hasattr(greenlang, "Report")

        # Context and artifacts
        assert hasattr(greenlang, "Context")
        assert hasattr(greenlang, "Artifact")

        # Pack system
        assert hasattr(greenlang, "PackRegistry")
        assert hasattr(greenlang, "PackLoader")

        # Runtime
        assert hasattr(greenlang, "Executor")

        # Policy
        assert hasattr(greenlang, "PolicyEnforcer")

    def test_all_exports_defined(self):
        """Test that __all__ contains all expected exports"""
        import greenlang

        expected_exports = [
            # Core SDK abstractions
            "Agent",
            "Pipeline",
            "Connector",
            "Dataset",
            "Report",
            "Context",
            "Artifact",
            # Pack system
            "PackRegistry",
            "PackLoader",
            # Runtime
            "Executor",
            # Policy
            "PolicyEnforcer",
        ]

        assert hasattr(greenlang, "__all__")
        assert isinstance(greenlang.__all__, list)

        for export in expected_exports:
            assert export in greenlang.__all__, f"{export} not in __all__"

    def test_agent_import(self):
        """Test Agent class import and basic functionality"""
        from greenlang import Agent

        # Should be a class
        assert callable(Agent)

        # Should be abstract
        from abc import ABC
        assert issubclass(Agent, ABC)

    def test_pipeline_import(self):
        """Test Pipeline class import and basic functionality"""
        from greenlang import Pipeline

        # Should be a class
        assert callable(Pipeline)

        # Should be abstract
        from abc import ABC
        assert issubclass(Pipeline, ABC)

    def test_connector_import(self):
        """Test Connector class import and basic functionality"""
        from greenlang import Connector

        # Should be a class
        assert callable(Connector)

        # Should be abstract
        from abc import ABC
        assert issubclass(Connector, ABC)

    def test_dataset_import(self):
        """Test Dataset class import and basic functionality"""
        from greenlang import Dataset

        # Should be a class
        assert callable(Dataset)

        # Should be abstract
        from abc import ABC
        assert issubclass(Dataset, ABC)

    def test_report_import(self):
        """Test Report class import and basic functionality"""
        from greenlang import Report

        # Should be a class
        assert callable(Report)

        # Should be abstract
        from abc import ABC
        assert issubclass(Report, ABC)

    def test_context_import(self):
        """Test Context class import and basic functionality"""
        from greenlang import Context

        # Should be a class that can be instantiated
        assert callable(Context)

        # Should be able to create instance
        context = Context()
        assert context is not None

    def test_artifact_import(self):
        """Test Artifact class import and basic functionality"""
        from greenlang import Artifact

        # Should be a class
        assert callable(Artifact)

    def test_pack_registry_import(self):
        """Test PackRegistry class import and basic functionality"""
        from greenlang import PackRegistry

        # Should be a class that can be instantiated
        assert callable(PackRegistry)

    def test_pack_loader_import(self):
        """Test PackLoader class import and basic functionality"""
        from greenlang import PackLoader

        # Should be a class that can be instantiated
        assert callable(PackLoader)

    def test_executor_import(self):
        """Test Executor class import and basic functionality"""
        from greenlang import Executor

        # Should be a class that can be instantiated
        assert callable(Executor)

    def test_policy_enforcer_import(self):
        """Test PolicyEnforcer class import and basic functionality"""
        from greenlang import PolicyEnforcer

        # Should be a class
        assert callable(PolicyEnforcer)

    def test_multiple_imports_same_session(self):
        """Test that multiple imports in same session work correctly"""
        # First import
        from greenlang import Agent as Agent1
        from greenlang import Pipeline as Pipeline1

        # Second import
        from greenlang import Agent as Agent2
        from greenlang import Pipeline as Pipeline2

        # Should be same objects
        assert Agent1 is Agent2
        assert Pipeline1 is Pipeline2

    def test_import_all_at_once(self):
        """Test importing all components at once"""
        from greenlang import (
            Agent, Pipeline, Connector, Dataset, Report,
            Context, Artifact, PackRegistry, PackLoader,
            Executor, PolicyEnforcer
        )

        # All should be available
        components = [
            Agent, Pipeline, Connector, Dataset, Report,
            Context, Artifact, PackRegistry, PackLoader,
            Executor, PolicyEnforcer
        ]

        for component in components:
            assert component is not None
            assert callable(component)

    def test_star_import(self):
        """Test star import functionality"""
        # Create a namespace for star import
        import types
        namespace = types.ModuleType("test_namespace")

        # Simulate star import
        exec("from greenlang import *", namespace.__dict__)

        # Check that all expected exports are available
        for export in [
            "Agent", "Pipeline", "Connector", "Dataset", "Report",
            "Context", "Artifact", "PackRegistry", "PackLoader",
            "Executor", "PolicyEnforcer"
        ]:
            assert hasattr(namespace, export), f"{export} not available in star import"

    @patch('greenlang.sdk.base')
    def test_import_error_handling_base(self, mock_base):
        """Test handling of import errors from base module"""
        mock_base.side_effect = ImportError("Mock import error")

        with pytest.raises(ImportError):
            import importlib
            # Force reload to trigger import
            if 'greenlang' in sys.modules:
                importlib.reload(sys.modules['greenlang'])
            else:
                import greenlang

    @patch('greenlang.sdk.context')
    def test_import_error_handling_context(self, mock_context):
        """Test handling of import errors from context module"""
        mock_context.side_effect = ImportError("Mock context import error")

        with pytest.raises(ImportError):
            import importlib
            # Force reload to trigger import
            if 'greenlang' in sys.modules:
                importlib.reload(sys.modules['greenlang'])
            else:
                import greenlang

    def test_module_docstring(self):
        """Test that module has proper docstring"""
        import greenlang

        assert greenlang.__doc__ is not None
        assert "GreenLang" in greenlang.__doc__
        assert "Infrastructure for Climate Intelligence" in greenlang.__doc__

    def test_no_unexpected_attributes(self):
        """Test that module doesn't expose unexpected attributes"""
        import greenlang

        # Get all public attributes
        public_attrs = [attr for attr in dir(greenlang) if not attr.startswith('_')]

        expected_attrs = set([
            # Metadata
            "__author__", "__email__", "__license__", "__version__",
            # Core exports from __all__
            "Agent", "Pipeline", "Connector", "Dataset", "Report",
            "Context", "Artifact", "PackRegistry", "PackLoader",
            "Executor", "PolicyEnforcer"
        ])

        # Remove private attributes from check
        public_attrs = [attr for attr in public_attrs if not attr.startswith('__')]

        # All public attributes should be in expected list
        for attr in public_attrs:
            assert attr in greenlang.__all__, f"Unexpected public attribute: {attr}"

    def test_component_types(self):
        """Test that imported components have correct types"""
        from greenlang import (
            Agent, Pipeline, Connector, Dataset, Report,
            Context, Artifact, PackRegistry, PackLoader,
            Executor, PolicyEnforcer
        )

        # Abstract base classes
        from abc import ABC
        assert issubclass(Agent, ABC)
        assert issubclass(Pipeline, ABC)
        assert issubclass(Connector, ABC)
        assert issubclass(Dataset, ABC)
        assert issubclass(Report, ABC)

        # Concrete classes
        assert not issubclass(Context, ABC)
        assert not issubclass(PackRegistry, ABC)
        assert not issubclass(PackLoader, ABC)
        assert not issubclass(Executor, ABC)

    def test_circular_import_protection(self):
        """Test that module handles circular imports gracefully"""
        # This should not cause infinite recursion
        import greenlang
        from greenlang import Agent
        from greenlang.sdk.base import Agent as BaseAgent

        # Should be the same class
        assert Agent is BaseAgent

    def test_module_level_constants(self):
        """Test module-level constants are properly defined"""
        import greenlang

        # Check metadata constants
        metadata_fields = ["__author__", "__email__", "__license__"]
        for field in metadata_fields:
            assert hasattr(greenlang, field)
            assert isinstance(getattr(greenlang, field), str)
            assert len(getattr(greenlang, field)) > 0

    def test_version_consistency(self):
        """Test version consistency across imports"""
        import greenlang
        from greenlang._version import __version__ as version_module_version

        # Should be the same version
        assert greenlang.__version__ == version_module_version

    def test_import_performance(self):
        """Test that imports don't take too long"""
        import time
        import importlib

        # Remove module if already imported
        modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith('greenlang')]
        for mod in modules_to_remove:
            del sys.modules[mod]

        # Time the import
        start_time = time.time()
        import greenlang
        end_time = time.time()

        # Should take less than 5 seconds
        import_time = end_time - start_time
        assert import_time < 5.0, f"Import took {import_time:.2f}s, which is too long"

    def test_component_instantiation_basic(self):
        """Test that concrete components can be instantiated"""
        from greenlang import Context, PackRegistry, PackLoader, Executor

        # These should be able to instantiate without errors
        context = Context()
        assert context is not None

        # Pack components might need mocking for full instantiation
        with patch('greenlang.packs.registry.Path'):
            registry = PackRegistry()
            assert registry is not None

        with patch('greenlang.packs.loader.Path'):
            loader = PackLoader()
            assert loader is not None

        executor = Executor()
        assert executor is not None

    @pytest.mark.parametrize("component_name", [
        "Agent", "Pipeline", "Connector", "Dataset", "Report",
        "Context", "Artifact", "PackRegistry", "PackLoader",
        "Executor", "PolicyEnforcer"
    ])
    def test_individual_component_import(self, component_name):
        """Test that each component can be imported individually"""
        module = __import__('greenlang', fromlist=[component_name])
        component = getattr(module, component_name)

        assert component is not None
        assert callable(component) or hasattr(component, '__class__')

    def test_module_reload_safety(self):
        """Test that module can be safely reloaded"""
        import greenlang
        import importlib

        # Get initial components
        initial_agent = greenlang.Agent
        initial_version = greenlang.__version__

        # Reload module
        importlib.reload(greenlang)

        # Components should still be available
        assert hasattr(greenlang, 'Agent')
        assert hasattr(greenlang, '__version__')

        # Version should be consistent
        assert greenlang.__version__ == initial_version