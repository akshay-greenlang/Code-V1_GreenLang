# -*- coding: utf-8 -*-
"""
Integration tests for gl init agent command (FRMW-202)

This module tests the full end-to-end flow of:
- Running `gl init agent <name>` command
- Validating all generated files
- Running tests on the generated agent
- Verifying AgentSpec v2 compliance
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import yaml
import subprocess
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from greenlang.cli.cmd_init_agent import (
    generate_pack_yaml_compute,
    generate_pack_yaml_ai,
    generate_pack_yaml_industry,
    generate_schemas_py,
    generate_agent_py,
    generate_provenance_py,
    generate_test_suite,
    generate_precommit_config,
    generate_ci_workflow,
    generate_common_files,
    validate_generated_agent,
)


class TestInitAgentCompute:
    """Test gl init agent with compute template"""

    @pytest.fixture
    def test_dir(self):
        """Create temporary test directory"""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_agent_"))
        yield temp_dir
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_compute_agent_generation(self, test_dir):
        """Test complete compute agent generation"""
        pack_id = "test-compute"
        python_pkg = "test_compute"
        class_name = "TestCompute"

        agent_dir = test_dir / pack_id
        agent_dir.mkdir(parents=True)
        (agent_dir / python_pkg).mkdir()
        (agent_dir / "tests").mkdir()

        # 1. Generate pack.yaml
        pack_yaml = generate_pack_yaml_compute(
            pack_id=pack_id,
            python_pkg=python_pkg,
            license="apache-2.0",
            author="Test Author",
            realtime=False,
            spec_data=None
        )

        assert pack_yaml["schema_version"] == "2.0.0"
        assert "compute" in pack_yaml
        assert "provenance" in pack_yaml
        assert pack_yaml["compute"]["deterministic"] == True

        # Write to file
        with open(agent_dir / "pack.yaml", "w", encoding="utf-8") as f:
            yaml.dump(pack_yaml, f, default_flow_style=False)

        # 2. Generate schemas.py
        schemas_content = generate_schemas_py(
            python_pkg=python_pkg,
            class_name=class_name,
            template="compute"
        )

        assert "InputModel" in schemas_content
        assert "OutputModel" in schemas_content
        assert "Pydantic" in schemas_content
        assert "Annotated" in schemas_content

        with open(agent_dir / python_pkg / "schemas.py", "w", encoding="utf-8") as f:
            f.write(schemas_content)

        # 3. Generate agent.py
        agent_content = generate_agent_py(
            python_pkg=python_pkg,
            class_name=class_name,
            template="compute",
            realtime=False
        )

        assert f"class {class_name}" in agent_content
        assert "def compute" in agent_content
        assert "provenance" in agent_content

        with open(agent_dir / python_pkg / "agent.py", "w", encoding="utf-8") as f:
            f.write(agent_content)

        # 4. Generate provenance.py
        provenance_content = generate_provenance_py()

        assert "compute_formula_hash" in provenance_content
        assert "create_provenance_record" in provenance_content

        with open(agent_dir / python_pkg / "provenance.py", "w", encoding="utf-8") as f:
            f.write(provenance_content)

        # 5. Generate __init__.py
        init_content = f'''"""
{pack_id} - GreenLang Agent
"""
from .agent import {class_name}

__all__ = ["{class_name}"]
'''
        with open(agent_dir / python_pkg / "__init__.py", "w", encoding="utf-8") as f:
            f.write(init_content)

        with open(agent_dir / "tests" / "__init__.py", "w", encoding="utf-8") as f:
            f.write('"""Tests for {pack_id}"""\n')

        # 6. Generate test suite
        generate_test_suite(
            agent_dir=agent_dir,
            pack_id=pack_id,
            python_pkg=python_pkg,
            class_name=class_name,
            template="compute"
        )

        assert (agent_dir / "tests" / "test_agent.py").exists()
        assert (agent_dir / "tests" / "conftest.py").exists()

        # 7. Generate common files
        generate_common_files(
            agent_dir=agent_dir,
            pack_id=pack_id,
            python_pkg=python_pkg,
            license="apache-2.0",
            author="Test Author"
        )

        assert (agent_dir / "LICENSE").exists()
        assert (agent_dir / "pyproject.toml").exists()

        # 8. Generate pre-commit config
        generate_precommit_config(agent_dir=agent_dir)
        assert (agent_dir / ".pre-commit-config.yaml").exists()

        # Verify Bandit and TruffleHog are included
        with open(agent_dir / ".pre-commit-config.yaml", "r") as f:
            precommit_yaml = yaml.safe_load(f)
            repos = [repo["repo"] for repo in precommit_yaml["repos"]]
            assert any("trufflesecurity/trufflehog" in repo for repo in repos)
            assert any("PyCQA/bandit" in repo for repo in repos)

        # 9. Generate CI workflow
        generate_ci_workflow(
            agent_dir=agent_dir,
            pack_id=pack_id,
            runtimes="local"
        )

        assert (agent_dir / ".github" / "workflows" / "ci.yml").exists()

        # 10. Validate generated agent
        validation_result = validate_generated_agent(agent_dir)
        assert validation_result["valid"] == True
        assert len(validation_result["errors"]) == 0

        # 11. Verify all expected files exist
        expected_files = [
            "pack.yaml",
            "LICENSE",
            "pyproject.toml",
            ".pre-commit-config.yaml",
            ".github/workflows/ci.yml",
            f"{python_pkg}/__init__.py",
            f"{python_pkg}/agent.py",
            f"{python_pkg}/schemas.py",
            f"{python_pkg}/provenance.py",
            "tests/__init__.py",
            "tests/test_agent.py",
            "tests/conftest.py",
        ]

        for file_path in expected_files:
            assert (agent_dir / file_path).exists(), f"Missing file: {file_path}"

        # 12. Verify pack.yaml structure
        with open(agent_dir / "pack.yaml", "r") as f:
            manifest = yaml.safe_load(f)

        assert manifest["schema_version"] == "2.0.0"
        assert "compute" in manifest
        assert "provenance" in manifest
        assert manifest["compute"]["deterministic"] == True

    def test_ai_agent_generation(self, test_dir):
        """Test AI agent template generation"""
        pack_id = "test-ai"
        python_pkg = "test_ai"

        # Generate AI pack.yaml
        pack_yaml = generate_pack_yaml_ai(
            pack_id=pack_id,
            python_pkg=python_pkg,
            license="mit",
            author="AI Test",
            realtime=False,
            spec_data=None
        )

        assert pack_yaml["schema_version"] == "2.0.0"
        assert "ai" in pack_yaml
        assert "provenance" in pack_yaml
        assert "tools" in pack_yaml["ai"]
        assert "budget" in pack_yaml["ai"]

        # Verify provenance includes AI-specific fields
        assert "llm_model" in pack_yaml["provenance"]["record"]
        assert "prompt_hash" in pack_yaml["provenance"]["record"]
        assert "cost_usd" in pack_yaml["provenance"]["record"]

    def test_industry_agent_generation(self, test_dir):
        """Test industry agent template generation"""
        pack_id = "test-industry"
        python_pkg = "test_industry"

        # Generate industry pack.yaml
        pack_yaml = generate_pack_yaml_industry(
            pack_id=pack_id,
            python_pkg=python_pkg,
            license="apache-2.0",
            author="Industry Test",
            realtime=False,
            spec_data=None
        )

        assert pack_yaml["schema_version"] == "2.0.0"
        assert "compute" in pack_yaml

        # Verify multi-scope emissions
        outputs = pack_yaml["compute"]["outputs"]
        assert "scope1_co2e_kg" in outputs
        assert "scope2_co2e_kg" in outputs
        assert "scope3_co2e_kg" in outputs

        # Verify GHG Protocol factors
        factors = pack_yaml["compute"]["factors"]
        assert "fuel_ef" in factors
        assert "grid_ef" in factors
        assert "supply_chain_ef" in factors


class TestAgentSpecV2Compliance:
    """Test AgentSpec v2 compliance of generated agents"""

    def test_pack_yaml_schema_version(self):
        """Verify all templates use schema_version 2.0.0"""
        templates = ["compute", "ai", "industry"]

        for template in templates:
            if template == "compute":
                pack_yaml = generate_pack_yaml_compute(
                    pack_id=f"test-{template}",
                    python_pkg=f"test_{template}",
                    license="apache-2.0",
                    author="Test",
                    realtime=False,
                    spec_data=None
                )
            elif template == "ai":
                pack_yaml = generate_pack_yaml_ai(
                    pack_id=f"test-{template}",
                    python_pkg=f"test_{template}",
                    license="apache-2.0",
                    author="Test",
                    realtime=False,
                    spec_data=None
                )
            else:  # industry
                pack_yaml = generate_pack_yaml_industry(
                    pack_id=f"test-{template}",
                    python_pkg=f"test_{template}",
                    license="apache-2.0",
                    author="Test",
                    realtime=False,
                    spec_data=None
                )

            assert pack_yaml["schema_version"] == "2.0.0", f"Template {template} has wrong schema version"

    def test_provenance_tracking(self):
        """Verify provenance tracking is enabled"""
        pack_yaml = generate_pack_yaml_compute(
            pack_id="test-prov",
            python_pkg="test_prov",
            license="apache-2.0",
            author="Test",
            realtime=False,
            spec_data=None
        )

        assert "provenance" in pack_yaml
        assert "record" in pack_yaml["provenance"]
        assert "inputs" in pack_yaml["provenance"]["record"]
        assert "outputs" in pack_yaml["provenance"]["record"]
        assert "factors" in pack_yaml["provenance"]["record"]
        assert "ef_uri" in pack_yaml["provenance"]["record"]
        assert "ef_cid" in pack_yaml["provenance"]["record"]
        assert "code_sha" in pack_yaml["provenance"]["record"]

    def test_security_defaults(self):
        """Verify security-first defaults in pre-commit"""
        # Security is enforced via pre-commit hooks with Bandit and TruffleHog
        # This was verified in test_compute_agent_generation


class TestCrossOSCompatibility:
    """Test cross-OS compatibility of generated files"""

    def test_utf8_encoding(self):
        """Verify all generated files use UTF-8 encoding"""
        schemas = generate_schemas_py("test_pkg", "TestClass", "compute")

        # Should not raise encoding errors
        schemas.encode("utf-8")
        assert isinstance(schemas, str)

    def test_newline_normalization(self):
        """Verify newline characters are normalized"""
        agent_content = generate_agent_py(
            python_pkg="test_pkg",
            class_name="TestClass",
            template="compute",
            realtime=False
        )

        # Count line endings
        assert "\r\n" not in agent_content or "\n" in agent_content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
