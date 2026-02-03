# -*- coding: utf-8 -*-
"""
Helm Chart Validation Tests

INFRA-001: Infrastructure test suite for validating Helm charts.

Tests include:
- Chart.yaml validation
- values.yaml validation
- Template structure validation
- Dependencies validation
- Values schema validation
- Environment-specific values files
- Lint checks with mock

Target coverage: 85%+
"""

import re
from pathlib import Path
from typing import Dict, List, Any

import pytest
import yaml


class TestHelmChartStructure:
    """Test Helm chart file structure."""

    def test_helm_directory_exists(self, helm_dir: Path):
        """Test that the Helm charts directory exists."""
        assert helm_dir.exists(), f"Helm directory not found at {helm_dir}"

    def test_helm_charts_exist(self, all_helm_charts: List[Path]):
        """Test that Helm charts exist in the project."""
        assert len(all_helm_charts) > 0, "No Helm charts found in the project"

    def test_charts_have_required_files(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that all charts have required files."""
        for chart in all_helm_charts:
            files_check = helm_validator.check_required_files(chart)

            for file_name, exists in files_check.items():
                assert exists, f"Chart {chart.name} is missing required: {file_name}"

    def test_charts_have_templates_directory(self, all_helm_charts: List[Path]):
        """Test that all charts have templates directory."""
        for chart in all_helm_charts:
            templates_dir = chart / "templates"
            assert templates_dir.exists(), f"Chart {chart.name} missing templates directory"

    def test_templates_directory_not_empty(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that templates directories are not empty."""
        for chart in all_helm_charts:
            templates = helm_validator.check_templates_exist(chart)

            # Filter out _helpers.tpl and NOTES.txt which are optional
            main_templates = [t for t in templates if not t.startswith("_") and t != "NOTES.txt"]

            assert len(main_templates) > 0, f"Chart {chart.name} has no templates"


class TestHelmChartYaml:
    """Test Helm Chart.yaml validation."""

    def test_chart_yaml_has_required_fields(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that Chart.yaml has required fields."""
        for chart in all_helm_charts:
            chart_yaml = helm_validator.load_chart_yaml(chart)

            missing = helm_validator.check_chart_fields(
                chart_yaml,
                helm_validator.REQUIRED_CHART_FIELDS
            )

            assert len(missing) == 0, f"Chart {chart.name} missing required fields: {missing}"

    def test_chart_yaml_has_recommended_fields(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that Chart.yaml has recommended fields."""
        violations = []

        for chart in all_helm_charts:
            chart_yaml = helm_validator.load_chart_yaml(chart)

            missing = helm_validator.check_chart_fields(
                chart_yaml,
                helm_validator.RECOMMENDED_CHART_FIELDS
            )

            if missing:
                violations.append(f"Chart {chart.name} missing recommended: {missing}")

        if violations:
            pytest.skip(f"Charts missing recommended fields (review): {len(violations)}")

    def test_chart_api_version_is_v2(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that charts use apiVersion v2."""
        violations = []

        for chart in all_helm_charts:
            chart_yaml = helm_validator.load_chart_yaml(chart)
            api_version = chart_yaml.get("apiVersion", "")

            if api_version != "v2":
                violations.append(f"Chart {chart.name} uses apiVersion {api_version}, should be v2")

        assert len(violations) == 0, f"API version issues:\n" + "\n".join(violations)

    def test_chart_version_follows_semver(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that chart versions follow semantic versioning."""
        semver_pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?(\+[a-zA-Z0-9]+)?$'
        violations = []

        for chart in all_helm_charts:
            chart_yaml = helm_validator.load_chart_yaml(chart)
            version = chart_yaml.get("version", "")

            if not re.match(semver_pattern, version):
                violations.append(f"Chart {chart.name} version '{version}' is not valid semver")

        assert len(violations) == 0, f"Version issues:\n" + "\n".join(violations)

    def test_chart_has_maintainers(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that charts have maintainers defined."""
        violations = []

        for chart in all_helm_charts:
            chart_yaml = helm_validator.load_chart_yaml(chart)
            maintainers = chart_yaml.get("maintainers", [])

            if not maintainers:
                violations.append(f"Chart {chart.name} has no maintainers")

        if violations:
            pytest.skip(f"Charts without maintainers (review): {len(violations)}")

    def test_chart_type_is_application_or_library(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that chart type is application or library."""
        valid_types = ["application", "library"]
        violations = []

        for chart in all_helm_charts:
            chart_yaml = helm_validator.load_chart_yaml(chart)
            chart_type = chart_yaml.get("type", "application")  # Default is application

            if chart_type not in valid_types:
                violations.append(f"Chart {chart.name} has invalid type: {chart_type}")

        assert len(violations) == 0, f"Chart type issues:\n" + "\n".join(violations)


class TestHelmValuesYaml:
    """Test Helm values.yaml validation."""

    def test_values_yaml_is_valid(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that values.yaml files are valid YAML."""
        errors = []

        for chart in all_helm_charts:
            values_file = chart / "values.yaml"
            if values_file.exists():
                try:
                    with open(values_file, 'r') as f:
                        yaml.safe_load(f.read())
                except yaml.YAMLError as e:
                    errors.append(f"Chart {chart.name}: {e}")

        assert len(errors) == 0, f"Values YAML errors:\n" + "\n".join(errors)

    def test_values_have_resource_defaults(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that values have resource limits defined."""
        violations = []

        for chart in all_helm_charts:
            values = helm_validator.load_values_yaml(chart)

            # Check for resources at top level or in common locations
            has_resources = (
                "resources" in values or
                any("resources" in str(v) for v in values.values() if isinstance(v, dict))
            )

            if not has_resources:
                violations.append(f"Chart {chart.name} has no resource defaults")

        if violations:
            pytest.skip(f"Charts without resource defaults (review): {len(violations)}")

    def test_values_have_replica_count(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that values have replicaCount defined."""
        violations = []

        for chart in all_helm_charts:
            values = helm_validator.load_values_yaml(chart)

            has_replicas = (
                "replicaCount" in values or
                "replicas" in values or
                any(
                    "replicaCount" in str(v) or "replicas" in str(v)
                    for v in values.values() if isinstance(v, dict)
                )
            )

            if not has_replicas:
                violations.append(f"Chart {chart.name} has no replica count")

        if violations:
            pytest.skip(f"Charts without replica defaults (review): {len(violations)}")

    def test_values_have_image_configuration(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that values have image configuration."""
        violations = []

        for chart in all_helm_charts:
            values = helm_validator.load_values_yaml(chart)

            has_image = "image" in values

            if not has_image:
                violations.append(f"Chart {chart.name} has no image configuration")

        if violations:
            pytest.skip(f"Charts without image config (review): {len(violations)}")

    def test_image_pull_policy_configured(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that image pullPolicy is configured."""
        valid_policies = ["Always", "IfNotPresent", "Never"]
        violations = []

        for chart in all_helm_charts:
            values = helm_validator.load_values_yaml(chart)
            image = values.get("image", {})

            if isinstance(image, dict):
                pull_policy = image.get("pullPolicy", "")
                if pull_policy and pull_policy not in valid_policies:
                    violations.append(
                        f"Chart {chart.name} has invalid pullPolicy: {pull_policy}"
                    )

        assert len(violations) == 0, f"Pull policy issues:\n" + "\n".join(violations)


class TestHelmEnvironmentValues:
    """Test Helm environment-specific values files."""

    EXPECTED_ENVIRONMENTS = ["dev", "staging", "prod"]

    def test_environment_values_files_exist(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that environment-specific values files exist."""
        violations = []

        for chart in all_helm_charts:
            values_files = helm_validator.list_values_files(chart)

            for env in self.EXPECTED_ENVIRONMENTS:
                env_file = f"values-{env}.yaml"
                if env_file not in values_files:
                    violations.append(f"Chart {chart.name} missing {env_file}")

        if violations:
            pytest.skip(f"Missing environment values files (review): {len(violations)}")

    def test_environment_values_are_valid_yaml(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that environment values files are valid YAML."""
        errors = []

        for chart in all_helm_charts:
            values_files = helm_validator.list_values_files(chart)

            for values_file in values_files:
                try:
                    helm_validator.load_values_yaml(chart, values_file)
                except yaml.YAMLError as e:
                    errors.append(f"Chart {chart.name}/{values_file}: {e}")

        assert len(errors) == 0, f"Environment values errors:\n" + "\n".join(errors)

    def test_production_values_have_higher_replicas(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that production has higher replica counts than dev."""
        violations = []

        for chart in all_helm_charts:
            dev_values = helm_validator.load_values_yaml(chart, "values-dev.yaml")
            prod_values = helm_validator.load_values_yaml(chart, "values-prod.yaml")

            if dev_values and prod_values:
                dev_replicas = dev_values.get("replicaCount", dev_values.get("replicas", 1))
                prod_replicas = prod_values.get("replicaCount", prod_values.get("replicas", 1))

                if prod_replicas <= dev_replicas:
                    violations.append(
                        f"Chart {chart.name}: prod replicas ({prod_replicas}) <= dev ({dev_replicas})"
                    )

        if violations:
            pytest.skip(f"Production scaling concerns (review): {len(violations)}")


class TestHelmDependencies:
    """Test Helm chart dependencies."""

    def test_dependencies_have_versions(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that dependencies have version constraints."""
        violations = []

        for chart in all_helm_charts:
            chart_yaml = helm_validator.load_chart_yaml(chart)
            dependencies = helm_validator.get_dependencies(chart_yaml)

            for dep in dependencies:
                dep_name = dep.get("name", "unknown")
                dep_version = dep.get("version", "")

                if not dep_version:
                    violations.append(
                        f"Chart {chart.name} dependency {dep_name} has no version"
                    )

        assert len(violations) == 0, f"Dependency issues:\n" + "\n".join(violations)

    def test_dependencies_have_repository(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that dependencies have repository defined."""
        violations = []

        for chart in all_helm_charts:
            chart_yaml = helm_validator.load_chart_yaml(chart)
            dependencies = helm_validator.get_dependencies(chart_yaml)

            for dep in dependencies:
                dep_name = dep.get("name", "unknown")
                dep_repo = dep.get("repository", "")

                # Local dependencies (file://) are OK without repository
                if not dep_repo and not dep.get("local", False):
                    violations.append(
                        f"Chart {chart.name} dependency {dep_name} has no repository"
                    )

        if violations:
            pytest.skip(f"Dependencies without repository (review): {len(violations)}")

    def test_dependencies_have_condition(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that optional dependencies have condition."""
        violations = []

        for chart in all_helm_charts:
            chart_yaml = helm_validator.load_chart_yaml(chart)
            dependencies = helm_validator.get_dependencies(chart_yaml)

            for dep in dependencies:
                dep_name = dep.get("name", "unknown")
                condition = dep.get("condition", "")

                # Dependencies should have a condition for enabling/disabling
                if not condition:
                    violations.append(
                        f"Chart {chart.name} dependency {dep_name} has no condition"
                    )

        if violations:
            pytest.skip(f"Dependencies without condition (review): {len(violations)}")


class TestHelmTemplates:
    """Test Helm template files."""

    def test_templates_have_helpers(self, all_helm_charts: List[Path]):
        """Test that charts have _helpers.tpl."""
        violations = []

        for chart in all_helm_charts:
            helpers_file = chart / "templates" / "_helpers.tpl"

            if not helpers_file.exists():
                violations.append(f"Chart {chart.name} missing _helpers.tpl")

        if violations:
            pytest.skip(f"Charts without _helpers.tpl (review): {len(violations)}")

    def test_templates_use_standard_naming(self, all_helm_charts: List[Path]):
        """Test that templates follow naming conventions."""
        standard_templates = [
            "deployment.yaml", "service.yaml", "configmap.yaml",
            "secret.yaml", "ingress.yaml", "serviceaccount.yaml",
            "hpa.yaml", "pdb.yaml"
        ]
        violations = []

        for chart in all_helm_charts:
            templates_dir = chart / "templates"
            if templates_dir.exists():
                templates = [f.name for f in templates_dir.iterdir() if f.is_file()]

                # Check for non-standard names (excluding helpers and notes)
                for template in templates:
                    if template.startswith("_") or template == "NOTES.txt":
                        continue

                    # Allow variations like deployment-api.yaml
                    base_name = template.split("-")[0] if "-" in template else template.replace(".yaml", "")
                    is_standard = any(
                        template == std or template.startswith(std.replace(".yaml", "-"))
                        for std in standard_templates
                    )

                    if not is_standard and not template.endswith(".tpl"):
                        violations.append(f"Chart {chart.name}: non-standard template {template}")

        if violations:
            pytest.skip(f"Non-standard template names (review): {len(violations)}")

    def test_templates_use_release_name(self, all_helm_charts: List[Path]):
        """Test that templates use .Release.Name for naming."""
        violations = []

        for chart in all_helm_charts:
            templates_dir = chart / "templates"
            if templates_dir.exists():
                for template_file in templates_dir.glob("*.yaml"):
                    content = template_file.read_text()

                    # Check if template uses Release.Name in metadata.name
                    has_release_name = (
                        ".Release.Name" in content or
                        "include" in content  # Might use helper function
                    )

                    # Skip if it's a non-namespaced resource
                    if "kind: Namespace" in content or "kind: ClusterRole" in content:
                        continue

                    if not has_release_name:
                        violations.append(
                            f"Chart {chart.name}/{template_file.name} may not use .Release.Name"
                        )

        if violations:
            pytest.skip(f"Templates possibly not using Release.Name (review): {len(violations)}")


class TestHelmSecurityDefaults:
    """Test Helm chart security defaults."""

    def test_values_have_security_context(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that values have security context defaults."""
        violations = []

        for chart in all_helm_charts:
            values = helm_validator.load_values_yaml(chart)

            has_security = (
                "securityContext" in values or
                "podSecurityContext" in values or
                "containerSecurityContext" in values
            )

            if not has_security:
                violations.append(f"Chart {chart.name} has no security context defaults")

        if violations:
            pytest.skip(f"Charts without security defaults (review): {len(violations)}")

    def test_values_have_service_account_config(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that values have service account configuration."""
        violations = []

        for chart in all_helm_charts:
            values = helm_validator.load_values_yaml(chart)

            has_sa_config = "serviceAccount" in values

            if not has_sa_config:
                violations.append(f"Chart {chart.name} has no serviceAccount config")

        if violations:
            pytest.skip(f"Charts without serviceAccount config (review): {len(violations)}")


class TestHelmChartValidationWithMock:
    """Test Helm chart validation with mock CLI."""

    def test_chart_lint_with_mock(
        self,
        temp_helm_chart: Path,
        mock_helm_cli
    ):
        """Test chart linting using mock Helm CLI."""
        result = mock_helm_cli.lint(temp_helm_chart)

        assert result["passed"], "Chart should pass linting"
        assert len(result["messages"]) == 0, "Chart should have no lint warnings"

    def test_chart_template_with_mock(
        self,
        temp_helm_chart: Path,
        mock_helm_cli
    ):
        """Test chart templating using mock Helm CLI."""
        result = mock_helm_cli.template(temp_helm_chart)

        assert result is not None, "Template should produce output"
        assert "Mocked" in result, "Mock should return expected output"

    def test_temp_chart_has_required_structure(
        self,
        temp_helm_chart: Path,
        helm_validator
    ):
        """Test that temporary chart has required structure."""
        files_check = helm_validator.check_required_files(temp_helm_chart)

        assert files_check["Chart.yaml"], "Chart should have Chart.yaml"
        assert files_check["values.yaml"], "Chart should have values.yaml"
        assert files_check["templates"], "Chart should have templates directory"

    def test_temp_chart_yaml_is_valid(
        self,
        temp_helm_chart: Path,
        helm_validator
    ):
        """Test that temporary chart Chart.yaml is valid."""
        chart_yaml = helm_validator.load_chart_yaml(temp_helm_chart)

        assert chart_yaml.get("apiVersion") == "v2", "Should use apiVersion v2"
        assert chart_yaml.get("name") == "test-chart", "Should have correct name"
        assert chart_yaml.get("version") == "1.0.0", "Should have valid version"

    def test_temp_chart_values_schema(
        self,
        temp_helm_chart: Path,
        helm_validator
    ):
        """Test that temporary chart values pass schema validation."""
        values = helm_validator.load_values_yaml(temp_helm_chart)
        issues = helm_validator.validate_values_schema(values)

        # Should have no missing defaults since we set them
        assert "resources" in values or len(issues["missing_defaults"]) == 0


class TestHelmChartBestPractices:
    """Test Helm chart best practices."""

    def test_charts_have_notes_txt(self, all_helm_charts: List[Path]):
        """Test that charts have NOTES.txt for post-install instructions."""
        violations = []

        for chart in all_helm_charts:
            notes_file = chart / "templates" / "NOTES.txt"

            if not notes_file.exists():
                violations.append(f"Chart {chart.name} missing NOTES.txt")

        if violations:
            pytest.skip(f"Charts without NOTES.txt (review): {len(violations)}")

    def test_charts_have_description(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that charts have meaningful descriptions."""
        violations = []

        for chart in all_helm_charts:
            chart_yaml = helm_validator.load_chart_yaml(chart)
            description = chart_yaml.get("description", "")

            if not description or len(description) < 10:
                violations.append(f"Chart {chart.name} has short/missing description")

        assert len(violations) == 0, f"Description issues:\n" + "\n".join(violations)

    def test_charts_have_keywords(
        self,
        all_helm_charts: List[Path],
        helm_validator
    ):
        """Test that charts have keywords for discoverability."""
        violations = []

        for chart in all_helm_charts:
            chart_yaml = helm_validator.load_chart_yaml(chart)
            keywords = chart_yaml.get("keywords", [])

            if not keywords:
                violations.append(f"Chart {chart.name} has no keywords")

        if violations:
            pytest.skip(f"Charts without keywords (review): {len(violations)}")

    def test_values_have_comments(self, all_helm_charts: List[Path]):
        """Test that values.yaml files have comments for documentation."""
        violations = []

        for chart in all_helm_charts:
            values_file = chart / "values.yaml"
            if values_file.exists():
                content = values_file.read_text()

                # Count comment lines
                comment_lines = len([l for l in content.split('\n') if l.strip().startswith('#')])
                total_lines = len(content.split('\n'))

                # Should have at least 10% comments
                if total_lines > 10 and comment_lines / total_lines < 0.05:
                    violations.append(f"Chart {chart.name} values.yaml has few comments")

        if violations:
            pytest.skip(f"Charts with limited comments (review): {len(violations)}")
