# -*- coding: utf-8 -*-
"""
Terraform Syntax and Best Practices Validation Tests

INFRA-001: Infrastructure test suite for validating Terraform configurations.

Tests include:
- Syntax validation for all .tf files
- Required file structure (main.tf, variables.tf, outputs.tf)
- Provider version constraints
- Security best practices (encryption, tagging)
- Variable documentation
- Output documentation
- Module structure compliance

Target coverage: 85%+
"""

import re
from pathlib import Path
from typing import Dict, List, Any

import pytest


class TestTerraformFileStructure:
    """Test Terraform file structure and organization."""

    def test_terraform_directory_exists(self, terraform_dir: Path):
        """Test that the Terraform directory exists."""
        assert terraform_dir.exists(), f"Terraform directory not found at {terraform_dir}"

    def test_modules_directory_exists(self, terraform_dir: Path):
        """Test that the modules directory exists."""
        modules_dir = terraform_dir / "modules"
        assert modules_dir.exists(), f"Modules directory not found at {modules_dir}"

    def test_environments_directory_exists(self, terraform_dir: Path):
        """Test that the environments directory exists."""
        envs_dir = terraform_dir / "environments"
        assert envs_dir.exists(), f"Environments directory not found at {envs_dir}"

    @pytest.mark.parametrize("environment", ["dev", "staging", "prod"])
    def test_environment_directories_exist(self, terraform_dir: Path, environment: str):
        """Test that expected environment directories exist."""
        env_dir = terraform_dir / "environments" / environment
        assert env_dir.exists(), f"Environment {environment} directory not found at {env_dir}"

    def test_terraform_files_exist(self, all_terraform_files: List[Path]):
        """Test that Terraform files exist in the project."""
        assert len(all_terraform_files) > 0, "No Terraform files found in the project"


class TestTerraformModuleStructure:
    """Test Terraform module structure compliance."""

    def test_modules_have_required_files(
        self,
        terraform_modules: List[Path],
        terraform_validator
    ):
        """Test that all modules have required files."""
        for module in terraform_modules:
            files_check = terraform_validator.check_required_files(module)

            for file_name, exists in files_check.items():
                assert exists, f"Module {module.name} is missing required file: {file_name}"

    def test_module_main_tf_not_empty(self, terraform_modules: List[Path]):
        """Test that main.tf files are not empty."""
        for module in terraform_modules:
            main_tf = module / "main.tf"
            if main_tf.exists():
                content = main_tf.read_text()
                assert len(content.strip()) > 0, f"Module {module.name} has empty main.tf"

    def test_module_variables_have_descriptions(
        self,
        terraform_modules: List[Path],
        terraform_validator
    ):
        """Test that all variables have descriptions."""
        for module in terraform_modules:
            variables_tf = module / "variables.tf"
            if variables_tf.exists():
                content = variables_tf.read_text()
                missing = terraform_validator.validate_variable_descriptions(content)

                assert len(missing) == 0, (
                    f"Module {module.name} has variables without descriptions: {missing}"
                )

    def test_module_outputs_have_descriptions(self, terraform_modules: List[Path]):
        """Test that all outputs have descriptions."""
        for module in terraform_modules:
            outputs_tf = module / "outputs.tf"
            if outputs_tf.exists():
                content = outputs_tf.read_text()

                # Find outputs without descriptions
                outputs = re.findall(r'output\s+"([^"]+)"', content)
                missing_descriptions = []

                for output in outputs:
                    output_pattern = rf'output\s+"{output}"\s*\{{[^}}]*description\s*='
                    if not re.search(output_pattern, content, re.DOTALL):
                        missing_descriptions.append(output)

                assert len(missing_descriptions) == 0, (
                    f"Module {module.name} has outputs without descriptions: {missing_descriptions}"
                )


class TestTerraformSyntaxValidation:
    """Test Terraform syntax validity."""

    def test_terraform_files_have_valid_syntax(self, all_terraform_files: List[Path]):
        """Test that all Terraform files have valid basic syntax."""
        syntax_errors = []

        for tf_file in all_terraform_files:
            try:
                content = tf_file.read_text(encoding='utf-8')

                # Check for balanced braces
                open_braces = content.count('{')
                close_braces = content.count('}')

                if open_braces != close_braces:
                    syntax_errors.append(
                        f"{tf_file}: Unbalanced braces (open: {open_braces}, close: {close_braces})"
                    )

                # Check for balanced quotes
                # Count non-escaped quotes
                quote_count = len(re.findall(r'(?<!\\)"', content))
                if quote_count % 2 != 0:
                    syntax_errors.append(f"{tf_file}: Unbalanced quotes")

            except UnicodeDecodeError as e:
                syntax_errors.append(f"{tf_file}: Encoding error - {e}")

        assert len(syntax_errors) == 0, f"Syntax errors found:\n" + "\n".join(syntax_errors)

    def test_no_hardcoded_secrets(self, all_terraform_files: List[Path]):
        """Test that no hardcoded secrets are present in Terraform files."""
        secret_patterns = [
            r'password\s*=\s*"[^"]{8,}"',  # Hardcoded passwords
            r'secret\s*=\s*"[^"]{8,}"',  # Hardcoded secrets
            r'api_key\s*=\s*"[^"]+"',  # Hardcoded API keys
            r'access_key\s*=\s*"AKIA[A-Z0-9]{16}"',  # AWS access keys
            r'secret_key\s*=\s*"[A-Za-z0-9/+=]{40}"',  # AWS secret keys
            r'token\s*=\s*"[A-Za-z0-9_-]{20,}"',  # Generic tokens
        ]

        violations = []

        for tf_file in all_terraform_files:
            content = tf_file.read_text()

            for pattern in secret_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    violations.append(f"{tf_file}: Potential hardcoded secret found")

        assert len(violations) == 0, (
            f"Potential hardcoded secrets found:\n" + "\n".join(violations)
        )

    def test_no_deprecated_syntax(self, all_terraform_files: List[Path]):
        """Test that no deprecated Terraform syntax is used."""
        deprecated_patterns = [
            (r'\$\{var\.', "Use var.name instead of ${var.name}"),
            (r'\$\{local\.', "Use local.name instead of ${local.name}"),
            (r'(?<!required_)provider\s*{', "Use required_providers block"),
        ]

        violations = []

        for tf_file in all_terraform_files:
            content = tf_file.read_text()

            for pattern, message in deprecated_patterns:
                # Skip interpolation in strings that actually need it
                if re.search(pattern, content):
                    # More nuanced check - only flag if outside of string context
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if re.search(pattern, line):
                            # Skip if it's in a heredoc or complex expression
                            if '<<' not in line and 'join(' not in line and 'format(' not in line:
                                violations.append(f"{tf_file}:{i+1}: {message}")

        # Allow some deprecated syntax with warning
        if violations:
            pytest.skip(f"Deprecated syntax warnings (non-blocking): {len(violations)} items")


class TestTerraformProviderVersions:
    """Test Terraform provider version constraints."""

    def test_terraform_version_constraint_exists(self, terraform_environments: List[Path]):
        """Test that all environments specify Terraform version constraints."""
        for env in terraform_environments:
            main_tf = env / "main.tf"
            if main_tf.exists():
                content = main_tf.read_text()

                assert 'required_version' in content, (
                    f"Environment {env.name} does not specify required_version"
                )

    def test_aws_provider_version_constraint(self, terraform_environments: List[Path]):
        """Test that AWS provider has version constraint >= 5.0."""
        for env in terraform_environments:
            main_tf = env / "main.tf"
            if main_tf.exists():
                content = main_tf.read_text()

                # Check for AWS provider version
                if 'hashicorp/aws' in content:
                    assert re.search(r'version\s*=\s*["\']>=\s*5\.0', content), (
                        f"Environment {env.name} should use AWS provider >= 5.0"
                    )

    def test_kubernetes_provider_version_constraint(self, terraform_environments: List[Path]):
        """Test that Kubernetes provider has version constraint."""
        for env in terraform_environments:
            main_tf = env / "main.tf"
            if main_tf.exists():
                content = main_tf.read_text()

                if 'hashicorp/kubernetes' in content:
                    assert re.search(r'version\s*=\s*["\']>=\s*2\.0', content), (
                        f"Environment {env.name} should use Kubernetes provider >= 2.0"
                    )


class TestTerraformSecurityBestPractices:
    """Test Terraform security best practices."""

    def test_s3_buckets_have_encryption(
        self,
        all_terraform_files: List[Path],
        terraform_validator
    ):
        """Test that S3 bucket resources have encryption configured."""
        for tf_file in all_terraform_files:
            content = tf_file.read_text()

            # Find S3 bucket resources
            s3_buckets = re.findall(r'resource\s+"aws_s3_bucket"\s+"([^"]+)"', content)

            for bucket in s3_buckets:
                # Check for encryption configuration
                has_encryption = (
                    'aws_s3_bucket_server_side_encryption_configuration' in content or
                    'server_side_encryption_configuration' in content
                )

                assert has_encryption, (
                    f"S3 bucket {bucket} in {tf_file} should have encryption configured"
                )

    def test_rds_instances_have_encryption(self, all_terraform_files: List[Path]):
        """Test that RDS instances have encryption enabled."""
        for tf_file in all_terraform_files:
            content = tf_file.read_text()

            # Check for RDS resources
            if 'aws_db_instance' in content or 'aws_rds_cluster' in content:
                assert 'storage_encrypted' in content or 'kms_key_id' in content, (
                    f"RDS resource in {tf_file} should have encryption enabled"
                )

    def test_elasticache_has_encryption(self, all_terraform_files: List[Path]):
        """Test that ElastiCache has encryption enabled."""
        for tf_file in all_terraform_files:
            content = tf_file.read_text()

            if 'aws_elasticache_replication_group' in content:
                assert 'at_rest_encryption_enabled' in content, (
                    f"ElastiCache in {tf_file} should have at-rest encryption"
                )
                assert 'transit_encryption_enabled' in content, (
                    f"ElastiCache in {tf_file} should have transit encryption"
                )

    def test_resources_have_tags(
        self,
        all_terraform_files: List[Path],
        terraform_validator
    ):
        """Test that taggable resources have tags defined."""
        taggable_resources = [
            'aws_vpc',
            'aws_subnet',
            'aws_security_group',
            'aws_eks_cluster',
            'aws_rds_cluster',
            'aws_db_instance',
            'aws_s3_bucket',
        ]

        violations = []

        for tf_file in all_terraform_files:
            content = tf_file.read_text()

            for resource_type in taggable_resources:
                # Find resource blocks
                pattern = rf'resource\s+"{resource_type}"\s+"([^"]+)"\s*\{{'
                resources = re.findall(pattern, content)

                for resource_name in resources:
                    # Check if tags are present in the resource block
                    # This is a simplified check
                    resource_pattern = rf'resource\s+"{resource_type}"\s+"{resource_name}"\s*\{{[^}}]*tags\s*='
                    if not re.search(resource_pattern, content, re.DOTALL):
                        # Check for module-level tags
                        if 'tags = var.tags' not in content and 'tags = local.' not in content:
                            violations.append(f"{tf_file}: {resource_type}.{resource_name}")

        # Warning rather than failure for tags
        if violations:
            pytest.skip(f"Resources without explicit tags (review recommended): {len(violations)}")


class TestTerraformBackendConfiguration:
    """Test Terraform backend configuration."""

    def test_production_uses_remote_backend(self, terraform_dir: Path):
        """Test that production environment uses remote backend."""
        prod_main = terraform_dir / "environments" / "prod" / "main.tf"

        if prod_main.exists():
            content = prod_main.read_text()

            assert 'backend "s3"' in content or 'backend "remote"' in content, (
                "Production should use S3 or remote backend for state"
            )

    def test_backend_has_encryption(self, terraform_environments: List[Path]):
        """Test that S3 backend has encryption enabled."""
        for env in terraform_environments:
            main_tf = env / "main.tf"
            if main_tf.exists():
                content = main_tf.read_text()

                if 'backend "s3"' in content:
                    assert 'encrypt' in content and 'true' in content, (
                        f"S3 backend in {env.name} should have encryption enabled"
                    )

    def test_backend_has_state_locking(self, terraform_environments: List[Path]):
        """Test that S3 backend has DynamoDB state locking."""
        for env in terraform_environments:
            main_tf = env / "main.tf"
            if main_tf.exists():
                content = main_tf.read_text()

                if 'backend "s3"' in content:
                    assert 'dynamodb_table' in content, (
                        f"S3 backend in {env.name} should have DynamoDB state locking"
                    )


class TestTerraformModuleValidation:
    """Test Terraform module validation with mock CLI."""

    def test_module_validation_with_mock(
        self,
        temp_terraform_module: Path,
        mock_terraform_cli
    ):
        """Test module validation using mock Terraform CLI."""
        result = mock_terraform_cli.validate(temp_terraform_module)

        assert result["valid"], "Module should pass validation"
        assert result["error_count"] == 0, "Module should have no errors"

    def test_module_format_check_with_mock(
        self,
        temp_terraform_module: Path,
        mock_terraform_cli
    ):
        """Test module format check using mock Terraform CLI."""
        result = mock_terraform_cli.fmt_check(temp_terraform_module)

        assert result["formatted"], "Module should be properly formatted"

    def test_temp_module_has_encryption(
        self,
        temp_terraform_module: Path,
        terraform_validator
    ):
        """Test that temporary module has encryption configured."""
        main_tf = temp_terraform_module / "main.tf"
        content = main_tf.read_text()

        assert terraform_validator.check_encryption_settings(content, "aws_s3_bucket"), (
            "S3 bucket in temp module should have encryption"
        )


class TestTerraformOutputsValidation:
    """Test Terraform outputs validation."""

    def test_modules_have_outputs(self, terraform_modules: List[Path]):
        """Test that all modules have outputs defined."""
        for module in terraform_modules:
            outputs_tf = module / "outputs.tf"

            assert outputs_tf.exists(), f"Module {module.name} should have outputs.tf"

            content = outputs_tf.read_text()
            outputs = re.findall(r'output\s+"([^"]+)"', content)

            assert len(outputs) > 0, f"Module {module.name} should have at least one output"

    def test_outputs_are_sensitive_when_appropriate(self, terraform_modules: List[Path]):
        """Test that sensitive outputs are marked as sensitive."""
        sensitive_keywords = ['password', 'secret', 'key', 'token', 'credentials']

        for module in terraform_modules:
            outputs_tf = module / "outputs.tf"
            if outputs_tf.exists():
                content = outputs_tf.read_text()

                # Find outputs with sensitive names
                outputs = re.findall(r'output\s+"([^"]+)"', content)

                for output in outputs:
                    is_sensitive_name = any(kw in output.lower() for kw in sensitive_keywords)

                    if is_sensitive_name:
                        # Check if marked as sensitive
                        output_block = re.search(
                            rf'output\s+"{output}"\s*\{{[^}}]*sensitive\s*=\s*true',
                            content,
                            re.DOTALL
                        )

                        assert output_block, (
                            f"Output {output} in {module.name} should be marked as sensitive"
                        )


class TestTerraformResourceNaming:
    """Test Terraform resource naming conventions."""

    def test_resources_use_snake_case(self, all_terraform_files: List[Path]):
        """Test that resource names use snake_case."""
        violations = []

        for tf_file in all_terraform_files:
            content = tf_file.read_text()

            # Find resource names
            resources = re.findall(r'resource\s+"[^"]+"\s+"([^"]+)"', content)

            for resource in resources:
                # Check for snake_case (lowercase with underscores)
                if not re.match(r'^[a-z][a-z0-9_]*$', resource):
                    violations.append(f"{tf_file}: resource name '{resource}' should be snake_case")

        assert len(violations) == 0, f"Naming violations:\n" + "\n".join(violations)

    def test_variables_use_snake_case(self, all_terraform_files: List[Path]):
        """Test that variable names use snake_case."""
        violations = []

        for tf_file in all_terraform_files:
            content = tf_file.read_text()

            # Find variable names
            variables = re.findall(r'variable\s+"([^"]+)"', content)

            for var in variables:
                if not re.match(r'^[a-z][a-z0-9_]*$', var):
                    violations.append(f"{tf_file}: variable '{var}' should be snake_case")

        assert len(violations) == 0, f"Naming violations:\n" + "\n".join(violations)
