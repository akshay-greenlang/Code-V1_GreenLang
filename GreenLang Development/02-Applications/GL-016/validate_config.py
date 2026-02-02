#!/usr/bin/env python3
"""
GL-016 WATERGUARD - Configuration Validation Script
Project: GreenLang Industrial Sustainability Framework

This script validates all configuration files before deployment.
"""

import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates WATERGUARD configuration files"""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> bool:
        """Run all validation checks"""
        logger.info("Starting configuration validation for GL-016 WATERGUARD...")

        checks = [
            self.validate_deployment_yaml,
            self.validate_service_yaml,
            self.validate_configmap_yaml,
            self.validate_secret_yaml,
            self.validate_hpa_yaml,
            self.validate_network_policy_yaml,
            self.validate_ingress_yaml,
            self.validate_dockerfile,
            self.validate_requirements,
        ]

        for check in checks:
            try:
                check()
            except Exception as e:
                self.errors.append(f"Exception in {check.__name__}: {str(e)}")

        self.print_results()
        return len(self.errors) == 0

    def validate_yaml_file(self, filepath: Path) -> Tuple[bool, Any]:
        """Validate YAML file syntax and load it"""
        if not filepath.exists():
            self.errors.append(f"File not found: {filepath}")
            return False, None

        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            return True, data
        except yaml.YAMLError as e:
            self.errors.append(f"YAML syntax error in {filepath}: {str(e)}")
            return False, None

    def validate_deployment_yaml(self):
        """Validate Kubernetes deployment manifest"""
        logger.info("Validating deployment.yaml...")
        filepath = self.base_path / "deployment" / "deployment.yaml"

        valid, data = self.validate_yaml_file(filepath)
        if not valid:
            return

        # Check required fields
        if data.get('kind') != 'Deployment':
            self.errors.append("deployment.yaml: kind must be 'Deployment'")

        spec = data.get('spec', {})
        if spec.get('replicas', 0) < 3:
            self.warnings.append("deployment.yaml: replicas should be at least 3 for HA")

        # Check container configuration
        containers = spec.get('template', {}).get('spec', {}).get('containers', [])
        if not containers:
            self.errors.append("deployment.yaml: No containers defined")
        else:
            container = containers[0]
            if 'resources' not in container:
                self.errors.append("deployment.yaml: Resource limits/requests not defined")
            if 'livenessProbe' not in container:
                self.warnings.append("deployment.yaml: Liveness probe not defined")
            if 'readinessProbe' not in container:
                self.warnings.append("deployment.yaml: Readiness probe not defined")

        logger.info("deployment.yaml validation complete")

    def validate_service_yaml(self):
        """Validate Kubernetes service manifest"""
        logger.info("Validating service.yaml...")
        filepath = self.base_path / "deployment" / "service.yaml"

        valid, data = self.validate_yaml_file(filepath)
        if not valid:
            return

        if data.get('kind') != 'Service':
            self.errors.append("service.yaml: kind must be 'Service'")

        spec = data.get('spec', {})
        ports = spec.get('ports', [])

        expected_ports = {8000, 9090}
        actual_ports = {p.get('port') for p in ports}

        if not expected_ports.issubset(actual_ports):
            self.warnings.append(
                f"service.yaml: Expected ports {expected_ports}, got {actual_ports}"
            )

        logger.info("service.yaml validation complete")

    def validate_configmap_yaml(self):
        """Validate Kubernetes ConfigMap manifest"""
        logger.info("Validating configmap.yaml...")
        filepath = self.base_path / "deployment" / "configmap.yaml"

        valid, data = self.validate_yaml_file(filepath)
        if not valid:
            return

        if data.get('kind') != 'ConfigMap':
            self.errors.append("configmap.yaml: kind must be 'ConfigMap'")

        logger.info("configmap.yaml validation complete")

    def validate_secret_yaml(self):
        """Validate Kubernetes Secret manifest"""
        logger.info("Validating secret.yaml...")
        filepath = self.base_path / "deployment" / "secret.yaml"

        valid, data = self.validate_yaml_file(filepath)
        if not valid:
            return

        if data.get('kind') != 'Secret':
            self.errors.append("secret.yaml: kind must be 'Secret'")

        if data.get('type') != 'Opaque':
            self.warnings.append("secret.yaml: type should typically be 'Opaque'")

        logger.info("secret.yaml validation complete")

    def validate_hpa_yaml(self):
        """Validate Horizontal Pod Autoscaler manifest"""
        logger.info("Validating hpa.yaml...")
        filepath = self.base_path / "deployment" / "hpa.yaml"

        valid, data = self.validate_yaml_file(filepath)
        if not valid:
            return

        if data.get('kind') != 'HorizontalPodAutoscaler':
            self.errors.append("hpa.yaml: kind must be 'HorizontalPodAutoscaler'")

        spec = data.get('spec', {})
        if spec.get('minReplicas', 0) < 3:
            self.warnings.append("hpa.yaml: minReplicas should be at least 3 for HA")

        logger.info("hpa.yaml validation complete")

    def validate_network_policy_yaml(self):
        """Validate NetworkPolicy manifest"""
        logger.info("Validating networkpolicy.yaml...")
        filepath = self.base_path / "deployment" / "networkpolicy.yaml"

        valid, data = self.validate_yaml_file(filepath)
        if not valid:
            return

        if data.get('kind') != 'NetworkPolicy':
            self.errors.append("networkpolicy.yaml: kind must be 'NetworkPolicy'")

        logger.info("networkpolicy.yaml validation complete")

    def validate_ingress_yaml(self):
        """Validate Ingress manifest"""
        logger.info("Validating ingress.yaml...")
        filepath = self.base_path / "deployment" / "ingress.yaml"

        valid, data = self.validate_yaml_file(filepath)
        if not valid:
            return

        if data.get('kind') != 'Ingress':
            self.errors.append("ingress.yaml: kind must be 'Ingress'")

        spec = data.get('spec', {})
        if 'tls' not in spec:
            self.warnings.append("ingress.yaml: TLS configuration not found")

        logger.info("ingress.yaml validation complete")

    def validate_dockerfile(self):
        """Validate Dockerfile"""
        logger.info("Validating Dockerfile...")
        filepath = self.base_path / "Dockerfile"

        if not filepath.exists():
            self.errors.append("Dockerfile not found")
            return

        with open(filepath, 'r') as f:
            content = f.read()

        required_elements = [
            'FROM',
            'WORKDIR',
            'COPY',
            'RUN',
            'USER',
            'EXPOSE',
            'HEALTHCHECK',
        ]

        for element in required_elements:
            if element not in content:
                self.errors.append(f"Dockerfile: Missing {element} instruction")

        if 'USER root' in content and content.rindex('USER root') > content.rindex('USER'):
            self.errors.append("Dockerfile: Container should not run as root")

        logger.info("Dockerfile validation complete")

    def validate_requirements(self):
        """Validate requirements.txt"""
        logger.info("Validating requirements.txt...")
        filepath = self.base_path / "requirements.txt"

        if not filepath.exists():
            self.errors.append("requirements.txt not found")
            return

        with open(filepath, 'r') as f:
            content = f.read()

        required_packages = [
            'pydantic',
            'numpy',
            'scipy',
            'asyncio',
            'aiohttp',
            'httpx',
            'opcua',
            'pymodbus',
            'prometheus-client',
            'structlog',
            'pytest',
        ]

        for package in required_packages:
            if package not in content:
                self.errors.append(f"requirements.txt: Missing required package: {package}")

        logger.info("requirements.txt validation complete")

    def print_results(self):
        """Print validation results"""
        print("\n" + "="*70)
        print("GL-016 WATERGUARD - Configuration Validation Results")
        print("="*70)

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print("\n✅ All validation checks passed!")

        print("="*70 + "\n")


def main():
    """Main entry point"""
    validator = ConfigValidator()
    success = validator.validate_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
