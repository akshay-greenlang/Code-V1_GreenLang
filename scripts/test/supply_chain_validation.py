#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supply Chain Security Validation Script
========================================
GL-SupplyChainSentinel Comprehensive Validation

This script validates all supply chain security components according to
strict security requirements.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from greenlang.determinism import DeterministicClock

# Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class SupplyChainValidator:
    """Validates supply chain security artifacts with zero tolerance"""

    def __init__(self):
        self.validation_results = []
        self.blocking_issues = []
        self.overall_status = "PASS"

    def validate_sbom(self, sbom_path: Path) -> Tuple[str, str, str]:
        """Validate SBOM completeness and SPDX compliance"""
        component = "SBOM"

        if not sbom_path.exists():
            status = "FAIL"
            reason = f"No SBOM found at {sbom_path}"
            details = "SBOM is required for all artifacts"
            remediation = f"Generate SBOM using: python -m greenlang.provenance.sbom generate {sbom_path.parent}"
            self.blocking_issues.append(f"Missing SBOM: {sbom_path}")
            return status, reason, remediation

        try:
            with open(sbom_path, 'r') as f:
                sbom = json.load(f)

            # Check for SPDX format
            if 'spdxVersion' in sbom:
                # Validate SPDX required fields
                required_fields = [
                    'spdxVersion', 'dataLicense', 'SPDXID',
                    'name', 'documentNamespace', 'creationInfo'
                ]

                missing_fields = []
                for field in required_fields:
                    if field not in sbom:
                        missing_fields.append(field)

                if missing_fields:
                    status = "FAIL"
                    reason = f"Invalid SPDX format - missing fields: {', '.join(missing_fields)}"
                    details = f"SPDX version: {sbom.get('spdxVersion', 'unknown')}"
                    remediation = "Regenerate SBOM with proper SPDX format using syft or cyclonedx-bom"
                    self.blocking_issues.append(f"Invalid SBOM format: {sbom_path}")
                else:
                    # Check SPDX version
                    version = sbom.get('spdxVersion', '')
                    if not version.startswith('SPDX-2.'):
                        status = "FAIL"
                        reason = f"Unsupported SPDX version: {version}"
                        details = "Minimum required: SPDX-2.2"
                        remediation = "Update SBOM generator to produce SPDX 2.2 or higher"
                        self.blocking_issues.append(f"Outdated SPDX version: {version}")
                    else:
                        status = "PASS"
                        reason = f"Valid SPDX {version} SBOM with all required fields"
                        details = f"Packages: {len(sbom.get('packages', []))}, Created: {sbom.get('creationInfo', {}).get('created', 'unknown')}"
                        remediation = ""

            elif 'bomFormat' in sbom and sbom['bomFormat'] == 'CycloneDX':
                # CycloneDX format
                status = "PASS"
                reason = f"Valid CycloneDX SBOM version {sbom.get('specVersion', 'unknown')}"
                details = f"Components: {len(sbom.get('components', []))}"
                remediation = ""
            else:
                status = "FAIL"
                reason = "Unknown SBOM format"
                details = "Expected SPDX or CycloneDX format"
                remediation = "Generate SBOM using: cyclonedx-py environment -o sbom.spdx.json --format json"
                self.blocking_issues.append(f"Unknown SBOM format: {sbom_path}")

        except json.JSONDecodeError as e:
            status = "FAIL"
            reason = f"Invalid JSON in SBOM: {e}"
            details = "SBOM file is corrupted or malformed"
            remediation = "Regenerate SBOM file"
            self.blocking_issues.append(f"Corrupted SBOM: {sbom_path}")
        except Exception as e:
            status = "FAIL"
            reason = f"Error reading SBOM: {e}"
            details = "Unexpected error during SBOM validation"
            remediation = "Check SBOM file permissions and format"
            self.blocking_issues.append(f"SBOM read error: {sbom_path}")

        return status, reason, remediation

    def validate_signatures(self, artifact_path: Path) -> Tuple[str, str, str]:
        """Validate cosign/sigstore signatures"""
        component = "SIGNATURE"

        # Check for signature files
        sig_extensions = ['.sig', '.asc', '.sigstore', '.bundle']
        sig_file = None

        for ext in sig_extensions:
            potential_sig = artifact_path.with_suffix(artifact_path.suffix + ext)
            if potential_sig.exists():
                sig_file = potential_sig
                break

        if artifact_path.is_dir():
            # Check for pack.sig in directory
            potential_sig = artifact_path / 'pack.sig'
            if potential_sig.exists():
                sig_file = potential_sig

        if not sig_file:
            status = "FAIL"
            reason = f"No signature found for {artifact_path.name}"
            details = f"Checked extensions: {', '.join(sig_extensions)}"
            remediation = f"Sign artifact using: python -m sigstore sign {artifact_path}"
            self.blocking_issues.append(f"Unsigned artifact: {artifact_path}")
            return status, reason, remediation

        # Try to validate signature
        try:
            # Import GreenLang's signature verifier
            from greenlang.security.signatures import PackVerifier

            verifier = PackVerifier()

            # Check if proper tools are available
            if not verifier.cosign_available and not verifier.sigstore_available:
                # Check if in dev mode
                if os.getenv('GREENLANG_DEV_MODE') == 'true':
                    status = "FAIL"
                    reason = "DEV MODE signatures are not acceptable for production"
                    details = "Cosign and sigstore-python not available, using stub verification"
                    remediation = "Install sigstore: pip install sigstore, or install cosign from GitHub releases"
                    self.blocking_issues.append("Development mode signatures detected")
                else:
                    status = "FAIL"
                    reason = "No signature verification tools available"
                    details = "Neither cosign nor sigstore-python installed"
                    remediation = "Install sigstore: pip install greenlang-cli[security]"
                    self.blocking_issues.append("Missing signature verification tools")
            else:
                # Attempt verification
                try:
                    verified, metadata = verifier.verify_pack(
                        artifact_path,
                        signature_path=sig_file,
                        require_signature=True
                    )

                    if verified:
                        if metadata.get('sigstore'):
                            status = "PASS"
                            reason = f"Valid Sigstore signature verified"
                            details = f"Publisher: {metadata.get('publisher')}, Timestamp: {metadata.get('timestamp')}"
                            remediation = ""
                        else:
                            status = "FAIL"
                            reason = "Signature not from Sigstore/cosign"
                            details = f"Using: {metadata.get('signed_with', 'unknown')}"
                            remediation = "Re-sign with cosign or sigstore"
                            self.blocking_issues.append(f"Non-Sigstore signature: {artifact_path}")
                    else:
                        status = "FAIL"
                        reason = "Signature verification failed"
                        details = str(metadata)
                        remediation = "Re-sign the artifact with valid keys"
                        self.blocking_issues.append(f"Invalid signature: {artifact_path}")

                except Exception as e:
                    status = "FAIL"
                    reason = f"Signature verification error: {e}"
                    details = "Verification process failed"
                    remediation = "Check signature format and re-sign if necessary"
                    self.blocking_issues.append(f"Signature verification failed: {artifact_path}")

        except ImportError:
            status = "FAIL"
            reason = "Cannot import GreenLang signature verifier"
            details = "greenlang.security.signatures module not found"
            remediation = "Ensure GreenLang is properly installed"
            self.blocking_issues.append("Missing GreenLang security module")

        return status, reason, remediation

    def validate_provenance(self, provenance_path: Path) -> Tuple[str, str, str]:
        """Validate provenance documentation"""
        component = "PROVENANCE"

        if not provenance_path.exists():
            status = "FAIL"
            reason = f"No provenance file found at {provenance_path}"
            details = "Human-readable provenance is required"
            remediation = "Create provenance.txt with build details: builder, timestamp, commit, version"
            self.blocking_issues.append(f"Missing provenance: {provenance_path}")
            return status, reason, remediation

        try:
            with open(provenance_path, 'r') as f:
                content = f.read()

            # Check for required provenance fields
            required_items = [
                ('Build Date', 'timestamp'),
                ('Git Commit', 'commit hash'),
                ('Builder', 'CI system identity'),
                ('Version', 'artifact version')
            ]

            missing_items = []
            for item, desc in required_items:
                if item.lower() not in content.lower():
                    missing_items.append(f"{item} ({desc})")

            if missing_items:
                status = "FAIL"
                reason = f"Incomplete provenance - missing: {', '.join(missing_items)}"
                details = f"File exists but lacks critical information"
                remediation = f"Add missing fields to {provenance_path}: " + ', '.join(missing_items)
                self.blocking_issues.append(f"Incomplete provenance: {provenance_path}")
            else:
                # Check if it's machine-only format
                if content.startswith('{') or content.startswith('['):
                    status = "FAIL"
                    reason = "Provenance is machine-only format (JSON)"
                    details = "Human-readable provenance required"
                    remediation = "Convert to human-readable text format with clear headers"
                    self.blocking_issues.append("Machine-only provenance format")
                else:
                    status = "PASS"
                    reason = "Valid human-readable provenance with all required fields"
                    details = f"File size: {len(content)} bytes"
                    remediation = ""

        except Exception as e:
            status = "FAIL"
            reason = f"Error reading provenance: {e}"
            details = "Cannot validate provenance content"
            remediation = f"Check file permissions and format of {provenance_path}"
            self.blocking_issues.append(f"Provenance read error: {provenance_path}")

        return status, reason, remediation

    def validate_oidc_identity(self, metadata: Dict[str, Any]) -> Tuple[str, str, str]:
        """Validate OIDC identity claims from CI"""
        component = "OIDC_IDENTITY"

        # For now, check if running in CI
        ci_indicators = [
            'GITHUB_ACTIONS',
            'CI',
            'GITLAB_CI',
            'CIRCLECI',
            'JENKINS_HOME'
        ]

        in_ci = any(os.getenv(var) for var in ci_indicators)

        if not in_ci:
            status = "FAIL"
            reason = "Not running in CI environment"
            details = "OIDC identity verification requires CI environment"
            remediation = "Run validation in GitHub Actions or other CI with OIDC support"
            self.blocking_issues.append("OIDC identity validation requires CI")
            return status, reason, remediation

        # Check GitHub Actions specific OIDC
        if os.getenv('GITHUB_ACTIONS'):
            repo = os.getenv('GITHUB_REPOSITORY', 'unknown')
            workflow = os.getenv('GITHUB_WORKFLOW', 'unknown')
            ref = os.getenv('GITHUB_REF', 'unknown')

            status = "PASS"
            reason = f"GitHub Actions OIDC identity available"
            details = f"Repository: {repo}, Workflow: {workflow}, Ref: {ref}"
            remediation = ""
        else:
            status = "FAIL"
            reason = "OIDC identity not configured for this CI"
            details = "Only GitHub Actions OIDC currently supported"
            remediation = "Use GitHub Actions with id-token: write permission"
            self.blocking_issues.append("Non-GitHub CI OIDC not supported")

        return status, reason, remediation

    def print_result(self, component: str, status: str, reason: str, details: str, remediation: str):
        """Print formatted validation result"""
        color = GREEN if status == "PASS" else RED

        print(f"\n{BOLD}[{component}]: {color}{status}{RESET}")
        print(f"Reason: {reason}")
        if details:
            print(f"Details: {details}")
        if remediation and status == "FAIL":
            print(f"{YELLOW}Remediation: {remediation}{RESET}")

    def run_validation(self):
        """Run complete supply chain validation"""
        print(f"{BOLD}{BLUE}={'='*60}")
        print("GL-SupplyChainSentinel Security Validation")
        print(f"{'='*60}{RESET}")
        print(f"Timestamp: {DeterministicClock.now().isoformat()}")
        print(f"Environment: {'DEV MODE' if os.getenv('GREENLANG_DEV_MODE') == 'true' else 'PRODUCTION'}")

        # Define test artifacts
        test_artifacts = [
            {
                'name': 'PyPI Package SBOM',
                'sbom': Path('artifacts/sbom/sbom-greenlang-0.2.0-sdist.spdx.json'),
                'artifact': Path('dist/greenlang-cli-0.2.0.tar.gz'),
                'provenance': Path('provenance.txt')
            },
            {
                'name': 'Test Pack',
                'sbom': Path('test-mit-pack/sbom.spdx.json'),
                'artifact': Path('test-mit-pack'),
                'provenance': Path('provenance.txt')
            }
        ]

        for artifact_set in test_artifacts:
            print(f"\n{BOLD}Validating: {artifact_set['name']}{RESET}")
            print("-" * 40)

            # Validate SBOM
            if artifact_set['sbom'].exists() or True:  # Always test
                status, reason, remediation = self.validate_sbom(artifact_set['sbom'])
                self.print_result("SBOM", status, reason, "", remediation)
                if status == "FAIL":
                    self.overall_status = "FAIL"

            # Validate Signatures
            if artifact_set['artifact'].exists() or True:  # Always test
                status, reason, remediation = self.validate_signatures(artifact_set['artifact'])
                self.print_result("SIGNATURE", status, reason, "", remediation)
                if status == "FAIL":
                    self.overall_status = "FAIL"

            # Validate Provenance
            if artifact_set['provenance'].exists() or True:  # Always test
                status, reason, remediation = self.validate_provenance(artifact_set['provenance'])
                self.print_result("PROVENANCE", status, reason, "", remediation)
                if status == "FAIL":
                    self.overall_status = "FAIL"

        # Validate OIDC if in CI
        print(f"\n{BOLD}CI/CD Security:{RESET}")
        print("-" * 40)
        status, reason, remediation = self.validate_oidc_identity({})
        self.print_result("OIDC_IDENTITY", status, reason, "", remediation)
        if status == "FAIL" and os.getenv('CI'):
            self.overall_status = "FAIL"

        # Final verdict
        print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
        verdict_color = GREEN if self.overall_status == "PASS" else RED
        print(f"{BOLD}OVERALL: {verdict_color}{self.overall_status}{RESET}")

        if self.overall_status == "PASS":
            print(f"Summary: {GREEN}All supply chain security requirements met{RESET}")
        else:
            print(f"Summary: {RED}Supply chain security validation failed{RESET}")

        if self.blocking_issues:
            print(f"\n{BOLD}Blocking Issues:{RESET}")
            for i, issue in enumerate(self.blocking_issues, 1):
                print(f"  {RED}{i}. {issue}{RESET}")

            print(f"\n{BOLD}Remediation Priority:{RESET}")
            priority_items = []

            if any("Missing SBOM" in issue for issue in self.blocking_issues):
                priority_items.append("Generate SBOM files using cyclonedx-py or syft")
            if any("Unsigned" in issue for issue in self.blocking_issues):
                priority_items.append("Sign all artifacts with cosign or sigstore")
            if any("Missing provenance" in issue for issue in self.blocking_issues):
                priority_items.append("Create provenance.txt with build metadata")
            if any("DEV MODE" in issue for issue in self.blocking_issues):
                priority_items.append("Disable GREENLANG_DEV_MODE for production")
            if any("verification tools" in issue for issue in self.blocking_issues):
                priority_items.append("Install sigstore: pip install greenlang-cli[security]")

            for i, item in enumerate(priority_items, 1):
                print(f"  {YELLOW}{i}. {item}{RESET}")

        print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")

        # Return exit code
        return 0 if self.overall_status == "PASS" else 1


if __name__ == "__main__":
    validator = SupplyChainValidator()
    sys.exit(validator.run_validation())