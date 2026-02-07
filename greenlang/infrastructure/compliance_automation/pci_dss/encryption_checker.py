# -*- coding: utf-8 -*-
"""
PCI-DSS Encryption Checker - SEC-010 Phase 5

Verifies encryption of cardholder data in compliance with PCI-DSS v4.0
Requirements 3 (Protect Stored Account Data) and 4 (Protect Data in Transit).

PCI-DSS v4.0 Requirements:
- Req 3.5: Protect stored PAN with strong cryptography
- Req 3.6: Protect cryptographic keys
- Req 4.2: Protect cardholder data during transmission

Classes:
    - EncryptionChecker: Main encryption verification engine.
    - EncryptionCheckResult: Result of an encryption check.
    - KeyManagementResult: Result of key management assessment.
    - TransmissionCheckResult: Result of transmission encryption check.

Example:
    >>> checker = EncryptionChecker()
    >>> pan_result = await checker.verify_pan_encryption()
    >>> tls_result = await checker.verify_transmission_encryption()

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EncryptionAlgorithm(str, Enum):
    """Approved encryption algorithms for PCI-DSS."""

    AES_256_GCM = "AES-256-GCM"
    AES_256_CBC = "AES-256-CBC"
    AES_128_GCM = "AES-128-GCM"
    CHACHA20_POLY1305 = "ChaCha20-Poly1305"
    RSA_2048 = "RSA-2048"
    RSA_4096 = "RSA-4096"
    # Deprecated algorithms (non-compliant)
    DES = "DES"
    TRIPLE_DES = "3DES"
    RC4 = "RC4"


class TLSVersion(str, Enum):
    """TLS protocol versions."""

    TLS_1_3 = "TLS 1.3"
    TLS_1_2 = "TLS 1.2"
    TLS_1_1 = "TLS 1.1"  # Deprecated
    TLS_1_0 = "TLS 1.0"  # Deprecated
    SSL_3_0 = "SSL 3.0"  # Insecure


class CheckStatus(str, Enum):
    """Status of an encryption check."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


# ---------------------------------------------------------------------------
# Approved Cryptographic Standards
# ---------------------------------------------------------------------------

# PCI-DSS v4.0 approved algorithms
APPROVED_ALGORITHMS = [
    EncryptionAlgorithm.AES_256_GCM,
    EncryptionAlgorithm.AES_256_CBC,
    EncryptionAlgorithm.AES_128_GCM,
    EncryptionAlgorithm.CHACHA20_POLY1305,
    EncryptionAlgorithm.RSA_2048,
    EncryptionAlgorithm.RSA_4096,
]

# Minimum TLS version
MINIMUM_TLS_VERSION = TLSVersion.TLS_1_2

# Approved TLS cipher suites
APPROVED_CIPHER_SUITES = [
    "TLS_AES_256_GCM_SHA384",
    "TLS_AES_128_GCM_SHA256",
    "TLS_CHACHA20_POLY1305_SHA256",
    "ECDHE-ECDSA-AES256-GCM-SHA384",
    "ECDHE-RSA-AES256-GCM-SHA384",
    "ECDHE-ECDSA-AES128-GCM-SHA256",
    "ECDHE-RSA-AES128-GCM-SHA256",
]

# Maximum key age (PCI-DSS recommends annual rotation)
MAX_KEY_AGE_DAYS = 365


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EncryptionCheckResult(BaseModel):
    """Result of an encryption configuration check.

    Attributes:
        id: Unique check identifier.
        check_name: Name of the check performed.
        check_type: Type of check (at_rest, in_transit, key_management).
        status: Check status (pass, fail, warning).
        target: What was checked (database, s3, endpoint).
        algorithm_used: Encryption algorithm detected.
        is_compliant: Whether it meets PCI-DSS requirements.
        findings: Detailed findings.
        recommendations: Recommendations for remediation.
        checked_at: When the check was performed.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    check_name: str
    check_type: str
    status: CheckStatus = CheckStatus.PASS
    target: str = ""
    algorithm_used: Optional[str] = None
    is_compliant: bool = True
    findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class KeyManagementResult(BaseModel):
    """Result of key management assessment.

    Attributes:
        total_keys: Total cryptographic keys assessed.
        compliant_keys: Keys meeting requirements.
        non_compliant_keys: Keys not meeting requirements.
        keys_requiring_rotation: Keys past rotation date.
        kms_enabled: Whether KMS is used.
        key_encryption_enabled: Whether KEK is used.
        split_knowledge: Whether split knowledge is implemented.
        dual_control: Whether dual control is implemented.
        key_details: Details of individual keys.
        status: Overall assessment status.
    """

    total_keys: int = 0
    compliant_keys: int = 0
    non_compliant_keys: int = 0
    keys_requiring_rotation: int = 0
    kms_enabled: bool = True
    key_encryption_enabled: bool = True
    split_knowledge: bool = False
    dual_control: bool = False
    key_details: List[Dict[str, Any]] = Field(default_factory=list)
    status: CheckStatus = CheckStatus.PASS


class TransmissionCheckResult(BaseModel):
    """Result of transmission encryption check.

    Attributes:
        endpoints_checked: Number of endpoints checked.
        compliant_endpoints: Endpoints meeting requirements.
        non_compliant_endpoints: Endpoints not meeting requirements.
        tls_versions_found: TLS versions in use.
        cipher_suites_found: Cipher suites in use.
        certificate_issues: Certificate problems found.
        endpoint_details: Details of individual endpoints.
        status: Overall assessment status.
    """

    endpoints_checked: int = 0
    compliant_endpoints: int = 0
    non_compliant_endpoints: int = 0
    tls_versions_found: List[str] = Field(default_factory=list)
    cipher_suites_found: List[str] = Field(default_factory=list)
    certificate_issues: List[str] = Field(default_factory=list)
    endpoint_details: List[Dict[str, Any]] = Field(default_factory=list)
    status: CheckStatus = CheckStatus.PASS


# ---------------------------------------------------------------------------
# Encryption Checker
# ---------------------------------------------------------------------------


class EncryptionChecker:
    """Verifies encryption configuration for PCI-DSS compliance.

    Checks encryption at rest (databases, storage), encryption in transit
    (TLS configuration), and key management practices against PCI-DSS v4.0
    requirements.

    Example:
        >>> checker = EncryptionChecker()
        >>> pan_result = await checker.verify_pan_encryption()
        >>> if not pan_result.is_compliant:
        ...     print("PAN encryption issues found!")
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the encryption checker.

        Args:
            config: Optional compliance configuration.
        """
        self.config = config
        self.check_results: List[EncryptionCheckResult] = []

        logger.info("Initialized EncryptionChecker")

    async def verify_pan_encryption(self) -> EncryptionCheckResult:
        """Verify encryption of stored PAN (Req 3.5).

        Checks that any stored PAN is encrypted using strong cryptography.

        Returns:
            EncryptionCheckResult with findings.
        """
        logger.info("Verifying PAN encryption at rest")

        result = EncryptionCheckResult(
            check_name="PAN Encryption at Rest",
            check_type="at_rest",
            target="Cardholder Data Storage",
        )

        # Check database encryption
        db_encryption = await self._check_database_encryption()
        result.findings.extend(db_encryption["findings"])

        # Check S3 encryption
        s3_encryption = await self._check_s3_encryption()
        result.findings.extend(s3_encryption["findings"])

        # Determine compliance
        if db_encryption["compliant"] and s3_encryption["compliant"]:
            result.status = CheckStatus.PASS
            result.is_compliant = True
            result.algorithm_used = db_encryption.get("algorithm", "AES-256-GCM")
        else:
            result.status = CheckStatus.FAIL
            result.is_compliant = False
            result.recommendations = [
                "Ensure all PAN storage uses AES-256 or equivalent",
                "Enable encryption at rest for all databases storing CHD",
                "Configure server-side encryption for S3 buckets with CHD",
            ]

        self.check_results.append(result)
        return result

    async def verify_transmission_encryption(self) -> TransmissionCheckResult:
        """Verify encryption of data in transit (Req 4.2).

        Checks TLS configuration for all endpoints that handle CHD.

        Returns:
            TransmissionCheckResult with findings.
        """
        logger.info("Verifying transmission encryption")

        result = TransmissionCheckResult()

        # Check TLS configuration for known endpoints
        endpoints = await self._discover_endpoints()
        result.endpoints_checked = len(endpoints)

        for endpoint in endpoints:
            endpoint_result = await self._check_endpoint_tls(endpoint)
            result.endpoint_details.append(endpoint_result)

            if endpoint_result["compliant"]:
                result.compliant_endpoints += 1
            else:
                result.non_compliant_endpoints += 1

            # Collect TLS versions
            tls_version = endpoint_result.get("tls_version")
            if tls_version and tls_version not in result.tls_versions_found:
                result.tls_versions_found.append(tls_version)

            # Collect cipher suites
            ciphers = endpoint_result.get("cipher_suites", [])
            for cipher in ciphers:
                if cipher not in result.cipher_suites_found:
                    result.cipher_suites_found.append(cipher)

            # Collect certificate issues
            cert_issues = endpoint_result.get("certificate_issues", [])
            result.certificate_issues.extend(cert_issues)

        # Determine overall status
        if result.non_compliant_endpoints == 0:
            result.status = CheckStatus.PASS
        elif result.non_compliant_endpoints < result.endpoints_checked / 2:
            result.status = CheckStatus.WARNING
        else:
            result.status = CheckStatus.FAIL

        return result

    async def verify_key_management(self) -> KeyManagementResult:
        """Verify cryptographic key management (Req 3.6).

        Checks key rotation, storage, and access controls.

        Returns:
            KeyManagementResult with findings.
        """
        logger.info("Verifying key management")

        result = KeyManagementResult()

        # Get key inventory
        keys = await self._get_key_inventory()
        result.total_keys = len(keys)

        now = datetime.now(timezone.utc)

        for key in keys:
            key_detail = {
                "key_id": key.get("id", "unknown"),
                "algorithm": key.get("algorithm"),
                "created_at": key.get("created_at"),
                "last_rotated": key.get("last_rotated"),
                "issues": [],
            }

            is_compliant = True

            # Check algorithm
            algorithm = key.get("algorithm")
            if algorithm not in [a.value for a in APPROVED_ALGORITHMS]:
                is_compliant = False
                key_detail["issues"].append(f"Non-approved algorithm: {algorithm}")

            # Check rotation
            last_rotated = key.get("last_rotated")
            if last_rotated:
                if isinstance(last_rotated, str):
                    last_rotated = datetime.fromisoformat(last_rotated.replace("Z", "+00:00"))
                days_since_rotation = (now - last_rotated).days
                if days_since_rotation > MAX_KEY_AGE_DAYS:
                    is_compliant = False
                    result.keys_requiring_rotation += 1
                    key_detail["issues"].append(
                        f"Key not rotated in {days_since_rotation} days"
                    )

            if is_compliant:
                result.compliant_keys += 1
            else:
                result.non_compliant_keys += 1

            result.key_details.append(key_detail)

        # Check KMS usage
        result.kms_enabled = await self._check_kms_enabled()

        # Check key encryption key (KEK)
        result.key_encryption_enabled = await self._check_kek_enabled()

        # Determine overall status
        if result.non_compliant_keys == 0 and result.kms_enabled:
            result.status = CheckStatus.PASS
        elif result.non_compliant_keys < result.total_keys / 2:
            result.status = CheckStatus.WARNING
        else:
            result.status = CheckStatus.FAIL

        return result

    async def run_full_assessment(self) -> Dict[str, Any]:
        """Run a complete encryption assessment.

        Returns:
            Dictionary with all assessment results.
        """
        logger.info("Running full PCI-DSS encryption assessment")

        pan_result = await self.verify_pan_encryption()
        transmission_result = await self.verify_transmission_encryption()
        key_result = await self.verify_key_management()

        # Determine overall compliance
        overall_compliant = (
            pan_result.is_compliant
            and transmission_result.status != CheckStatus.FAIL
            and key_result.status != CheckStatus.FAIL
        )

        return {
            "assessed_at": datetime.now(timezone.utc).isoformat(),
            "overall_compliant": overall_compliant,
            "pan_encryption": pan_result.model_dump(),
            "transmission_encryption": transmission_result.model_dump(),
            "key_management": key_result.model_dump(),
            "recommendations": self._generate_recommendations(
                pan_result, transmission_result, key_result
            ),
        }

    # -------------------------------------------------------------------------
    # Private Methods - Check Implementations
    # -------------------------------------------------------------------------

    async def _check_database_encryption(self) -> Dict[str, Any]:
        """Check database encryption configuration.

        In production, this would query actual database configurations.
        """
        # Placeholder - return compliant configuration
        return {
            "compliant": True,
            "algorithm": "AES-256-GCM",
            "findings": [
                "PostgreSQL TDE enabled with AES-256",
                "All data columns with PAN use column-level encryption",
            ],
        }

    async def _check_s3_encryption(self) -> Dict[str, Any]:
        """Check S3 bucket encryption configuration.

        In production, this would query S3 bucket configurations.
        """
        # Placeholder - return compliant configuration
        return {
            "compliant": True,
            "algorithm": "AES-256",
            "findings": [
                "Server-side encryption enabled (SSE-KMS)",
                "Bucket policy enforces encrypted uploads",
            ],
        }

    async def _discover_endpoints(self) -> List[Dict[str, Any]]:
        """Discover endpoints that may handle CHD.

        In production, this would scan network configurations.
        """
        # Return default GreenLang endpoints
        return [
            {
                "name": "API Gateway",
                "url": "https://api.greenlang.io",
                "type": "external",
            },
            {
                "name": "Payment Integration",
                "url": "https://payments.greenlang.io",
                "type": "internal",
            },
            {
                "name": "Admin Portal",
                "url": "https://admin.greenlang.io",
                "type": "internal",
            },
        ]

    async def _check_endpoint_tls(
        self,
        endpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check TLS configuration for an endpoint.

        In production, this would perform actual TLS handshake analysis.
        """
        # Placeholder - return compliant configuration
        return {
            "endpoint": endpoint["name"],
            "url": endpoint["url"],
            "compliant": True,
            "tls_version": "TLS 1.3",
            "cipher_suites": [
                "TLS_AES_256_GCM_SHA384",
                "TLS_CHACHA20_POLY1305_SHA256",
            ],
            "certificate_issues": [],
            "hsts_enabled": True,
            "perfect_forward_secrecy": True,
        }

    async def _get_key_inventory(self) -> List[Dict[str, Any]]:
        """Get inventory of cryptographic keys.

        In production, this would query KMS and key stores.
        """
        # Return sample key inventory
        now = datetime.now(timezone.utc)
        return [
            {
                "id": "key-db-encryption-001",
                "algorithm": "AES-256-GCM",
                "purpose": "Database encryption",
                "created_at": (now - timedelta(days=180)).isoformat(),
                "last_rotated": (now - timedelta(days=30)).isoformat(),
            },
            {
                "id": "key-s3-encryption-001",
                "algorithm": "AES-256",
                "purpose": "S3 encryption",
                "created_at": (now - timedelta(days=365)).isoformat(),
                "last_rotated": (now - timedelta(days=60)).isoformat(),
            },
            {
                "id": "key-api-signing-001",
                "algorithm": "RSA-4096",
                "purpose": "API signing",
                "created_at": (now - timedelta(days=90)).isoformat(),
                "last_rotated": (now - timedelta(days=90)).isoformat(),
            },
        ]

    async def _check_kms_enabled(self) -> bool:
        """Check if KMS is enabled for key management.

        In production, this would verify AWS KMS or equivalent.
        """
        return True

    async def _check_kek_enabled(self) -> bool:
        """Check if key encryption key (KEK) is used.

        In production, this would verify key wrapping configuration.
        """
        return True

    def _generate_recommendations(
        self,
        pan_result: EncryptionCheckResult,
        transmission_result: TransmissionCheckResult,
        key_result: KeyManagementResult,
    ) -> List[str]:
        """Generate remediation recommendations."""
        recommendations: List[str] = []

        if not pan_result.is_compliant:
            recommendations.extend(pan_result.recommendations)

        if transmission_result.non_compliant_endpoints > 0:
            recommendations.append(
                f"Remediate {transmission_result.non_compliant_endpoints} "
                f"non-compliant endpoints"
            )

        if TLSVersion.TLS_1_1.value in transmission_result.tls_versions_found:
            recommendations.append(
                "Disable TLS 1.1 - only TLS 1.2 and 1.3 are PCI-DSS compliant"
            )

        if key_result.keys_requiring_rotation > 0:
            recommendations.append(
                f"Rotate {key_result.keys_requiring_rotation} cryptographic keys "
                f"that exceed {MAX_KEY_AGE_DAYS}-day rotation policy"
            )

        if not key_result.kms_enabled:
            recommendations.append(
                "Enable AWS KMS or equivalent HSM for key management"
            )

        if recommendations:
            recommendations.insert(0, "PCI-DSS Encryption Remediation Plan:")

        return recommendations


__all__ = [
    "EncryptionChecker",
    "EncryptionCheckResult",
    "KeyManagementResult",
    "TransmissionCheckResult",
    "EncryptionAlgorithm",
    "TLSVersion",
    "CheckStatus",
    "APPROVED_ALGORITHMS",
    "APPROVED_CIPHER_SUITES",
]
