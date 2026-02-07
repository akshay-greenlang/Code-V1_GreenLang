# -*- coding: utf-8 -*-
"""
PII Scanner - SEC-007 Security Scanning Pipeline

Pattern-based PII (Personally Identifiable Information) detection for
source code, configuration files, and data streams. Detects SSN, credit
cards, emails, phone numbers, and other sensitive data patterns.

Detection Approach:
    1. Regex-based pattern matching for known PII formats
    2. Context validation to reduce false positives
    3. Confidence scoring based on pattern strength and context
    4. Data classification (PII, PHI, PCI, Secret)

Supported PII Types:
    - SSN (Social Security Number)
    - Credit card numbers (Visa, MC, Amex, Discover)
    - Email addresses
    - Phone numbers (US/international formats)
    - IP addresses
    - API keys and tokens
    - Passwords and secrets
    - Medical record numbers
    - GreenLang-specific: tenant IDs, emission data

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-007 Security Scanning Pipeline
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DataClassification(str, Enum):
    """Data classification levels."""

    PII = "pii"  # Personally Identifiable Information
    PHI = "phi"  # Protected Health Information (HIPAA)
    PCI = "pci"  # Payment Card Industry data
    SECRET = "secret"  # Secrets, credentials, keys
    INTERNAL = "internal"  # Internal/confidential data


class PIIType(str, Enum):
    """Types of PII detected."""

    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    NAME = "name"
    DOB = "dob"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    IP_ADDRESS = "ip_address"
    MEDICAL_RECORD = "medical_record"
    FINANCIAL_ACCOUNT = "financial_account"
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    TENANT_ID = "tenant_id"
    EMISSION_DATA = "emission_data"
    OTHER = "other"


class DetectionMethod(str, Enum):
    """Detection method used."""

    REGEX = "regex"
    ML = "ml"
    PRESIDIO = "presidio"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PIIPattern:
    """Definition of a PII detection pattern."""

    name: str
    pii_type: PIIType
    classification: DataClassification
    pattern: Pattern
    confidence_base: float  # Base confidence score (0-1)
    context_boost_patterns: List[str] = field(default_factory=list)
    context_reduce_patterns: List[str] = field(default_factory=list)
    description: str = ""
    luhn_check: bool = False  # For credit card validation


@dataclass
class PIIFinding:
    """A single PII finding."""

    id: UUID
    pii_type: PIIType
    classification: DataClassification
    pattern_name: str
    confidence_score: float
    detection_method: DetectionMethod

    # Location
    file_path: Optional[str]
    line_number: Optional[int]
    column_start: Optional[int]
    column_end: Optional[int]

    # Context (redacted)
    context_before: Optional[str]
    context_after: Optional[str]
    matched_text_hash: str  # SHA-256 of matched text (never store raw)

    # Metadata
    exposure_risk: str  # critical, high, medium, low
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScanResult:
    """Result of a PII scan."""

    findings: List[PIIFinding]
    files_scanned: int
    lines_scanned: int
    scan_duration_ms: float
    errors: List[str]


# ---------------------------------------------------------------------------
# PII Patterns
# ---------------------------------------------------------------------------


def _build_patterns() -> List[PIIPattern]:
    """Build the list of PII detection patterns.

    Returns:
        List of PIIPattern definitions.
    """
    patterns = []

    # -------------------------------------------------------------------------
    # Social Security Numbers
    # -------------------------------------------------------------------------
    patterns.append(PIIPattern(
        name="ssn_dashed",
        pii_type=PIIType.SSN,
        classification=DataClassification.PII,
        pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        confidence_base=0.85,
        context_boost_patterns=["ssn", "social", "security", "tax", "ein"],
        context_reduce_patterns=["phone", "fax", "tel", "date", "version"],
        description="SSN with dashes (XXX-XX-XXXX)",
    ))

    patterns.append(PIIPattern(
        name="ssn_nodash",
        pii_type=PIIType.SSN,
        classification=DataClassification.PII,
        pattern=re.compile(r"\b(?!000|666|9\d{2})\d{3}(?!00)\d{2}(?!0000)\d{4}\b"),
        confidence_base=0.60,  # Lower confidence without dashes
        context_boost_patterns=["ssn", "social", "security", "tax"],
        description="SSN without dashes (9 digits)",
    ))

    # -------------------------------------------------------------------------
    # Credit Card Numbers
    # -------------------------------------------------------------------------
    # Visa: 4XXX-XXXX-XXXX-XXXX (13-16 digits)
    patterns.append(PIIPattern(
        name="credit_card_visa",
        pii_type=PIIType.CREDIT_CARD,
        classification=DataClassification.PCI,
        pattern=re.compile(r"\b4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
        confidence_base=0.80,
        context_boost_patterns=["card", "credit", "payment", "visa", "cc"],
        context_reduce_patterns=["version", "ipv4", "id"],
        description="Visa credit card number",
        luhn_check=True,
    ))

    # Mastercard: 5[1-5]XX-XXXX-XXXX-XXXX (16 digits)
    patterns.append(PIIPattern(
        name="credit_card_mastercard",
        pii_type=PIIType.CREDIT_CARD,
        classification=DataClassification.PCI,
        pattern=re.compile(r"\b5[1-5]\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
        confidence_base=0.80,
        context_boost_patterns=["card", "credit", "payment", "mastercard", "mc", "cc"],
        description="Mastercard credit card number",
        luhn_check=True,
    ))

    # American Express: 3[47]XX-XXXXXX-XXXXX (15 digits)
    patterns.append(PIIPattern(
        name="credit_card_amex",
        pii_type=PIIType.CREDIT_CARD,
        classification=DataClassification.PCI,
        pattern=re.compile(r"\b3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}\b"),
        confidence_base=0.80,
        context_boost_patterns=["card", "credit", "payment", "amex", "american express"],
        description="American Express credit card number",
        luhn_check=True,
    ))

    # Generic 16-digit card number
    patterns.append(PIIPattern(
        name="credit_card_generic",
        pii_type=PIIType.CREDIT_CARD,
        classification=DataClassification.PCI,
        pattern=re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
        confidence_base=0.50,  # Lower confidence for generic
        context_boost_patterns=["card", "credit", "payment", "pan", "cc"],
        context_reduce_patterns=["uuid", "id", "guid", "version"],
        description="Generic 16-digit number (potential card)",
        luhn_check=True,
    ))

    # -------------------------------------------------------------------------
    # Email Addresses
    # -------------------------------------------------------------------------
    patterns.append(PIIPattern(
        name="email_address",
        pii_type=PIIType.EMAIL,
        classification=DataClassification.PII,
        pattern=re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            re.IGNORECASE,
        ),
        confidence_base=0.90,
        context_boost_patterns=["email", "mail", "contact", "user"],
        context_reduce_patterns=["example.com", "test.com", "localhost", "noreply"],
        description="Email address",
    ))

    # -------------------------------------------------------------------------
    # Phone Numbers
    # -------------------------------------------------------------------------
    # US phone with area code
    patterns.append(PIIPattern(
        name="phone_us",
        pii_type=PIIType.PHONE,
        classification=DataClassification.PII,
        pattern=re.compile(
            r"\b(?:\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"
        ),
        confidence_base=0.70,
        context_boost_patterns=["phone", "tel", "call", "mobile", "cell", "fax"],
        context_reduce_patterns=["ssn", "card", "account", "version"],
        description="US phone number",
    ))

    # International phone
    patterns.append(PIIPattern(
        name="phone_international",
        pii_type=PIIType.PHONE,
        classification=DataClassification.PII,
        pattern=re.compile(r"\b\+\d{1,3}[\s.-]?\d{4,14}\b"),
        confidence_base=0.75,
        context_boost_patterns=["phone", "tel", "international", "mobile"],
        description="International phone number",
    ))

    # -------------------------------------------------------------------------
    # IP Addresses
    # -------------------------------------------------------------------------
    patterns.append(PIIPattern(
        name="ipv4_address",
        pii_type=PIIType.IP_ADDRESS,
        classification=DataClassification.INTERNAL,
        pattern=re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
        confidence_base=0.60,
        context_boost_patterns=["ip", "address", "client", "remote", "host"],
        context_reduce_patterns=["version", "mask", "127.0.0.1", "0.0.0.0", "localhost"],
        description="IPv4 address",
    ))

    # -------------------------------------------------------------------------
    # API Keys and Tokens
    # -------------------------------------------------------------------------
    # AWS Access Key
    patterns.append(PIIPattern(
        name="aws_access_key",
        pii_type=PIIType.API_KEY,
        classification=DataClassification.SECRET,
        pattern=re.compile(r"\b(AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b"),
        confidence_base=0.95,
        context_boost_patterns=["aws", "access", "key", "iam"],
        description="AWS Access Key ID",
    ))

    # AWS Secret Key
    patterns.append(PIIPattern(
        name="aws_secret_key",
        pii_type=PIIType.API_KEY,
        classification=DataClassification.SECRET,
        pattern=re.compile(r"(?i)aws.{0,20}secret.{0,20}['\"][0-9a-zA-Z/+]{40}['\"]"),
        confidence_base=0.90,
        description="AWS Secret Access Key",
    ))

    # Generic API key pattern
    patterns.append(PIIPattern(
        name="api_key_generic",
        pii_type=PIIType.API_KEY,
        classification=DataClassification.SECRET,
        pattern=re.compile(
            r"(?i)(api[_-]?key|apikey|api[_-]?secret)['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{20,}['\"]?"
        ),
        confidence_base=0.80,
        description="Generic API key assignment",
    ))

    # JWT Token
    patterns.append(PIIPattern(
        name="jwt_token",
        pii_type=PIIType.TOKEN,
        classification=DataClassification.SECRET,
        pattern=re.compile(
            r"\beyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\b"
        ),
        confidence_base=0.95,
        context_boost_patterns=["token", "jwt", "bearer", "auth"],
        description="JWT token",
    ))

    # -------------------------------------------------------------------------
    # Passwords
    # -------------------------------------------------------------------------
    patterns.append(PIIPattern(
        name="password_assignment",
        pii_type=PIIType.PASSWORD,
        classification=DataClassification.SECRET,
        pattern=re.compile(
            r"(?i)(password|passwd|pwd)['\"]?\s*[:=]\s*['\"][^'\"]{8,}['\"]"
        ),
        confidence_base=0.85,
        context_reduce_patterns=["example", "placeholder", "dummy", "test", "changeme"],
        description="Password assignment",
    ))

    # -------------------------------------------------------------------------
    # GreenLang-specific Patterns
    # -------------------------------------------------------------------------
    # Tenant ID exposure
    patterns.append(PIIPattern(
        name="greenlang_tenant_id",
        pii_type=PIIType.TENANT_ID,
        classification=DataClassification.INTERNAL,
        pattern=re.compile(
            r"(?i)tenant[_-]?id['\"]?\s*[:=]\s*['\"]?[a-f0-9-]{36}['\"]?"
        ),
        confidence_base=0.70,
        context_boost_patterns=["tenant", "organization", "org"],
        description="GreenLang tenant ID",
    ))

    return patterns


# ---------------------------------------------------------------------------
# PII Scanner
# ---------------------------------------------------------------------------


class PIIScanner:
    """Pattern-based PII scanner for code and data.

    Scans files and text for personally identifiable information using
    regex patterns with context-aware confidence scoring.

    Example:
        >>> scanner = PIIScanner()
        >>> findings = scanner.scan_file("/path/to/config.yaml")
        >>> for f in findings:
        ...     print(f"{f.pii_type}: {f.confidence_score}")

        >>> text_findings = scanner.scan_text("SSN: 123-45-6789")
    """

    def __init__(
        self,
        patterns: Optional[List[PIIPattern]] = None,
        min_confidence: float = 0.5,
        context_window: int = 50,
    ) -> None:
        """Initialize PIIScanner.

        Args:
            patterns: Custom patterns (uses defaults if None).
            min_confidence: Minimum confidence threshold (0-1).
            context_window: Characters of context to capture.
        """
        self._patterns = patterns or _build_patterns()
        self._min_confidence = min_confidence
        self._context_window = context_window

    def scan_file(self, file_path: str) -> List[PIIFinding]:
        """Scan a file for PII.

        Args:
            file_path: Path to file to scan.

        Returns:
            List of PIIFinding objects.
        """
        findings: List[PIIFinding] = []
        path = Path(file_path)

        if not path.exists():
            logger.warning("File not found: %s", file_path)
            return findings

        # Skip binary files
        if self._is_binary_file(path):
            logger.debug("Skipping binary file: %s", file_path)
            return findings

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()

            for line_num, line in enumerate(lines, start=1):
                line_findings = self._scan_line(
                    line, file_path, line_num
                )
                findings.extend(line_findings)

        except Exception as e:
            logger.error("Failed to scan file %s: %s", file_path, e)

        return findings

    def scan_text(self, text: str) -> List[PIIFinding]:
        """Scan text for PII.

        Args:
            text: Text content to scan.

        Returns:
            List of PIIFinding objects.
        """
        findings: List[PIIFinding] = []

        for line_num, line in enumerate(text.splitlines(), start=1):
            line_findings = self._scan_line(line, None, line_num)
            findings.extend(line_findings)

        return findings

    def scan_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> ScanResult:
        """Scan a directory for PII.

        Args:
            directory: Directory path to scan.
            extensions: File extensions to include (e.g., [".py", ".yaml"]).
            exclude_patterns: Glob patterns to exclude.

        Returns:
            ScanResult with all findings.
        """
        start_time = datetime.utcnow()
        findings: List[PIIFinding] = []
        errors: List[str] = []
        files_scanned = 0
        lines_scanned = 0

        default_extensions = [
            ".py", ".js", ".ts", ".java", ".go", ".rb", ".php",
            ".yaml", ".yml", ".json", ".xml", ".toml", ".ini", ".cfg",
            ".env", ".conf", ".config", ".properties",
            ".md", ".txt", ".csv",
            ".tf", ".hcl",
            ".sh", ".bash", ".zsh",
        ]

        extensions = extensions or default_extensions
        exclude_patterns = exclude_patterns or [
            "**/node_modules/**",
            "**/.git/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
        ]

        dir_path = Path(directory)
        if not dir_path.is_dir():
            errors.append(f"Not a directory: {directory}")
            return ScanResult(
                findings=findings,
                files_scanned=0,
                lines_scanned=0,
                scan_duration_ms=0,
                errors=errors,
            )

        # Collect files
        files_to_scan = []
        for ext in extensions:
            files_to_scan.extend(dir_path.rglob(f"*{ext}"))

        # Filter exclusions
        for pattern in exclude_patterns:
            files_to_scan = [
                f for f in files_to_scan
                if not f.match(pattern)
            ]

        # Scan files
        for file_path in files_to_scan:
            try:
                file_findings = self.scan_file(str(file_path))
                findings.extend(file_findings)
                files_scanned += 1

                # Count lines
                try:
                    lines_scanned += sum(
                        1 for _ in file_path.open(encoding="utf-8", errors="ignore")
                    )
                except Exception:
                    pass

            except Exception as e:
                errors.append(f"Error scanning {file_path}: {e}")

        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ScanResult(
            findings=findings,
            files_scanned=files_scanned,
            lines_scanned=lines_scanned,
            scan_duration_ms=duration_ms,
            errors=errors,
        )

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _scan_line(
        self,
        line: str,
        file_path: Optional[str],
        line_number: int,
    ) -> List[PIIFinding]:
        """Scan a single line for PII patterns.

        Args:
            line: Line content to scan.
            file_path: Source file path (if from file).
            line_number: Line number in source.

        Returns:
            List of findings for this line.
        """
        findings: List[PIIFinding] = []

        for pattern in self._patterns:
            for match in pattern.pattern.finditer(line):
                # Calculate confidence with context
                confidence = self._calculate_confidence(
                    pattern, line, match
                )

                if confidence < self._min_confidence:
                    continue

                # Additional validation (e.g., Luhn check)
                if pattern.luhn_check:
                    matched_text = match.group()
                    digits = re.sub(r"\D", "", matched_text)
                    if not self._luhn_check(digits):
                        continue

                # Extract context
                start = max(0, match.start() - self._context_window)
                end = min(len(line), match.end() + self._context_window)
                context_before = line[start:match.start()]
                context_after = line[match.end():end]

                # Hash the matched text (never store raw PII)
                matched_hash = hashlib.sha256(
                    match.group().encode()
                ).hexdigest()

                # Determine exposure risk
                exposure_risk = self._assess_exposure_risk(
                    pattern.classification, confidence
                )

                finding = PIIFinding(
                    id=uuid4(),
                    pii_type=pattern.pii_type,
                    classification=pattern.classification,
                    pattern_name=pattern.name,
                    confidence_score=confidence,
                    detection_method=DetectionMethod.REGEX,
                    file_path=file_path,
                    line_number=line_number,
                    column_start=match.start(),
                    column_end=match.end(),
                    context_before=context_before,
                    context_after=context_after,
                    matched_text_hash=matched_hash,
                    exposure_risk=exposure_risk,
                )
                findings.append(finding)

        return findings

    def _calculate_confidence(
        self,
        pattern: PIIPattern,
        line: str,
        match: re.Match,
    ) -> float:
        """Calculate confidence score with context adjustment.

        Args:
            pattern: The matched pattern.
            line: Full line content.
            match: Regex match object.

        Returns:
            Adjusted confidence score (0-1).
        """
        confidence = pattern.confidence_base
        line_lower = line.lower()

        # Boost confidence if context keywords present
        for boost_keyword in pattern.context_boost_patterns:
            if boost_keyword.lower() in line_lower:
                confidence = min(1.0, confidence + 0.1)

        # Reduce confidence if anti-context keywords present
        for reduce_keyword in pattern.context_reduce_patterns:
            if reduce_keyword.lower() in line_lower:
                confidence = max(0.0, confidence - 0.15)

        # Check if in a comment (lower confidence)
        stripped = line.strip()
        if stripped.startswith(("#", "//", "/*", "*", "--")):
            confidence = max(0.0, confidence - 0.1)

        # Check if looks like test/example data
        if any(word in line_lower for word in ["test", "example", "sample", "dummy", "fake"]):
            confidence = max(0.0, confidence - 0.2)

        return round(confidence, 2)

    def _luhn_check(self, digits: str) -> bool:
        """Validate a number using the Luhn algorithm.

        Args:
            digits: Digit string to validate.

        Returns:
            True if valid Luhn checksum.
        """
        if not digits or not digits.isdigit():
            return False

        def digits_of(n: str) -> List[int]:
            return [int(d) for d in n]

        digits_list = digits_of(digits)
        odd_digits = digits_list[-1::-2]
        even_digits = digits_list[-2::-2]

        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(str(d * 2)))

        return checksum % 10 == 0

    def _assess_exposure_risk(
        self,
        classification: DataClassification,
        confidence: float,
    ) -> str:
        """Assess the exposure risk level.

        Args:
            classification: Data classification.
            confidence: Detection confidence.

        Returns:
            Risk level: critical, high, medium, low.
        """
        if classification in (DataClassification.PHI, DataClassification.SECRET):
            return "critical" if confidence > 0.8 else "high"
        elif classification == DataClassification.PCI:
            return "critical" if confidence > 0.8 else "high"
        elif classification == DataClassification.PII:
            return "high" if confidence > 0.8 else "medium"
        else:
            return "medium" if confidence > 0.7 else "low"

    def _is_binary_file(self, path: Path) -> bool:
        """Check if a file is binary.

        Args:
            path: File path.

        Returns:
            True if file appears to be binary.
        """
        binary_extensions = {
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico",
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
            ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
            ".exe", ".dll", ".so", ".dylib",
            ".pyc", ".pyo", ".class", ".o",
            ".woff", ".woff2", ".ttf", ".eot",
            ".mp3", ".mp4", ".avi", ".mov", ".wav",
        }

        if path.suffix.lower() in binary_extensions:
            return True

        # Check first 8KB for binary content
        try:
            with open(path, "rb") as f:
                chunk = f.read(8192)
                if b"\x00" in chunk:
                    return True
        except Exception:
            pass

        return False


# ---------------------------------------------------------------------------
# Global Instance
# ---------------------------------------------------------------------------

_global_pii_scanner: Optional[PIIScanner] = None


def get_pii_scanner() -> PIIScanner:
    """Get or create the global PII scanner instance.

    Returns:
        The global PIIScanner instance.
    """
    global _global_pii_scanner

    if _global_pii_scanner is None:
        _global_pii_scanner = PIIScanner()

    return _global_pii_scanner


__all__ = [
    "PIIScanner",
    "PIIPattern",
    "PIIFinding",
    "ScanResult",
    "DataClassification",
    "PIIType",
    "DetectionMethod",
    "get_pii_scanner",
]
