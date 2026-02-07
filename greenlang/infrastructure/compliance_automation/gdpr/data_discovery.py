# -*- coding: utf-8 -*-
"""
GDPR Data Discovery - SEC-010 Phase 5

Discovers and inventories personal data (PII) across GreenLang systems.
Scans databases, object storage, logs, and caches to create a complete
data map for DSAR processing and compliance reporting.

Supported Data Sources:
- PostgreSQL databases
- S3/Object storage
- Application logs (Loki)
- Redis caches
- TimescaleDB time-series data

Classes:
    - DataDiscovery: Main data discovery engine.
    - ScanResult: Result of a data source scan.
    - PIIDetector: Detects PII in data records.

Example:
    >>> discovery = DataDiscovery()
    >>> records = await discovery.discover_user_data(
    ...     user_email="user@example.com",
    ...     user_id="user-123",
    ... )
    >>> inventory = await discovery.generate_data_inventory()

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

from greenlang.infrastructure.compliance_automation.models import (
    DataCategory,
    DataRecord,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PII Detection Patterns
# ---------------------------------------------------------------------------


class PIIPattern:
    """Regular expression patterns for PII detection."""

    # Email addresses
    EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

    # Phone numbers (international)
    PHONE = re.compile(r"\+?[\d\s\-\(\)]{10,}")

    # Credit card numbers (basic pattern)
    CREDIT_CARD = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

    # Social Security Numbers (US)
    SSN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")

    # IP addresses
    IP_ADDRESS = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )

    # Date of birth patterns
    DATE_OF_BIRTH = re.compile(
        r"\b(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])\b"
    )


# PII field name patterns
PII_FIELD_PATTERNS = [
    re.compile(r".*email.*", re.IGNORECASE),
    re.compile(r".*phone.*", re.IGNORECASE),
    re.compile(r".*address.*", re.IGNORECASE),
    re.compile(r".*name.*", re.IGNORECASE),
    re.compile(r".*ssn.*", re.IGNORECASE),
    re.compile(r".*social.*security.*", re.IGNORECASE),
    re.compile(r".*dob.*", re.IGNORECASE),
    re.compile(r".*birth.*date.*", re.IGNORECASE),
    re.compile(r".*ip.*address.*", re.IGNORECASE),
    re.compile(r".*credit.*card.*", re.IGNORECASE),
    re.compile(r".*passport.*", re.IGNORECASE),
    re.compile(r".*driver.*license.*", re.IGNORECASE),
]

# Sensitive PII categories
SENSITIVE_PII_PATTERNS = [
    re.compile(r".*race.*", re.IGNORECASE),
    re.compile(r".*ethnic.*", re.IGNORECASE),
    re.compile(r".*religion.*", re.IGNORECASE),
    re.compile(r".*political.*", re.IGNORECASE),
    re.compile(r".*health.*", re.IGNORECASE),
    re.compile(r".*medical.*", re.IGNORECASE),
    re.compile(r".*genetic.*", re.IGNORECASE),
    re.compile(r".*biometric.*", re.IGNORECASE),
    re.compile(r".*sexual.*", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Result Models
# ---------------------------------------------------------------------------


class ScanResult(BaseModel):
    """Result of scanning a data source.

    Attributes:
        source_system: The system that was scanned.
        source_location: Specific location scanned.
        records_found: Number of records found.
        pii_detected: Whether PII was detected.
        scan_duration_ms: Duration of scan in milliseconds.
        errors: Any errors encountered.
    """

    source_system: str
    source_location: str = ""
    records_found: int = 0
    pii_detected: bool = False
    scan_duration_ms: float = 0.0
    errors: List[str] = Field(default_factory=list)


class DataInventoryItem(BaseModel):
    """Item in the data inventory (ROPA support).

    Attributes:
        id: Unique inventory item ID.
        data_category: Category of data.
        source_system: System where data is stored.
        source_location: Specific location.
        record_count: Estimated record count.
        pii_fields: List of PII fields present.
        retention_days: Retention period in days.
        purpose: Purpose of processing.
        legal_basis: Legal basis for processing.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    data_category: DataCategory
    source_system: str
    source_location: str
    record_count: int = 0
    pii_fields: List[str] = Field(default_factory=list)
    retention_days: int = 365
    purpose: str = ""
    legal_basis: str = "legitimate_interest"


# ---------------------------------------------------------------------------
# PII Detector
# ---------------------------------------------------------------------------


class PIIDetector:
    """Detects PII in data records and text.

    Provides methods to scan field names and values for personally
    identifiable information.

    Example:
        >>> detector = PIIDetector()
        >>> pii_fields = detector.detect_pii_fields({"email": "test@example.com"})
        >>> print(pii_fields)  # ["email"]
    """

    def detect_pii_fields(self, data: Dict[str, Any]) -> List[str]:
        """Detect fields containing PII based on field names.

        Args:
            data: Dictionary of field names and values.

        Returns:
            List of field names likely to contain PII.
        """
        pii_fields: List[str] = []

        for field_name in data.keys():
            for pattern in PII_FIELD_PATTERNS:
                if pattern.match(field_name):
                    pii_fields.append(field_name)
                    break

        return pii_fields

    def detect_pii_values(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text content.

        Args:
            text: Text to scan for PII.

        Returns:
            Dictionary mapping PII types to found values.
        """
        results: Dict[str, List[str]] = {}

        # Check for emails
        emails = PIIPattern.EMAIL.findall(text)
        if emails:
            results["email"] = emails[:5]  # Limit to 5

        # Check for phones
        phones = PIIPattern.PHONE.findall(text)
        if phones:
            results["phone"] = [p.strip() for p in phones[:5]]

        # Check for credit cards
        cards = PIIPattern.CREDIT_CARD.findall(text)
        if cards:
            results["credit_card"] = ["REDACTED" for _ in cards[:5]]

        # Check for SSNs
        ssns = PIIPattern.SSN.findall(text)
        if ssns:
            results["ssn"] = ["REDACTED" for _ in ssns[:5]]

        # Check for IPs
        ips = PIIPattern.IP_ADDRESS.findall(text)
        if ips:
            results["ip_address"] = ips[:5]

        return results

    def is_sensitive_pii(self, field_name: str) -> bool:
        """Check if a field contains sensitive PII.

        Sensitive PII includes race, religion, health, etc.
        (GDPR Article 9 special categories).

        Args:
            field_name: The field name to check.

        Returns:
            True if the field likely contains sensitive PII.
        """
        for pattern in SENSITIVE_PII_PATTERNS:
            if pattern.match(field_name):
                return True
        return False

    def categorize_data(self, data: Dict[str, Any]) -> DataCategory:
        """Categorize data based on content.

        Args:
            data: Dictionary of field names and values.

        Returns:
            The appropriate DataCategory.
        """
        field_names = list(data.keys())

        # Check for sensitive PII first
        for field_name in field_names:
            if self.is_sensitive_pii(field_name):
                return DataCategory.SENSITIVE_PII

        # Check for regular PII
        pii_fields = self.detect_pii_fields(data)
        if pii_fields:
            return DataCategory.PII

        # Check for financial data
        financial_patterns = ["payment", "transaction", "invoice", "credit", "debit"]
        for field_name in field_names:
            for pattern in financial_patterns:
                if pattern in field_name.lower():
                    return DataCategory.FINANCIAL

        # Default to operational
        return DataCategory.OPERATIONAL


# ---------------------------------------------------------------------------
# Data Discovery Engine
# ---------------------------------------------------------------------------


class DataDiscovery:
    """Discovers personal data across GreenLang systems.

    Scans databases, object storage, logs, and caches to create a complete
    inventory of personal data for DSAR processing and ROPA compliance.

    Attributes:
        pii_detector: PII detection engine.
        scan_results: Results from completed scans.

    Example:
        >>> discovery = DataDiscovery()
        >>> records = await discovery.discover_user_data(
        ...     user_email="user@example.com"
        ... )
        >>> print(f"Found {len(records)} records")
    """

    # Database tables known to contain user data
    USER_DATA_TABLES = [
        ("security", "users", "user_id"),
        ("security", "user_sessions", "user_id"),
        ("security", "audit_logs", "user_id"),
        ("security_ops", "dsar_requests", "subject_id"),
        ("security_ops", "consent_records", "user_id"),
        ("public", "user_profiles", "user_id"),
        ("public", "user_preferences", "user_id"),
        ("greenlang", "emission_reports", "created_by"),
        ("greenlang", "supply_chain_data", "owner_id"),
    ]

    # S3 prefixes known to contain user data
    USER_DATA_PREFIXES = [
        "user-uploads/",
        "exports/",
        "reports/user/",
        "avatars/",
        "documents/",
    ]

    # Log sources to scan
    LOG_SOURCES = [
        "application",
        "audit",
        "security",
        "access",
    ]

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the data discovery engine.

        Args:
            config: Optional compliance configuration.
        """
        self.config = config
        self.pii_detector = PIIDetector()
        self.scan_results: List[ScanResult] = []
        logger.info("Initialized DataDiscovery engine")

    async def discover_user_data(
        self,
        user_email: str,
        user_id: Optional[str] = None,
    ) -> List[DataRecord]:
        """Discover all data for a specific user.

        Scans all configured data sources for records belonging to the user.

        Args:
            user_email: User's email address.
            user_id: Optional user ID for more precise matching.

        Returns:
            List of discovered data records.
        """
        logger.info(
            "Discovering data for user: email=%s, id=%s",
            user_email,
            user_id or "unknown",
        )

        records: List[DataRecord] = []

        # Scan databases
        db_records = await self.scan_databases(user_email, user_id)
        records.extend(db_records)

        # Scan object storage
        s3_records = await self.scan_object_storage(user_email, user_id)
        records.extend(s3_records)

        # Scan logs
        log_records = await self.scan_logs(user_email, user_id)
        records.extend(log_records)

        # Scan caches
        cache_records = await self._scan_caches(user_email, user_id)
        records.extend(cache_records)

        logger.info(
            "Data discovery complete: found %d records for %s",
            len(records),
            user_email,
        )

        return records

    async def scan_databases(
        self,
        user_email: str,
        user_id: Optional[str] = None,
    ) -> List[DataRecord]:
        """Scan PostgreSQL databases for user data.

        Args:
            user_email: User's email address.
            user_id: Optional user ID.

        Returns:
            List of discovered database records.
        """
        start_time = datetime.now(timezone.utc)
        records: List[DataRecord] = []

        for schema, table, id_column in self.USER_DATA_TABLES:
            try:
                table_records = await self._scan_table(
                    schema=schema,
                    table=table,
                    id_column=id_column,
                    user_email=user_email,
                    user_id=user_id,
                )
                records.extend(table_records)

                self.scan_results.append(ScanResult(
                    source_system="postgresql",
                    source_location=f"{schema}.{table}",
                    records_found=len(table_records),
                    pii_detected=len(table_records) > 0,
                ))
            except Exception as e:
                logger.error("Error scanning %s.%s: %s", schema, table, str(e))
                self.scan_results.append(ScanResult(
                    source_system="postgresql",
                    source_location=f"{schema}.{table}",
                    errors=[str(e)],
                ))

        duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Database scan complete: %d records in %.2f ms",
            len(records),
            duration,
        )

        return records

    async def scan_object_storage(
        self,
        user_email: str,
        user_id: Optional[str] = None,
    ) -> List[DataRecord]:
        """Scan S3/object storage for user data.

        Args:
            user_email: User's email address.
            user_id: Optional user ID.

        Returns:
            List of discovered S3 records.
        """
        start_time = datetime.now(timezone.utc)
        records: List[DataRecord] = []

        for prefix in self.USER_DATA_PREFIXES:
            try:
                s3_records = await self._scan_s3_prefix(
                    prefix=prefix,
                    user_email=user_email,
                    user_id=user_id,
                )
                records.extend(s3_records)

                self.scan_results.append(ScanResult(
                    source_system="s3",
                    source_location=prefix,
                    records_found=len(s3_records),
                    pii_detected=len(s3_records) > 0,
                ))
            except Exception as e:
                logger.error("Error scanning S3 prefix %s: %s", prefix, str(e))
                self.scan_results.append(ScanResult(
                    source_system="s3",
                    source_location=prefix,
                    errors=[str(e)],
                ))

        duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "S3 scan complete: %d records in %.2f ms",
            len(records),
            duration,
        )

        return records

    async def scan_logs(
        self,
        user_email: str,
        user_id: Optional[str] = None,
    ) -> List[DataRecord]:
        """Scan application logs for user data.

        Args:
            user_email: User's email address.
            user_id: Optional user ID.

        Returns:
            List of discovered log records.
        """
        start_time = datetime.now(timezone.utc)
        records: List[DataRecord] = []

        for source in self.LOG_SOURCES:
            try:
                log_records = await self._scan_log_source(
                    source=source,
                    user_email=user_email,
                    user_id=user_id,
                )
                records.extend(log_records)

                self.scan_results.append(ScanResult(
                    source_system="loki",
                    source_location=source,
                    records_found=len(log_records),
                    pii_detected=len(log_records) > 0,
                ))
            except Exception as e:
                logger.error("Error scanning logs %s: %s", source, str(e))
                self.scan_results.append(ScanResult(
                    source_system="loki",
                    source_location=source,
                    errors=[str(e)],
                ))

        duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Log scan complete: %d records in %.2f ms",
            len(records),
            duration,
        )

        return records

    async def generate_data_inventory(self) -> List[DataInventoryItem]:
        """Generate a complete data inventory (ROPA support).

        Creates a Records of Processing Activities (ROPA) compliant
        inventory of all personal data processing.

        Returns:
            List of data inventory items.
        """
        logger.info("Generating data inventory")

        inventory: List[DataInventoryItem] = []

        # Database tables
        for schema, table, _ in self.USER_DATA_TABLES:
            inventory.append(DataInventoryItem(
                data_category=DataCategory.PII,
                source_system="postgresql",
                source_location=f"{schema}.{table}",
                record_count=await self._estimate_record_count(schema, table),
                pii_fields=await self._get_pii_fields(schema, table),
                retention_days=self._get_retention_days(schema, table),
                purpose=self._get_processing_purpose(schema, table),
                legal_basis=self._get_legal_basis(schema, table),
            ))

        # S3 prefixes
        for prefix in self.USER_DATA_PREFIXES:
            inventory.append(DataInventoryItem(
                data_category=DataCategory.PII,
                source_system="s3",
                source_location=prefix,
                record_count=await self._estimate_s3_objects(prefix),
                retention_days=365,
                purpose="User file storage",
                legal_basis="consent",
            ))

        # Log sources
        for source in self.LOG_SOURCES:
            inventory.append(DataInventoryItem(
                data_category=DataCategory.AUDIT if source == "audit" else DataCategory.OPERATIONAL,
                source_system="loki",
                source_location=source,
                retention_days=2555 if source == "audit" else 365,
                purpose="System logging and security monitoring",
                legal_basis="legitimate_interest",
            ))

        return inventory

    async def get_pii_summary(
        self,
        records: List[DataRecord],
    ) -> Dict[str, Any]:
        """Generate a summary of PII in discovered records.

        Args:
            records: List of discovered data records.

        Returns:
            Summary of PII types and locations.
        """
        summary: Dict[str, Any] = {
            "total_records": len(records),
            "by_system": {},
            "by_category": {},
            "pii_fields": set(),
            "sensitive_pii_detected": False,
        }

        for record in records:
            # Count by system
            system = record.source_system
            summary["by_system"][system] = summary["by_system"].get(system, 0) + 1

            # Count by category
            category = record.data_category.value
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1

            # Collect PII fields
            summary["pii_fields"].update(record.pii_fields)

            # Check for sensitive PII
            if record.data_category == DataCategory.SENSITIVE_PII:
                summary["sensitive_pii_detected"] = True

        # Convert set to list for JSON serialization
        summary["pii_fields"] = list(summary["pii_fields"])

        return summary

    # -------------------------------------------------------------------------
    # Private Methods - Data Source Scanners
    # -------------------------------------------------------------------------

    async def _scan_table(
        self,
        schema: str,
        table: str,
        id_column: str,
        user_email: str,
        user_id: Optional[str],
    ) -> List[DataRecord]:
        """Scan a database table for user records.

        In production, this would execute actual SQL queries.
        """
        # Placeholder implementation
        records: List[DataRecord] = []

        # Simulate finding records
        if user_email:
            # In production, query: SELECT * FROM schema.table WHERE email = $1
            sample_data = {
                "id": str(uuid4()),
                "email": user_email,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            pii_fields = self.pii_detector.detect_pii_fields(sample_data)
            category = self.pii_detector.categorize_data(sample_data)

            records.append(DataRecord(
                user_id=user_id or "unknown",
                source_system="postgresql",
                source_location=f"{schema}.{table}",
                data_type="database_record",
                data_category=category,
                record_data=sample_data,
                pii_fields=pii_fields,
                is_sensitive=category == DataCategory.SENSITIVE_PII,
            ))

        return records

    async def _scan_s3_prefix(
        self,
        prefix: str,
        user_email: str,
        user_id: Optional[str],
    ) -> List[DataRecord]:
        """Scan S3 prefix for user objects.

        In production, this would list and scan S3 objects.
        """
        records: List[DataRecord] = []

        # Placeholder - simulate finding user files
        if user_id:
            user_prefix = f"{prefix}{user_id}/"
            records.append(DataRecord(
                user_id=user_id,
                source_system="s3",
                source_location=user_prefix,
                data_type="s3_object",
                data_category=DataCategory.PII,
                record_data={"prefix": user_prefix, "object_count": 5},
                pii_fields=["user_id"],
            ))

        return records

    async def _scan_log_source(
        self,
        source: str,
        user_email: str,
        user_id: Optional[str],
    ) -> List[DataRecord]:
        """Scan log source for user data.

        In production, this would query Loki.
        """
        records: List[DataRecord] = []

        # Placeholder - logs may contain email/IP
        if user_email:
            records.append(DataRecord(
                user_id=user_id or "unknown",
                source_system="loki",
                source_location=source,
                data_type="log_entries",
                data_category=DataCategory.AUDIT if source == "audit" else DataCategory.OPERATIONAL,
                record_data={
                    "source": source,
                    "entries_found": 150,
                    "date_range": "last_365_days",
                },
                pii_fields=["email", "ip_address"],
            ))

        return records

    async def _scan_caches(
        self,
        user_email: str,
        user_id: Optional[str],
    ) -> List[DataRecord]:
        """Scan Redis caches for user data.

        In production, this would scan Redis keys.
        """
        records: List[DataRecord] = []

        if user_id:
            records.append(DataRecord(
                user_id=user_id,
                source_system="redis",
                source_location="user_cache",
                data_type="cache_entry",
                data_category=DataCategory.OPERATIONAL,
                record_data={"cached_keys": ["session", "preferences"]},
                pii_fields=["user_id"],
            ))

        return records

    # -------------------------------------------------------------------------
    # Private Methods - Helpers
    # -------------------------------------------------------------------------

    async def _estimate_record_count(self, schema: str, table: str) -> int:
        """Estimate record count for a table."""
        # In production, query pg_stat_user_tables
        return 10000

    async def _get_pii_fields(self, schema: str, table: str) -> List[str]:
        """Get list of PII fields in a table."""
        # In production, inspect table schema
        pii_map = {
            "users": ["email", "name", "phone"],
            "user_sessions": ["user_id", "ip_address"],
            "audit_logs": ["user_id", "ip_address", "user_agent"],
        }
        return pii_map.get(table, ["user_id"])

    def _get_retention_days(self, schema: str, table: str) -> int:
        """Get retention period for a table."""
        retention_map = {
            "audit_logs": 2555,  # 7 years
            "user_sessions": 90,
            "consent_records": 2555,  # 7 years (proof of consent)
        }
        return retention_map.get(table, 365)

    def _get_processing_purpose(self, schema: str, table: str) -> str:
        """Get processing purpose for a table."""
        purpose_map = {
            "users": "User account management",
            "user_sessions": "Authentication and security",
            "audit_logs": "Security and compliance auditing",
            "consent_records": "Consent tracking and compliance",
        }
        return purpose_map.get(table, "Application functionality")

    def _get_legal_basis(self, schema: str, table: str) -> str:
        """Get legal basis for processing."""
        basis_map = {
            "users": "contract",
            "user_sessions": "legitimate_interest",
            "audit_logs": "legal_obligation",
            "consent_records": "consent",
        }
        return basis_map.get(table, "legitimate_interest")

    async def _estimate_s3_objects(self, prefix: str) -> int:
        """Estimate object count for S3 prefix."""
        # In production, query S3 inventory
        return 5000


__all__ = [
    "DataDiscovery",
    "PIIDetector",
    "ScanResult",
    "DataInventoryItem",
    "PIIPattern",
]
