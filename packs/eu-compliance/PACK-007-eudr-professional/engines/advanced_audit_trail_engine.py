"""
AdvancedAuditTrailEngine - Comprehensive audit management for EUDR compliance

This module implements the advanced audit trail engine for PACK-007 EUDR Professional Pack.
Provides comprehensive audit logging, chain-of-custody verification, competent authority
inspection preparation, and evidence package assembly per EU Regulation 2023/1115.

Example:
    >>> config = AuditConfig(retention_years=5)
    >>> engine = AdvancedAuditTrailEngine(config)
    >>> entry = engine.log_action("DDS_SUBMITTED", "due_diligence_statement", "DDS-001", {...})
    >>> package = engine.prepare_ca_inspection("OPERATOR-001")
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field, validator
import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
import json
import base64

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats for audit trails."""
    JSON = "JSON"
    XML = "XML"
    PDF = "PDF"
    CSV = "CSV"


class ActionType(str, Enum):
    """EUDR audit action types."""
    DDS_CREATED = "DDS_CREATED"
    DDS_SUBMITTED = "DDS_SUBMITTED"
    DDS_UPDATED = "DDS_UPDATED"
    RISK_ASSESSED = "RISK_ASSESSED"
    MITIGATION_APPLIED = "MITIGATION_APPLIED"
    PLOT_VERIFIED = "PLOT_VERIFIED"
    DOCUMENT_UPLOADED = "DOCUMENT_UPLOADED"
    SUPPLIER_VERIFIED = "SUPPLIER_VERIFIED"
    TRACEABILITY_CONFIRMED = "TRACEABILITY_CONFIRMED"
    CA_INSPECTION_REQUESTED = "CA_INSPECTION_REQUESTED"


class DocumentType(str, Enum):
    """Document classification for EUDR evidence."""
    GEOLOCATION_DATA = "GEOLOCATION_DATA"
    SUPPLY_CHAIN_DOC = "SUPPLY_CHAIN_DOC"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    MITIGATION_PLAN = "MITIGATION_PLAN"
    DUE_DILIGENCE_STATEMENT = "DUE_DILIGENCE_STATEMENT"
    THIRD_PARTY_CERTIFICATE = "THIRD_PARTY_CERTIFICATE"
    CUSTOMS_DECLARATION = "CUSTOMS_DECLARATION"
    OPERATOR_PROFILE = "OPERATOR_PROFILE"


class AuditConfig(BaseModel):
    """Configuration for advanced audit trail engine."""

    retention_years: int = Field(5, ge=1, le=10, description="Audit record retention period in years")
    hash_algorithm: str = Field("SHA-256", description="Cryptographic hash algorithm for chain integrity")
    export_formats: List[ExportFormat] = Field(
        default=[ExportFormat.JSON, ExportFormat.XML, ExportFormat.PDF],
        description="Supported export formats"
    )
    enable_blockchain: bool = Field(False, description="Enable blockchain anchoring for immutability")
    auto_archive: bool = Field(True, description="Automatically archive old records")
    compression_enabled: bool = Field(True, description="Compress archived audit logs")

    @validator('hash_algorithm')
    def validate_hash_algorithm(cls, v):
        """Validate hash algorithm is supported."""
        allowed = ["SHA-256", "SHA-512", "SHA3-256"]
        if v not in allowed:
            raise ValueError(f"Hash algorithm must be one of {allowed}")
        return v


class AuditEntry(BaseModel):
    """Individual audit log entry with cryptographic chaining."""

    entry_id: str = Field(..., description="Unique entry identifier")
    timestamp: datetime = Field(..., description="Entry creation timestamp (UTC)")
    action: ActionType = Field(..., description="Type of action performed")
    entity_type: str = Field(..., description="Type of entity affected (e.g., DDS, plot, supplier)")
    entity_id: str = Field(..., description="Unique identifier of affected entity")
    user_id: str = Field(..., description="User who performed the action")
    data_before: Optional[Dict[str, Any]] = Field(None, description="Entity state before action")
    data_after: Optional[Dict[str, Any]] = Field(None, description="Entity state after action")
    hash: str = Field(..., description="SHA-256 hash of this entry")
    previous_hash: Optional[str] = Field(None, description="Hash of previous entry (chain link)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional contextual metadata")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    session_id: Optional[str] = Field(None, description="User session identifier")


class EvidenceDocument(BaseModel):
    """Document included in evidence package."""

    document_id: str = Field(..., description="Unique document identifier")
    document_type: DocumentType = Field(..., description="Type of document")
    filename: str = Field(..., description="Original filename")
    upload_date: datetime = Field(..., description="Upload timestamp")
    file_hash: str = Field(..., description="SHA-256 hash of file contents")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    related_entities: List[str] = Field(default_factory=list, description="Related entity IDs")


class EvidencePackage(BaseModel):
    """Comprehensive evidence package for competent authority inspection."""

    package_id: str = Field(..., description="Unique package identifier")
    operator_id: str = Field(..., description="Operator this package pertains to")
    creation_date: datetime = Field(..., description="Package creation timestamp")
    articles_covered: List[str] = Field(..., description="EUDR articles covered (e.g., Art. 9, Art. 10)")
    documents: List[EvidenceDocument] = Field(..., description="All supporting documents")
    dds_list: List[str] = Field(..., description="Due Diligence Statement IDs included")
    risk_assessments: List[str] = Field(..., description="Risk assessment IDs included")
    mitigation_records: List[str] = Field(..., description="Mitigation measure record IDs")
    audit_entries: List[str] = Field(..., description="Audit entry IDs relevant to this package")
    package_hash: str = Field(..., description="SHA-256 hash of entire package")
    digital_signature: Optional[str] = Field(None, description="Digital signature for authenticity")


class ComplianceDeadline(BaseModel):
    """Compliance deadline for EUDR obligations."""

    deadline_id: str = Field(..., description="Unique deadline identifier")
    article: str = Field(..., description="EUDR article reference")
    description: str = Field(..., description="Deadline description")
    due_date: datetime = Field(..., description="Deadline date")
    recurrence: Optional[str] = Field(None, description="Recurrence pattern (e.g., 'quarterly', 'annual')")
    status: str = Field("PENDING", description="Deadline status (PENDING, MET, OVERDUE)")
    related_obligations: List[str] = Field(default_factory=list, description="Related obligation IDs")


class InspectionWindow(BaseModel):
    """Competent authority inspection window."""

    window_id: str = Field(..., description="Unique window identifier")
    start_date: datetime = Field(..., description="Window start date")
    end_date: datetime = Field(..., description="Window end date")
    inspection_type: str = Field(..., description="Type of inspection (routine, targeted, random)")
    probability: float = Field(0.0, ge=0.0, le=1.0, description="Estimated inspection probability")


class ComplianceCalendar(BaseModel):
    """Compliance calendar with deadlines and inspection windows."""

    operator_id: str = Field(..., description="Operator identifier")
    calendar_year: int = Field(..., description="Calendar year")
    deadlines: List[ComplianceDeadline] = Field(..., description="All compliance deadlines")
    inspection_windows: List[InspectionWindow] = Field(..., description="Potential inspection windows")
    submission_dates: Dict[str, datetime] = Field(..., description="Key submission dates")


class ChainVerification(BaseModel):
    """Result of audit chain integrity verification."""

    is_valid: bool = Field(..., description="Whether chain is valid")
    total_entries: int = Field(..., ge=0, description="Total entries verified")
    broken_links: List[Tuple[str, str]] = Field(default_factory=list, description="Broken chain links (entry_id, reason)")
    hash_mismatches: List[str] = Field(default_factory=list, description="Entry IDs with hash mismatches")
    verification_timestamp: datetime = Field(..., description="Verification timestamp")
    verification_hash: str = Field(..., description="Hash of verification result")


class RetentionReport(BaseModel):
    """Retention compliance report."""

    total_records: int = Field(..., ge=0, description="Total records evaluated")
    expired_records: List[str] = Field(..., description="Record IDs past retention period")
    active_records: int = Field(..., ge=0, description="Records within retention period")
    retention_compliance_rate: float = Field(..., ge=0.0, le=1.0, description="Compliance rate (0-1)")
    oldest_record_date: Optional[datetime] = Field(None, description="Oldest active record date")
    archive_candidates: List[str] = Field(default_factory=list, description="Records eligible for archiving")


class DocumentClassification(BaseModel):
    """Document classification result."""

    document_id: str = Field(..., description="Document identifier")
    classified_type: DocumentType = Field(..., description="Classified document type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence (0-1)")
    extracted_metadata: Dict[str, Any] = Field(default_factory=dict, description="Extracted metadata")
    validation_status: str = Field(..., description="Validation status (VALID, INVALID, NEEDS_REVIEW)")


class MockAuditFinding(BaseModel):
    """Finding from mock audit."""

    finding_id: str = Field(..., description="Finding identifier")
    severity: str = Field(..., description="Severity (CRITICAL, HIGH, MEDIUM, LOW, INFO)")
    article: str = Field(..., description="EUDR article reference")
    description: str = Field(..., description="Finding description")
    recommendation: str = Field(..., description="Recommended corrective action")
    affected_entities: List[str] = Field(default_factory=list, description="Affected entity IDs")


class MockAuditResult(BaseModel):
    """Result of mock internal audit."""

    audit_id: str = Field(..., description="Audit identifier")
    operator_id: str = Field(..., description="Operator audited")
    audit_date: datetime = Field(..., description="Audit execution date")
    findings: List[MockAuditFinding] = Field(..., description="All findings")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall compliance score (0-100)")
    readiness_level: str = Field(..., description="Inspection readiness (NOT_READY, PARTIALLY_READY, READY)")
    recommendations: List[str] = Field(..., description="High-level recommendations")


class InspectionPackage(BaseModel):
    """Complete package for competent authority inspection."""

    package_id: str = Field(..., description="Package identifier")
    operator_id: str = Field(..., description="Operator identifier")
    preparation_date: datetime = Field(..., description="Package preparation date")
    evidence_package: EvidencePackage = Field(..., description="Evidence package")
    compliance_calendar: ComplianceCalendar = Field(..., description="Compliance calendar")
    mock_audit_result: Optional[MockAuditResult] = Field(None, description="Latest mock audit result")
    operator_profile: Dict[str, Any] = Field(..., description="Operator profile data")
    statistics: Dict[str, Any] = Field(..., description="Key compliance statistics")
    package_hash: str = Field(..., description="SHA-256 hash of complete package")


class AdvancedAuditTrailEngine:
    """
    Advanced audit trail engine for EUDR compliance.

    Implements comprehensive audit logging with cryptographic chaining, evidence package
    assembly, competent authority inspection preparation, and retention management.

    Attributes:
        config: Engine configuration
        audit_entries: In-memory audit entry storage (use DB in production)
        documents: Document registry

    Example:
        >>> config = AuditConfig(retention_years=5)
        >>> engine = AdvancedAuditTrailEngine(config)
        >>> entry = engine.log_action("DDS_SUBMITTED", "due_diligence_statement", "DDS-001", {...})
        >>> assert entry.hash is not None
    """

    def __init__(self, config: AuditConfig):
        """Initialize advanced audit trail engine."""
        self.config = config
        self.audit_entries: Dict[str, AuditEntry] = {}
        self.documents: Dict[str, EvidenceDocument] = {}
        self.last_entry_hash: Optional[str] = None
        logger.info(f"AdvancedAuditTrailEngine initialized with retention={config.retention_years} years")

    def log_action(
        self,
        action: str,
        entity_type: str,
        entity_id: str,
        data: Dict[str, Any],
        user_id: str = "SYSTEM",
        data_before: Optional[Dict[str, Any]] = None
    ) -> AuditEntry:
        """
        Log an action to the audit trail with cryptographic chaining.

        Args:
            action: Action type performed
            entity_type: Type of entity affected
            entity_id: Entity identifier
            data: Entity state after action
            user_id: User who performed action
            data_before: Entity state before action (optional)

        Returns:
            Created audit entry with hash chain

        Raises:
            ValueError: If action type is invalid
        """
        try:
            timestamp = datetime.utcnow()
            entry_id = f"AUDIT-{timestamp.strftime('%Y%m%d%H%M%S%f')}-{entity_id}"

            # Create entry with all data
            entry_data = {
                "entry_id": entry_id,
                "timestamp": timestamp.isoformat(),
                "action": action,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "user_id": user_id,
                "data_before": data_before,
                "data_after": data,
            }

            # Calculate hash of entry data
            entry_hash = self._calculate_hash(entry_data)

            # Create audit entry with chain link
            entry = AuditEntry(
                entry_id=entry_id,
                timestamp=timestamp,
                action=ActionType(action),
                entity_type=entity_type,
                entity_id=entity_id,
                user_id=user_id,
                data_before=data_before,
                data_after=data,
                hash=entry_hash,
                previous_hash=self.last_entry_hash
            )

            # Store entry and update chain
            self.audit_entries[entry_id] = entry
            self.last_entry_hash = entry_hash

            logger.info(f"Logged action {action} for {entity_type} {entity_id}, hash={entry_hash[:16]}...")
            return entry

        except Exception as e:
            logger.error(f"Failed to log action {action}: {str(e)}", exc_info=True)
            raise

    def get_audit_chain(
        self,
        entity_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[AuditEntry]:
        """
        Retrieve audit chain for an entity within date range.

        Args:
            entity_id: Entity identifier
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of audit entries in chronological order
        """
        try:
            # Filter entries by entity and date range
            entries = [
                entry for entry in self.audit_entries.values()
                if entry.entity_id == entity_id
            ]

            if start_date:
                entries = [e for e in entries if e.timestamp >= start_date]
            if end_date:
                entries = [e for e in entries if e.timestamp <= end_date]

            # Sort chronologically
            entries.sort(key=lambda e: e.timestamp)

            logger.info(f"Retrieved {len(entries)} audit entries for entity {entity_id}")
            return entries

        except Exception as e:
            logger.error(f"Failed to retrieve audit chain: {str(e)}", exc_info=True)
            return []

    def verify_chain_integrity(self, entries: List[AuditEntry]) -> ChainVerification:
        """
        Verify cryptographic integrity of audit chain.

        Args:
            entries: List of audit entries to verify (must be chronologically sorted)

        Returns:
            Chain verification result with detailed diagnostics
        """
        try:
            is_valid = True
            broken_links = []
            hash_mismatches = []

            for i, entry in enumerate(entries):
                # Verify entry hash
                entry_data = {
                    "entry_id": entry.entry_id,
                    "timestamp": entry.timestamp.isoformat(),
                    "action": entry.action.value,
                    "entity_type": entry.entity_type,
                    "entity_id": entry.entity_id,
                    "user_id": entry.user_id,
                    "data_before": entry.data_before,
                    "data_after": entry.data_after,
                }
                expected_hash = self._calculate_hash(entry_data)

                if entry.hash != expected_hash:
                    is_valid = False
                    hash_mismatches.append(entry.entry_id)
                    logger.warning(f"Hash mismatch for entry {entry.entry_id}")

                # Verify chain link
                if i > 0:
                    previous_entry = entries[i - 1]
                    if entry.previous_hash != previous_entry.hash:
                        is_valid = False
                        broken_links.append((entry.entry_id, "previous_hash does not match"))
                        logger.warning(f"Broken chain link at entry {entry.entry_id}")

            verification_timestamp = datetime.utcnow()
            verification_data = {
                "is_valid": is_valid,
                "total_entries": len(entries),
                "timestamp": verification_timestamp.isoformat()
            }
            verification_hash = self._calculate_hash(verification_data)

            result = ChainVerification(
                is_valid=is_valid,
                total_entries=len(entries),
                broken_links=broken_links,
                hash_mismatches=hash_mismatches,
                verification_timestamp=verification_timestamp,
                verification_hash=verification_hash
            )

            logger.info(f"Chain verification complete: valid={is_valid}, entries={len(entries)}")
            return result

        except Exception as e:
            logger.error(f"Chain verification failed: {str(e)}", exc_info=True)
            raise

    def assemble_evidence(self, articles: List[str], operator_id: str) -> EvidencePackage:
        """
        Assemble evidence package for specified EUDR articles.

        Args:
            articles: List of EUDR articles to cover (e.g., ["Art. 9", "Art. 10"])
            operator_id: Operator identifier

        Returns:
            Complete evidence package with all supporting documents
        """
        try:
            package_id = f"EVPKG-{operator_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

            # Collect all documents (in production, query from database)
            documents = list(self.documents.values())

            # Collect DDS IDs (mock data)
            dds_list = [f"DDS-{operator_id}-{i:03d}" for i in range(1, 6)]

            # Collect risk assessments
            risk_assessments = [f"RISK-{operator_id}-{i:03d}" for i in range(1, 4)]

            # Collect mitigation records
            mitigation_records = [f"MIT-{operator_id}-{i:03d}" for i in range(1, 3)]

            # Collect relevant audit entries
            audit_entries = [
                entry.entry_id for entry in self.audit_entries.values()
                if operator_id in entry.entity_id or operator_id in str(entry.data_after)
            ]

            # Calculate package hash
            package_data = {
                "package_id": package_id,
                "operator_id": operator_id,
                "articles": articles,
                "dds_list": dds_list,
                "risk_assessments": risk_assessments,
                "mitigation_records": mitigation_records
            }
            package_hash = self._calculate_hash(package_data)

            package = EvidencePackage(
                package_id=package_id,
                operator_id=operator_id,
                creation_date=datetime.utcnow(),
                articles_covered=articles,
                documents=documents,
                dds_list=dds_list,
                risk_assessments=risk_assessments,
                mitigation_records=mitigation_records,
                audit_entries=audit_entries,
                package_hash=package_hash
            )

            logger.info(f"Assembled evidence package {package_id} with {len(documents)} documents")
            return package

        except Exception as e:
            logger.error(f"Failed to assemble evidence package: {str(e)}", exc_info=True)
            raise

    def prepare_ca_inspection(self, operator_id: str) -> InspectionPackage:
        """
        Prepare complete package for competent authority inspection.

        Args:
            operator_id: Operator identifier

        Returns:
            Complete inspection package with all materials
        """
        try:
            # Assemble evidence package
            articles = ["Art. 9", "Art. 10", "Art. 11", "Art. 13"]
            evidence_package = self.assemble_evidence(articles, operator_id)

            # Get compliance calendar
            compliance_calendar = self.get_compliance_calendar(operator_id)

            # Run mock audit
            mock_audit_result = self.run_mock_audit(operator_id)

            # Prepare operator profile
            operator_profile = {
                "operator_id": operator_id,
                "name": f"Operator {operator_id}",
                "country": "DEU",
                "registration_date": "2024-01-15",
                "commodity_types": ["TIMBER", "PALM_OIL"],
                "annual_volume_tonnes": 15000
            }

            # Calculate statistics
            statistics = {
                "total_dds_submitted": len(evidence_package.dds_list),
                "total_risk_assessments": len(evidence_package.risk_assessments),
                "total_documents": len(evidence_package.documents),
                "audit_entries_count": len(evidence_package.audit_entries),
                "compliance_rate": mock_audit_result.overall_score / 100.0
            }

            # Create inspection package
            package_data = {
                "operator_id": operator_id,
                "evidence_package_id": evidence_package.package_id,
                "calendar_year": compliance_calendar.calendar_year,
                "statistics": statistics
            }
            package_hash = self._calculate_hash(package_data)

            inspection_package = InspectionPackage(
                package_id=f"INSPKG-{operator_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                operator_id=operator_id,
                preparation_date=datetime.utcnow(),
                evidence_package=evidence_package,
                compliance_calendar=compliance_calendar,
                mock_audit_result=mock_audit_result,
                operator_profile=operator_profile,
                statistics=statistics,
                package_hash=package_hash
            )

            logger.info(f"Prepared CA inspection package for operator {operator_id}")
            return inspection_package

        except Exception as e:
            logger.error(f"Failed to prepare CA inspection package: {str(e)}", exc_info=True)
            raise

    def check_retention_compliance(self, record_ids: List[str]) -> RetentionReport:
        """
        Check retention policy compliance for records.

        Args:
            record_ids: List of record identifiers to check

        Returns:
            Retention compliance report
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_years * 365)

            expired_records = []
            active_count = 0
            oldest_date = None
            archive_candidates = []

            for record_id in record_ids:
                # Get record (in production, query from database)
                if record_id in self.audit_entries:
                    entry = self.audit_entries[record_id]

                    if entry.timestamp < cutoff_date:
                        expired_records.append(record_id)
                    else:
                        active_count += 1
                        if oldest_date is None or entry.timestamp < oldest_date:
                            oldest_date = entry.timestamp

                    # Check if eligible for archiving (within 30 days of expiration)
                    archive_cutoff = datetime.utcnow() - timedelta(days=(self.config.retention_years * 365) - 30)
                    if entry.timestamp < archive_cutoff:
                        archive_candidates.append(record_id)

            compliance_rate = active_count / len(record_ids) if record_ids else 1.0

            report = RetentionReport(
                total_records=len(record_ids),
                expired_records=expired_records,
                active_records=active_count,
                retention_compliance_rate=compliance_rate,
                oldest_record_date=oldest_date,
                archive_candidates=archive_candidates
            )

            logger.info(f"Retention check: {active_count}/{len(record_ids)} records compliant")
            return report

        except Exception as e:
            logger.error(f"Failed to check retention compliance: {str(e)}", exc_info=True)
            raise

    def export_audit_trail(self, entity_id: str, format: str = "JSON") -> bytes:
        """
        Export audit trail in specified format.

        Args:
            entity_id: Entity identifier
            format: Export format (JSON, XML, PDF, CSV)

        Returns:
            Exported data as bytes

        Raises:
            ValueError: If format is not supported
        """
        try:
            if format not in [f.value for f in ExportFormat]:
                raise ValueError(f"Unsupported format: {format}")

            entries = self.get_audit_chain(entity_id)

            if format == "JSON":
                data = [entry.dict() for entry in entries]
                json_str = json.dumps(data, indent=2, default=str)
                return json_str.encode('utf-8')

            elif format == "XML":
                xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<audit_trail>']
                for entry in entries:
                    xml_lines.append('  <entry>')
                    xml_lines.append(f'    <entry_id>{entry.entry_id}</entry_id>')
                    xml_lines.append(f'    <timestamp>{entry.timestamp.isoformat()}</timestamp>')
                    xml_lines.append(f'    <action>{entry.action.value}</action>')
                    xml_lines.append(f'    <entity_type>{entry.entity_type}</entity_type>')
                    xml_lines.append(f'    <entity_id>{entry.entity_id}</entity_id>')
                    xml_lines.append(f'    <hash>{entry.hash}</hash>')
                    xml_lines.append('  </entry>')
                xml_lines.append('</audit_trail>')
                return '\n'.join(xml_lines).encode('utf-8')

            elif format == "CSV":
                csv_lines = ['entry_id,timestamp,action,entity_type,entity_id,user_id,hash']
                for entry in entries:
                    csv_lines.append(
                        f'{entry.entry_id},{entry.timestamp.isoformat()},{entry.action.value},'
                        f'{entry.entity_type},{entry.entity_id},{entry.user_id},{entry.hash}'
                    )
                return '\n'.join(csv_lines).encode('utf-8')

            elif format == "PDF":
                # In production, use reportlab or similar
                pdf_content = f"AUDIT TRAIL REPORT\nEntity: {entity_id}\nEntries: {len(entries)}\n"
                return pdf_content.encode('utf-8')

        except Exception as e:
            logger.error(f"Failed to export audit trail: {str(e)}", exc_info=True)
            raise

    def classify_document(self, document: Dict[str, Any]) -> DocumentClassification:
        """
        Classify document type based on content and metadata.

        Args:
            document: Document data with filename, content, metadata

        Returns:
            Document classification result
        """
        try:
            filename = document.get('filename', '').lower()
            content = document.get('content', '')

            # Simple rule-based classification (in production, use ML)
            if 'geolocation' in filename or 'gps' in filename or 'coordinates' in content.lower():
                doc_type = DocumentType.GEOLOCATION_DATA
                confidence = 0.95
            elif 'supply_chain' in filename or 'supplier' in content.lower():
                doc_type = DocumentType.SUPPLY_CHAIN_DOC
                confidence = 0.90
            elif 'risk' in filename or 'assessment' in filename:
                doc_type = DocumentType.RISK_ASSESSMENT
                confidence = 0.85
            elif 'mitigation' in filename or 'plan' in filename:
                doc_type = DocumentType.MITIGATION_PLAN
                confidence = 0.80
            elif 'dds' in filename or 'due_diligence' in filename:
                doc_type = DocumentType.DUE_DILIGENCE_STATEMENT
                confidence = 0.90
            elif 'certificate' in filename or 'cert' in filename:
                doc_type = DocumentType.THIRD_PARTY_CERTIFICATE
                confidence = 0.85
            elif 'customs' in filename or 'declaration' in filename:
                doc_type = DocumentType.CUSTOMS_DECLARATION
                confidence = 0.88
            else:
                doc_type = DocumentType.OPERATOR_PROFILE
                confidence = 0.60

            # Validate document
            validation_status = "VALID" if confidence >= 0.75 else "NEEDS_REVIEW"

            result = DocumentClassification(
                document_id=document.get('document_id', 'UNKNOWN'),
                classified_type=doc_type,
                confidence=confidence,
                extracted_metadata={
                    "filename": filename,
                    "size_bytes": len(content),
                    "classification_timestamp": datetime.utcnow().isoformat()
                },
                validation_status=validation_status
            )

            logger.info(f"Classified document as {doc_type.value} with confidence {confidence:.2f}")
            return result

        except Exception as e:
            logger.error(f"Failed to classify document: {str(e)}", exc_info=True)
            raise

    def run_mock_audit(self, operator_id: str) -> MockAuditResult:
        """
        Run internal mock audit to assess inspection readiness.

        Args:
            operator_id: Operator identifier

        Returns:
            Mock audit result with findings and recommendations
        """
        try:
            audit_id = f"MOCK-{operator_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

            # Simulate audit findings (in production, run actual checks)
            findings = [
                MockAuditFinding(
                    finding_id=f"{audit_id}-F001",
                    severity="MEDIUM",
                    article="Art. 9",
                    description="3 DDS records missing geolocation coordinates for 2 plots",
                    recommendation="Update DDS records with complete geolocation data (latitude/longitude)",
                    affected_entities=["DDS-001", "DDS-003", "DDS-005"]
                ),
                MockAuditFinding(
                    finding_id=f"{audit_id}-F002",
                    severity="LOW",
                    article="Art. 10",
                    description="Risk assessment for Plot-042 is 45 days old",
                    recommendation="Refresh risk assessment to ensure currency",
                    affected_entities=["PLOT-042"]
                ),
                MockAuditFinding(
                    finding_id=f"{audit_id}-F003",
                    severity="INFO",
                    article="Art. 13",
                    description="Audit trail retention at 4.8 years (compliant)",
                    recommendation="Continue current retention practices",
                    affected_entities=[]
                )
            ]

            # Calculate overall score
            severity_weights = {"CRITICAL": 0, "HIGH": 60, "MEDIUM": 80, "LOW": 90, "INFO": 100}
            if findings:
                overall_score = sum(severity_weights.get(f.severity, 75) for f in findings) / len(findings)
            else:
                overall_score = 100.0

            # Determine readiness
            if overall_score >= 90:
                readiness_level = "READY"
            elif overall_score >= 70:
                readiness_level = "PARTIALLY_READY"
            else:
                readiness_level = "NOT_READY"

            recommendations = [
                "Address all MEDIUM and HIGH findings before inspection",
                "Ensure all DDS records have complete geolocation data",
                "Maintain audit trail retention compliance at ≥5 years",
                "Keep risk assessments current (refresh every 30 days for high-risk plots)"
            ]

            result = MockAuditResult(
                audit_id=audit_id,
                operator_id=operator_id,
                audit_date=datetime.utcnow(),
                findings=findings,
                overall_score=overall_score,
                readiness_level=readiness_level,
                recommendations=recommendations
            )

            logger.info(f"Mock audit complete for {operator_id}: score={overall_score:.1f}, readiness={readiness_level}")
            return result

        except Exception as e:
            logger.error(f"Failed to run mock audit: {str(e)}", exc_info=True)
            raise

    def get_compliance_calendar(self, operator_id: str) -> ComplianceCalendar:
        """
        Generate compliance calendar with deadlines and inspection windows.

        Args:
            operator_id: Operator identifier

        Returns:
            Compliance calendar for current year
        """
        try:
            current_year = datetime.utcnow().year

            # Define EUDR compliance deadlines
            deadlines = [
                ComplianceDeadline(
                    deadline_id=f"DL-{current_year}-Q1-DDS",
                    article="Art. 9",
                    description="Q1 Due Diligence Statements submission",
                    due_date=datetime(current_year, 3, 31),
                    recurrence="quarterly",
                    status="PENDING"
                ),
                ComplianceDeadline(
                    deadline_id=f"DL-{current_year}-Q2-DDS",
                    article="Art. 9",
                    description="Q2 Due Diligence Statements submission",
                    due_date=datetime(current_year, 6, 30),
                    recurrence="quarterly",
                    status="PENDING"
                ),
                ComplianceDeadline(
                    deadline_id=f"DL-{current_year}-Q3-DDS",
                    article="Art. 9",
                    description="Q3 Due Diligence Statements submission",
                    due_date=datetime(current_year, 9, 30),
                    recurrence="quarterly",
                    status="PENDING"
                ),
                ComplianceDeadline(
                    deadline_id=f"DL-{current_year}-Q4-DDS",
                    article="Art. 9",
                    description="Q4 Due Diligence Statements submission",
                    due_date=datetime(current_year, 12, 31),
                    recurrence="quarterly",
                    status="PENDING"
                ),
                ComplianceDeadline(
                    deadline_id=f"DL-{current_year}-ANNUAL-REPORT",
                    article="Art. 13",
                    description="Annual compliance report to competent authority",
                    due_date=datetime(current_year, 12, 31),
                    recurrence="annual",
                    status="PENDING"
                )
            ]

            # Define potential inspection windows
            inspection_windows = [
                InspectionWindow(
                    window_id=f"INSP-{current_year}-Q1",
                    start_date=datetime(current_year, 1, 1),
                    end_date=datetime(current_year, 3, 31),
                    inspection_type="routine",
                    probability=0.15
                ),
                InspectionWindow(
                    window_id=f"INSP-{current_year}-Q3",
                    start_date=datetime(current_year, 7, 1),
                    end_date=datetime(current_year, 9, 30),
                    inspection_type="routine",
                    probability=0.20
                ),
                InspectionWindow(
                    window_id=f"INSP-{current_year}-TARGETED",
                    start_date=datetime(current_year, 10, 1),
                    end_date=datetime(current_year, 11, 30),
                    inspection_type="targeted",
                    probability=0.35
                )
            ]

            submission_dates = {
                "Q1_DDS": datetime(current_year, 3, 31),
                "Q2_DDS": datetime(current_year, 6, 30),
                "Q3_DDS": datetime(current_year, 9, 30),
                "Q4_DDS": datetime(current_year, 12, 31),
                "ANNUAL_REPORT": datetime(current_year, 12, 31)
            }

            calendar = ComplianceCalendar(
                operator_id=operator_id,
                calendar_year=current_year,
                deadlines=deadlines,
                inspection_windows=inspection_windows,
                submission_dates=submission_dates
            )

            logger.info(f"Generated compliance calendar for {operator_id}, year {current_year}")
            return calendar

        except Exception as e:
            logger.error(f"Failed to generate compliance calendar: {str(e)}", exc_info=True)
            raise

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """
        Calculate SHA-256 hash of data for provenance tracking.

        Args:
            data: Data to hash

        Returns:
            Hexadecimal hash string
        """
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
