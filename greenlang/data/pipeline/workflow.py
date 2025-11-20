"""
Update Workflow and Approval System

Enterprise workflow for emission factor updates:
- Change request submission
- Review and approval process
- Testing before deployment
- Communication to users
- Version control
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import sqlite3
import logging
import json
from enum import Enum

from .models import (
    ChangeRequest,
    ChangeType,
    ReviewStatus,
    FactorVersion,
    ChangeLog
)

logger = logging.getLogger(__name__)


class ApprovalLevel(str, Enum):
    """Approval level required for changes."""
    AUTOMATIC = "automatic"  # Minor metadata changes
    PEER_REVIEW = "peer_review"  # Standard changes
    ADMIN_APPROVAL = "admin_approval"  # Major value changes
    COMMITTEE_REVIEW = "committee_review"  # New factors or deprecation


class UpdateWorkflow:
    """
    Manage update workflow for emission factors.

    Implements controlled process for factor updates with review and testing.
    """

    def __init__(
        self,
        db_path: str,
        approval_manager: 'ApprovalManager',
        version_db_path: Optional[str] = None
    ):
        """
        Initialize update workflow.

        Args:
            db_path: Path to main database
            approval_manager: Approval manager instance
            version_db_path: Path to versioning database
        """
        self.db_path = db_path
        self.approval_manager = approval_manager
        self.version_db_path = version_db_path or f"{db_path}.versions.db"
        self._init_version_db()

    def _init_version_db(self):
        """Initialize version tracking database."""
        conn = sqlite3.connect(self.version_db_path)
        cursor = conn.cursor()

        # Create versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS factor_versions (
                version_id TEXT PRIMARY KEY,
                factor_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                factor_data TEXT NOT NULL,
                previous_data TEXT,
                change_type TEXT NOT NULL,
                change_summary TEXT NOT NULL,
                changed_fields TEXT,
                changed_by TEXT NOT NULL,
                change_reason TEXT,
                version_timestamp TEXT NOT NULL,
                effective_from TEXT NOT NULL,
                effective_until TEXT,
                validation_passed INTEGER NOT NULL,
                validation_warnings TEXT,
                data_hash TEXT NOT NULL,
                UNIQUE(factor_id, version_number)
            )
        """)

        # Create changelog table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS change_log (
                log_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                change_type TEXT NOT NULL,
                factor_id TEXT,
                affected_factors TEXT,
                summary TEXT NOT NULL,
                details TEXT,
                before_value TEXT,
                after_value TEXT,
                changed_by TEXT NOT NULL,
                change_reason TEXT,
                import_job_id TEXT,
                review_status TEXT NOT NULL,
                reviewed_by TEXT,
                review_timestamp TEXT,
                review_notes TEXT
            )
        """)

        # Create change requests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS change_requests (
                request_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                change_type TEXT NOT NULL,
                factor_id TEXT NOT NULL,
                proposed_changes TEXT NOT NULL,
                current_values TEXT NOT NULL,
                change_reason TEXT NOT NULL,
                supporting_documentation TEXT,
                source_references TEXT,
                requested_by TEXT NOT NULL,
                requester_organization TEXT,
                review_status TEXT NOT NULL,
                assigned_reviewer TEXT,
                reviewed_by TEXT,
                review_timestamp TEXT,
                review_notes TEXT,
                impact_assessment TEXT,
                affected_calculations INTEGER DEFAULT 0,
                approved INTEGER DEFAULT 0,
                approval_conditions TEXT,
                implemented INTEGER DEFAULT 0,
                implementation_timestamp TEXT,
                implementation_job_id TEXT
            )
        """)

        conn.commit()
        conn.close()

        logger.info(f"Version database initialized: {self.version_db_path}")

    def submit_change_request(
        self,
        factor_id: str,
        change_type: ChangeType,
        proposed_changes: Dict[str, Any],
        change_reason: str,
        requested_by: str,
        supporting_docs: Optional[List[str]] = None,
        source_refs: Optional[List[str]] = None
    ) -> ChangeRequest:
        """
        Submit a change request for review.

        Args:
            factor_id: Factor to modify
            change_type: Type of change
            proposed_changes: Dictionary of proposed changes
            change_reason: Justification for change
            requested_by: User submitting request
            supporting_docs: Supporting documentation URLs
            source_refs: Source references

        Returns:
            ChangeRequest object
        """
        logger.info(f"Submitting change request for {factor_id}")

        # Get current values
        current_values = self._get_current_factor_values(factor_id)

        # Generate request ID
        request_id = f"cr_{factor_id}_{int(datetime.now().timestamp())}"

        # Create change request
        request = ChangeRequest(
            request_id=request_id,
            change_type=change_type,
            factor_id=factor_id,
            proposed_changes=proposed_changes,
            current_values=current_values,
            change_reason=change_reason,
            supporting_documentation=supporting_docs or [],
            source_references=source_refs or [],
            requested_by=requested_by
        )

        # Determine approval level required
        approval_level = self._determine_approval_level(change_type, proposed_changes, current_values)

        # Perform impact assessment
        request.impact_assessment = self._assess_impact(factor_id, proposed_changes)

        # Save to database
        self._save_change_request(request)

        # Route to appropriate approver
        self.approval_manager.assign_reviewer(request, approval_level)

        logger.info(f"Change request submitted: {request_id} (approval level: {approval_level})")

        return request

    def _get_current_factor_values(self, factor_id: str) -> Dict[str, Any]:
        """Get current factor values from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                factor_id, name, category, subcategory,
                emission_factor_value, unit, scope,
                source_org, source_uri, standard,
                last_updated, geographic_scope
            FROM emission_factors
            WHERE factor_id = ?
        """, (factor_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return {}

        return {
            'factor_id': row[0],
            'name': row[1],
            'category': row[2],
            'subcategory': row[3],
            'emission_factor_value': row[4],
            'unit': row[5],
            'scope': row[6],
            'source_org': row[7],
            'source_uri': row[8],
            'standard': row[9],
            'last_updated': row[10],
            'geographic_scope': row[11]
        }

    def _determine_approval_level(
        self,
        change_type: ChangeType,
        proposed: Dict[str, Any],
        current: Dict[str, Any]
    ) -> ApprovalLevel:
        """Determine required approval level."""
        # New factors require committee review
        if change_type == ChangeType.ADDED:
            return ApprovalLevel.COMMITTEE_REVIEW

        # Deletions require admin approval
        if change_type in [ChangeType.DELETED, ChangeType.DEPRECATED]:
            return ApprovalLevel.ADMIN_APPROVAL

        # Value changes
        if 'emission_factor_value' in proposed:
            current_value = current.get('emission_factor_value', 0)
            proposed_value = proposed['emission_factor_value']

            # Calculate percent change
            if current_value > 0:
                percent_change = abs((proposed_value - current_value) / current_value * 100)

                # >20% change requires admin approval
                if percent_change > 20:
                    return ApprovalLevel.ADMIN_APPROVAL
                # >10% change requires peer review
                elif percent_change > 10:
                    return ApprovalLevel.PEER_REVIEW
                # <10% change is automatic (if well-documented)
                else:
                    return ApprovalLevel.PEER_REVIEW

        # Metadata-only changes
        metadata_fields = {'notes', 'metadata_json', 'renewable_share'}
        if all(key in metadata_fields for key in proposed.keys()):
            return ApprovalLevel.AUTOMATIC

        # Default to peer review
        return ApprovalLevel.PEER_REVIEW

    def _assess_impact(
        self,
        factor_id: str,
        proposed_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess impact of proposed changes."""
        # Check how many calculations use this factor
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM calculation_audit_log
            WHERE factor_id = ?
        """, (factor_id,))

        usage_count = cursor.fetchone()[0]
        conn.close()

        # Estimate impact
        if 'emission_factor_value' in proposed_changes:
            impact_level = "high" if usage_count > 100 else "medium" if usage_count > 10 else "low"
        else:
            impact_level = "low"

        return {
            'usage_count': usage_count,
            'impact_level': impact_level,
            'requires_user_notification': usage_count > 50,
            'requires_testing': impact_level in ['high', 'medium']
        }

    def _save_change_request(self, request: ChangeRequest):
        """Save change request to database."""
        conn = sqlite3.connect(self.version_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO change_requests (
                request_id, timestamp, change_type, factor_id,
                proposed_changes, current_values, change_reason,
                supporting_documentation, source_references,
                requested_by, requester_organization,
                review_status, impact_assessment, affected_calculations
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request.request_id,
            request.timestamp.isoformat(),
            request.change_type.value,
            request.factor_id,
            json.dumps(request.proposed_changes),
            json.dumps(request.current_values),
            request.change_reason,
            json.dumps(request.supporting_documentation),
            json.dumps(request.source_references),
            request.requested_by,
            request.requester_organization,
            request.review_status.value,
            json.dumps(request.impact_assessment),
            request.affected_calculations
        ))

        conn.commit()
        conn.close()

    def implement_approved_change(
        self,
        request_id: str,
        implemented_by: str
    ) -> bool:
        """
        Implement an approved change request.

        Args:
            request_id: Change request ID
            implemented_by: User implementing change

        Returns:
            True if successful
        """
        # Load change request
        request = self._load_change_request(request_id)

        if not request:
            logger.error(f"Change request not found: {request_id}")
            return False

        if not request.approved:
            logger.error(f"Change request not approved: {request_id}")
            return False

        if request.implemented:
            logger.warning(f"Change request already implemented: {request_id}")
            return True

        logger.info(f"Implementing change request: {request_id}")

        try:
            # Apply changes to database
            self._apply_changes(request)

            # Create version record
            self._create_version_record(request, implemented_by)

            # Create changelog entry
            self._create_changelog_entry(request, implemented_by)

            # Mark request as implemented
            self._mark_implemented(request_id)

            logger.info(f"Change request implemented: {request_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to implement change request: {e}")
            return False

    def _load_change_request(self, request_id: str) -> Optional[ChangeRequest]:
        """Load change request from database."""
        conn = sqlite3.connect(self.version_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM change_requests WHERE request_id = ?
        """, (request_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        # Parse row into ChangeRequest object (simplified)
        return ChangeRequest(
            request_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            change_type=ChangeType(row[2]),
            factor_id=row[3],
            proposed_changes=json.loads(row[4]),
            current_values=json.loads(row[5]),
            change_reason=row[6],
            requested_by=row[9],
            review_status=ReviewStatus(row[11]),
            approved=bool(row[17])
        )

    def _apply_changes(self, request: ChangeRequest):
        """Apply changes to main database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build UPDATE statement
        updates = []
        values = []

        for field, value in request.proposed_changes.items():
            updates.append(f"{field} = ?")
            values.append(value)

        # Always update last_updated
        updates.append("updated_at = CURRENT_TIMESTAMP")

        values.append(request.factor_id)

        sql = f"""
            UPDATE emission_factors
            SET {', '.join(updates)}
            WHERE factor_id = ?
        """

        cursor.execute(sql, values)
        conn.commit()
        conn.close()

    def _create_version_record(self, request: ChangeRequest, implemented_by: str):
        """Create version history record."""
        # Get current version number
        conn = sqlite3.connect(self.version_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT MAX(version_number) FROM factor_versions
            WHERE factor_id = ?
        """, (request.factor_id,))

        max_version = cursor.fetchone()[0]
        new_version = (max_version or 0) + 1

        # Create version record
        version_id = f"v_{request.factor_id}_{new_version}"

        cursor.execute("""
            INSERT INTO factor_versions (
                version_id, factor_id, version_number,
                factor_data, previous_data,
                change_type, change_summary, changed_fields,
                changed_by, change_reason,
                version_timestamp, effective_from,
                validation_passed, data_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            version_id,
            request.factor_id,
            new_version,
            json.dumps(request.proposed_changes),
            json.dumps(request.current_values),
            request.change_type.value,
            f"Change request {request.request_id}",
            json.dumps(list(request.proposed_changes.keys())),
            implemented_by,
            request.change_reason,
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            1,  # validation_passed
            self._calculate_hash(request.proposed_changes)
        ))

        conn.commit()
        conn.close()

    def _create_changelog_entry(self, request: ChangeRequest, implemented_by: str):
        """Create changelog entry."""
        conn = sqlite3.connect(self.version_db_path)
        cursor = conn.cursor()

        log_id = f"log_{int(datetime.now().timestamp())}"

        cursor.execute("""
            INSERT INTO change_log (
                log_id, timestamp, change_type, factor_id,
                summary, details, before_value, after_value,
                changed_by, change_reason, review_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log_id,
            datetime.now().isoformat(),
            request.change_type.value,
            request.factor_id,
            f"Implemented change request {request.request_id}",
            json.dumps(request.proposed_changes),
            json.dumps(request.current_values),
            json.dumps(request.proposed_changes),
            implemented_by,
            request.change_reason,
            ReviewStatus.APPROVED.value
        ))

        conn.commit()
        conn.close()

    def _mark_implemented(self, request_id: str):
        """Mark change request as implemented."""
        conn = sqlite3.connect(self.version_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE change_requests
            SET implemented = 1, implementation_timestamp = ?
            WHERE request_id = ?
        """, (datetime.now().isoformat(), request_id))

        conn.commit()
        conn.close()

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of data."""
        import hashlib
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


class ApprovalManager:
    """
    Manage approval workflow for change requests.

    Routes requests to appropriate reviewers and tracks approval status.
    """

    def __init__(self, version_db_path: str):
        """
        Initialize approval manager.

        Args:
            version_db_path: Path to versioning database
        """
        self.version_db_path = version_db_path

        # Configure reviewers by approval level
        self.reviewers = {
            ApprovalLevel.AUTOMATIC: None,  # No reviewer needed
            ApprovalLevel.PEER_REVIEW: ['data_engineer_1', 'data_engineer_2'],
            ApprovalLevel.ADMIN_APPROVAL: ['data_admin'],
            ApprovalLevel.COMMITTEE_REVIEW: ['committee_chair']
        }

    def assign_reviewer(
        self,
        request: ChangeRequest,
        approval_level: ApprovalLevel
    ):
        """
        Assign reviewer to change request.

        Args:
            request: Change request
            approval_level: Required approval level
        """
        if approval_level == ApprovalLevel.AUTOMATIC:
            # Auto-approve
            self.approve_request(request.request_id, "system", "Automatic approval for minor change")
            return

        # Assign to reviewer
        reviewers = self.reviewers.get(approval_level, [])
        if reviewers:
            # Simple round-robin assignment (could be more sophisticated)
            assigned_reviewer = reviewers[0]

            conn = sqlite3.connect(self.version_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE change_requests
                SET assigned_reviewer = ?
                WHERE request_id = ?
            """, (assigned_reviewer, request.request_id))

            conn.commit()
            conn.close()

            logger.info(f"Assigned reviewer {assigned_reviewer} to request {request.request_id}")

    def approve_request(
        self,
        request_id: str,
        reviewed_by: str,
        review_notes: Optional[str] = None
    ):
        """
        Approve a change request.

        Args:
            request_id: Request to approve
            reviewed_by: Reviewer name
            review_notes: Optional review notes
        """
        conn = sqlite3.connect(self.version_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE change_requests
            SET review_status = ?,
                reviewed_by = ?,
                review_timestamp = ?,
                review_notes = ?,
                approved = 1
            WHERE request_id = ?
        """, (
            ReviewStatus.APPROVED.value,
            reviewed_by,
            datetime.now().isoformat(),
            review_notes,
            request_id
        ))

        conn.commit()
        conn.close()

        logger.info(f"Change request approved: {request_id} by {reviewed_by}")

    def reject_request(
        self,
        request_id: str,
        reviewed_by: str,
        review_notes: str
    ):
        """
        Reject a change request.

        Args:
            request_id: Request to reject
            reviewed_by: Reviewer name
            review_notes: Reason for rejection
        """
        conn = sqlite3.connect(self.version_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE change_requests
            SET review_status = ?,
                reviewed_by = ?,
                review_timestamp = ?,
                review_notes = ?
            WHERE request_id = ?
        """, (
            ReviewStatus.REJECTED.value,
            reviewed_by,
            datetime.now().isoformat(),
            review_notes,
            request_id
        ))

        conn.commit()
        conn.close()

        logger.info(f"Change request rejected: {request_id} by {reviewed_by}")

    def get_pending_reviews(self, reviewer: str) -> List[Dict[str, Any]]:
        """
        Get pending change requests for reviewer.

        Args:
            reviewer: Reviewer name

        Returns:
            List of pending change requests
        """
        conn = sqlite3.connect(self.version_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT request_id, timestamp, change_type, factor_id, change_reason
            FROM change_requests
            WHERE assigned_reviewer = ?
              AND review_status = ?
            ORDER BY timestamp ASC
        """, (reviewer, ReviewStatus.PENDING_REVIEW.value))

        results = []
        for row in cursor.fetchall():
            results.append({
                'request_id': row[0],
                'timestamp': row[1],
                'change_type': row[2],
                'factor_id': row[3],
                'change_reason': row[4]
            })

        conn.close()

        return results
