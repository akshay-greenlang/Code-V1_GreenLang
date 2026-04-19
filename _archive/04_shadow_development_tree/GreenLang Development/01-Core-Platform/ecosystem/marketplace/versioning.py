# -*- coding: utf-8 -*-
"""
Agent Versioning System

Implements semantic versioning, breaking change detection,
and version compatibility checking for marketplace agents.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from sqlalchemy.orm import Session
from sqlalchemy import desc

from greenlang.marketplace.models import MarketplaceAgent, AgentVersion
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class VersionConstraintOperator(str, Enum):
    """Version constraint operators"""
    EXACT = "=="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    COMPATIBLE = "~="  # Compatible release (same major.minor)
    CARET = "^"  # Caret (same major)


@dataclass
class SemanticVersion:
    """Semantic version (MAJOR.MINOR.PATCH)"""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    @classmethod
    def parse(cls, version_string: str) -> 'SemanticVersion':
        """
        Parse semantic version string.

        Args:
            version_string: Version string (e.g., "1.2.3-alpha+build")

        Returns:
            SemanticVersion instance
        """
        # Pattern: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$'
        match = re.match(pattern, version_string.strip())

        if not match:
            raise ValueError(f"Invalid semantic version: {version_string}")

        major, minor, patch, prerelease, build = match.groups()

        return cls(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            prerelease=prerelease,
            build=build
        )

    def __str__(self) -> str:
        """Convert to string"""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __eq__(self, other: 'SemanticVersion') -> bool:
        """Equality comparison"""
        return (
            self.major == other.major and
            self.minor == other.minor and
            self.patch == other.patch and
            self.prerelease == other.prerelease
        )

    def __lt__(self, other: 'SemanticVersion') -> bool:
        """Less than comparison"""
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch

        # Handle prereleases
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease

        return False

    def __le__(self, other: 'SemanticVersion') -> bool:
        return self < other or self == other

    def __gt__(self, other: 'SemanticVersion') -> bool:
        return not self <= other

    def __ge__(self, other: 'SemanticVersion') -> bool:
        return not self < other

    def is_compatible_with(self, other: 'SemanticVersion') -> bool:
        """Check if versions are compatible (same major.minor)"""
        return self.major == other.major and self.minor == other.minor

    def is_major_compatible(self, other: 'SemanticVersion') -> bool:
        """Check if versions have same major version"""
        return self.major == other.major


@dataclass
class VersionConstraint:
    """Version constraint (e.g., ">=1.0.0,<2.0.0")"""
    operator: VersionConstraintOperator
    version: SemanticVersion

    @classmethod
    def parse(cls, constraint_string: str) -> List['VersionConstraint']:
        """
        Parse version constraint string.

        Args:
            constraint_string: Constraint (e.g., ">=1.0.0,<2.0.0")

        Returns:
            List of VersionConstraint
        """
        constraints = []

        # Split by comma
        parts = constraint_string.split(',')

        for part in parts:
            part = part.strip()

            # Match operator and version
            for op in VersionConstraintOperator:
                if part.startswith(op.value):
                    version_str = part[len(op.value):].strip()
                    version = SemanticVersion.parse(version_str)
                    constraints.append(cls(operator=op, version=version))
                    break

        return constraints

    def matches(self, version: SemanticVersion) -> bool:
        """Check if version satisfies constraint"""
        if self.operator == VersionConstraintOperator.EXACT:
            return version == self.version
        elif self.operator == VersionConstraintOperator.GREATER_THAN:
            return version > self.version
        elif self.operator == VersionConstraintOperator.GREATER_EQUAL:
            return version >= self.version
        elif self.operator == VersionConstraintOperator.LESS_THAN:
            return version < self.version
        elif self.operator == VersionConstraintOperator.LESS_EQUAL:
            return version <= self.version
        elif self.operator == VersionConstraintOperator.COMPATIBLE:
            return version.is_compatible_with(self.version) and version >= self.version
        elif self.operator == VersionConstraintOperator.CARET:
            return version.is_major_compatible(self.version) and version >= self.version
        return False


class BreakingChangeDetector:
    """
    Detect breaking changes between agent versions.

    Compares input/output schemas to identify breaking changes.
    """

    def __init__(self, session: Session):
        self.session = session

    def has_breaking_changes(
        self,
        agent_id: str,
        new_input_schema: Dict[str, Any],
        new_output_schema: Dict[str, Any]
    ) -> bool:
        """
        Check if new schemas have breaking changes.

        Args:
            agent_id: Agent UUID
            new_input_schema: New input schema
            new_output_schema: New output schema

        Returns:
            True if breaking changes detected
        """
        # Get latest version
        latest_version = self.session.query(AgentVersion).filter(
            AgentVersion.agent_id == agent_id
        ).order_by(desc(AgentVersion.published_at)).first()

        if not latest_version:
            return False  # First version, no breaking changes

        # Compare schemas
        input_breaking = self._compare_schemas(
            latest_version.schema_input,
            new_input_schema,
            check_input=True
        )

        output_breaking = self._compare_schemas(
            latest_version.schema_output,
            new_output_schema,
            check_input=False
        )

        return input_breaking or output_breaking

    def _compare_schemas(
        self,
        old_schema: Optional[Dict[str, Any]],
        new_schema: Optional[Dict[str, Any]],
        check_input: bool
    ) -> bool:
        """
        Compare two schemas for breaking changes.

        For inputs: Adding required fields or removing fields is breaking
        For outputs: Removing fields is breaking

        Args:
            old_schema: Old schema
            new_schema: New schema
            check_input: True for input schema, False for output

        Returns:
            True if breaking changes detected
        """
        if not old_schema or not new_schema:
            return False

        old_props = old_schema.get('properties', {})
        new_props = new_schema.get('properties', {})

        old_required = set(old_schema.get('required', []))
        new_required = set(new_schema.get('required', []))

        if check_input:
            # For inputs:
            # 1. Adding new required field is breaking
            if new_required - old_required:
                return True

            # 2. Removing any field is breaking (consumers may depend on it)
            if set(old_props.keys()) - set(new_props.keys()):
                return True

            # 3. Changing field type is breaking
            for field in old_props:
                if field in new_props:
                    if old_props[field].get('type') != new_props[field].get('type'):
                        return True

        else:
            # For outputs:
            # 1. Removing field is breaking
            if set(old_props.keys()) - set(new_props.keys()):
                return True

            # 2. Changing field type is breaking
            for field in old_props:
                if field in new_props:
                    if old_props[field].get('type') != new_props[field].get('type'):
                        return True

        return False

    def detect_changes(
        self,
        agent_id: str,
        new_version: SemanticVersion,
        new_input_schema: Dict[str, Any],
        new_output_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect all changes between versions.

        Args:
            agent_id: Agent UUID
            new_version: New version
            new_input_schema: New input schema
            new_output_schema: New output schema

        Returns:
            Dictionary with change details
        """
        latest_version = self.session.query(AgentVersion).filter(
            AgentVersion.agent_id == agent_id
        ).order_by(desc(AgentVersion.published_at)).first()

        if not latest_version:
            return {
                "is_first_version": True,
                "breaking_changes": False,
                "changes": []
            }

        changes = []

        # Compare input schemas
        if latest_version.schema_input and new_input_schema:
            input_changes = self._list_schema_changes(
                latest_version.schema_input,
                new_input_schema,
                "input"
            )
            changes.extend(input_changes)

        # Compare output schemas
        if latest_version.schema_output and new_output_schema:
            output_changes = self._list_schema_changes(
                latest_version.schema_output,
                new_output_schema,
                "output"
            )
            changes.extend(output_changes)

        breaking = self.has_breaking_changes(agent_id, new_input_schema, new_output_schema)

        return {
            "is_first_version": False,
            "previous_version": latest_version.version,
            "new_version": str(new_version),
            "breaking_changes": breaking,
            "changes": changes
        }

    def _list_schema_changes(
        self,
        old_schema: Dict[str, Any],
        new_schema: Dict[str, Any],
        schema_type: str
    ) -> List[Dict[str, Any]]:
        """List all changes between schemas"""
        changes = []

        old_props = old_schema.get('properties', {})
        new_props = new_schema.get('properties', {})

        # Added fields
        added = set(new_props.keys()) - set(old_props.keys())
        for field in added:
            changes.append({
                "type": "added",
                "schema": schema_type,
                "field": field,
                "breaking": False
            })

        # Removed fields
        removed = set(old_props.keys()) - set(new_props.keys())
        for field in removed:
            changes.append({
                "type": "removed",
                "schema": schema_type,
                "field": field,
                "breaking": True
            })

        # Modified fields
        for field in old_props:
            if field in new_props:
                if old_props[field] != new_props[field]:
                    breaking = old_props[field].get('type') != new_props[field].get('type')
                    changes.append({
                        "type": "modified",
                        "schema": schema_type,
                        "field": field,
                        "breaking": breaking,
                        "old": old_props[field],
                        "new": new_props[field]
                    })

        return changes


class VersionManager:
    """
    Manage agent versions.

    Handles version history, deprecation, and rollback.
    """

    def __init__(self, session: Session):
        self.session = session

    def get_versions(
        self,
        agent_id: str,
        include_deprecated: bool = False
    ) -> List[AgentVersion]:
        """
        Get all versions for an agent.

        Args:
            agent_id: Agent UUID
            include_deprecated: Include deprecated versions

        Returns:
            List of versions
        """
        query = self.session.query(AgentVersion).filter(
            AgentVersion.agent_id == agent_id
        )

        if not include_deprecated:
            query = query.filter(AgentVersion.deprecated == False)

        versions = query.order_by(desc(AgentVersion.published_at)).all()
        return versions

    def get_latest_version(self, agent_id: str) -> Optional[AgentVersion]:
        """Get latest non-deprecated version"""
        version = self.session.query(AgentVersion).filter(
            AgentVersion.agent_id == agent_id,
            AgentVersion.deprecated == False
        ).order_by(desc(AgentVersion.published_at)).first()

        return version

    def find_compatible_version(
        self,
        agent_id: str,
        constraint: str
    ) -> Optional[AgentVersion]:
        """
        Find version matching constraint.

        Args:
            agent_id: Agent UUID
            constraint: Version constraint (e.g., ">=1.0.0,<2.0.0")

        Returns:
            Matching version or None
        """
        constraints = VersionConstraint.parse(constraint)
        versions = self.get_versions(agent_id)

        # Find latest matching version
        for version in versions:
            sem_ver = SemanticVersion.parse(version.version)

            # Check all constraints
            if all(c.matches(sem_ver) for c in constraints):
                return version

        return None

    def deprecate_version(
        self,
        version_id: str,
        reason: str,
        superseded_by: Optional[str] = None
    ) -> bool:
        """
        Deprecate a version.

        Args:
            version_id: Version UUID
            reason: Deprecation reason
            superseded_by: Version that supersedes this one

        Returns:
            True if successful
        """
        version = self.session.query(AgentVersion).filter(
            AgentVersion.id == version_id
        ).first()

        if not version:
            return False

        version.deprecated = True
        version.deprecated_reason = reason
        version.superseded_by = superseded_by
        version.deprecated_at = DeterministicClock.now()

        self.session.commit()

        logger.info(f"Deprecated version {version_id}: {reason}")

        return True

    def suggest_version(
        self,
        agent_id: str,
        has_breaking_changes: bool
    ) -> str:
        """
        Suggest next version number.

        Args:
            agent_id: Agent UUID
            has_breaking_changes: Whether changes are breaking

        Returns:
            Suggested version string
        """
        latest = self.get_latest_version(agent_id)

        if not latest:
            return "1.0.0"

        current = SemanticVersion.parse(latest.version)

        if has_breaking_changes:
            # Increment major version
            return f"{current.major + 1}.0.0"
        else:
            # Increment minor version
            return f"{current.major}.{current.minor + 1}.0"
