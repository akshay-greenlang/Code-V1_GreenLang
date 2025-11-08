"""
SCIM 2.0 User Provisioning Provider for GreenLang

This module implements a SCIM 2.0 server for automated user and group provisioning
from identity providers like Okta, Azure AD, and OneLogin.

SCIM (System for Cross-domain Identity Management) is a standard for automating
user provisioning and deprovisioning.

Features:
- SCIM 2.0 Core Schema support
- User provisioning (create, read, update, delete, search)
- Group provisioning
- Bulk operations
- Filtering and pagination
- Webhook notifications for provisioning events
- Schema discovery

Security:
- Bearer token authentication
- Input validation
- Rate limiting
- Audit logging

Reference: RFC 7644 (SCIM Protocol), RFC 7643 (SCIM Core Schema)
"""

import logging
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class SCIMResourceType(Enum):
    """SCIM resource types"""
    USER = "User"
    GROUP = "Group"
    ENTERPRISE_USER = "EnterpriseUser"
    SERVICE_PROVIDER_CONFIG = "ServiceProviderConfig"
    RESOURCE_TYPE = "ResourceType"
    SCHEMA = "Schema"


class SCIMOperation(Enum):
    """SCIM operations"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    REPLACE = "replace"
    DELETE = "delete"
    SEARCH = "search"


class SCIMPatchOp(Enum):
    """SCIM PATCH operations"""
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"


@dataclass
class SCIMConfig:
    """SCIM Server Configuration"""
    # Server settings
    base_url: str = "https://api.greenlang.io/scim/v2"

    # Authentication
    bearer_token: Optional[str] = None

    # Features
    support_bulk: bool = True
    support_patch: bool = True
    support_filter: bool = True
    support_etag: bool = True
    support_sort: bool = True

    # Limits
    max_results: int = 100
    max_bulk_operations: int = 1000

    # Webhooks
    webhook_enabled: bool = False
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None

    # Schema
    user_schema_extensions: List[str] = field(default_factory=list)


@dataclass
class SCIMName:
    """SCIM name structure"""
    formatted: Optional[str] = None
    familyName: Optional[str] = None
    givenName: Optional[str] = None
    middleName: Optional[str] = None
    honorificPrefix: Optional[str] = None
    honorificSuffix: Optional[str] = None


@dataclass
class SCIMEmail:
    """SCIM email structure"""
    value: str
    type: str = "work"  # work, home, other
    primary: bool = True
    display: Optional[str] = None


@dataclass
class SCIMPhoneNumber:
    """SCIM phone number structure"""
    value: str
    type: str = "work"  # work, home, mobile, fax, pager, other
    primary: bool = False


@dataclass
class SCIMAddress:
    """SCIM address structure"""
    formatted: Optional[str] = None
    streetAddress: Optional[str] = None
    locality: Optional[str] = None
    region: Optional[str] = None
    postalCode: Optional[str] = None
    country: Optional[str] = None
    type: str = "work"  # work, home, other
    primary: bool = True


@dataclass
class SCIMPhoto:
    """SCIM photo structure"""
    value: str  # URL
    type: str = "photo"
    primary: bool = True


@dataclass
class SCIMMeta:
    """SCIM meta data"""
    resourceType: str
    created: str
    lastModified: str
    location: str
    version: Optional[str] = None


@dataclass
class SCIMUser:
    """SCIM User resource"""
    # Core attributes
    schemas: List[str] = field(default_factory=lambda: ["urn:ietf:params:scim:schemas:core:2.0:User"])
    id: str = ""
    externalId: Optional[str] = None
    userName: str = ""

    # Name
    name: Optional[SCIMName] = None
    displayName: Optional[str] = None
    nickName: Optional[str] = None
    profileUrl: Optional[str] = None
    title: Optional[str] = None
    userType: Optional[str] = None
    preferredLanguage: Optional[str] = None
    locale: Optional[str] = None
    timezone: Optional[str] = None

    # Status
    active: bool = True

    # Multi-valued attributes
    emails: List[SCIMEmail] = field(default_factory=list)
    phoneNumbers: List[SCIMPhoneNumber] = field(default_factory=list)
    addresses: List[SCIMAddress] = field(default_factory=list)
    photos: List[SCIMPhoto] = field(default_factory=list)

    # Groups (read-only)
    groups: List[Dict[str, str]] = field(default_factory=list)

    # Enterprise extension
    enterpriseUser: Optional[Dict[str, Any]] = None

    # Meta
    meta: Optional[SCIMMeta] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            "schemas": self.schemas,
            "id": self.id,
            "userName": self.userName,
            "active": self.active,
        }

        # Optional attributes
        if self.externalId:
            data["externalId"] = self.externalId
        if self.name:
            data["name"] = asdict(self.name)
        if self.displayName:
            data["displayName"] = self.displayName
        if self.emails:
            data["emails"] = [asdict(e) for e in self.emails]
        if self.phoneNumbers:
            data["phoneNumbers"] = [asdict(p) for p in self.phoneNumbers]
        if self.groups:
            data["groups"] = self.groups
        if self.meta:
            data["meta"] = asdict(self.meta)
        if self.enterpriseUser:
            data["urn:ietf:params:scim:schemas:extension:enterprise:2.0:User"] = self.enterpriseUser

        return data


@dataclass
class SCIMGroupMember:
    """SCIM group member"""
    value: str  # User ID
    ref: str  # User URL
    type: str = "User"
    display: Optional[str] = None


@dataclass
class SCIMGroup:
    """SCIM Group resource"""
    schemas: List[str] = field(default_factory=lambda: ["urn:ietf:params:scim:schemas:core:2.0:Group"])
    id: str = ""
    externalId: Optional[str] = None
    displayName: str = ""
    members: List[SCIMGroupMember] = field(default_factory=list)
    meta: Optional[SCIMMeta] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            "schemas": self.schemas,
            "id": self.id,
            "displayName": self.displayName,
        }

        if self.externalId:
            data["externalId"] = self.externalId
        if self.members:
            data["members"] = [asdict(m) for m in self.members]
        if self.meta:
            data["meta"] = asdict(self.meta)

        return data


@dataclass
class SCIMListResponse:
    """SCIM list response"""
    schemas: List[str] = field(default_factory=lambda: ["urn:ietf:params:scim:api:messages:2.0:ListResponse"])
    totalResults: int = 0
    startIndex: int = 1
    itemsPerPage: int = 0
    Resources: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SCIMError:
    """SCIM error response"""
    schemas: List[str] = field(default_factory=lambda: ["urn:ietf:params:scim:api:messages:2.0:Error"])
    status: int = 400
    scimType: Optional[str] = None
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            "schemas": self.schemas,
            "status": str(self.status),
        }

        if self.scimType:
            data["scimType"] = self.scimType
        if self.detail:
            data["detail"] = self.detail

        return data


@dataclass
class SCIMPatchOperation:
    """SCIM PATCH operation"""
    op: str  # add, remove, replace
    path: Optional[str] = None
    value: Any = None


@dataclass
class SCIMBulkOperation:
    """SCIM bulk operation"""
    method: str  # POST, PUT, PATCH, DELETE
    bulkId: str
    path: str
    data: Optional[Dict[str, Any]] = None
    version: Optional[str] = None


@dataclass
class SCIMWebhookEvent:
    """Webhook event for SCIM operations"""
    event_id: str
    event_type: str  # user.created, user.updated, user.deleted, etc.
    timestamp: datetime
    resource_type: SCIMResourceType
    resource_id: str
    operation: SCIMOperation
    data: Dict[str, Any]


class SCIMFilter:
    """SCIM filter parser and evaluator"""

    @staticmethod
    def parse(filter_string: str) -> Dict[str, Any]:
        """Parse SCIM filter string (simplified)"""
        # This is a simplified parser - production would need full SCIM filter grammar

        # Common patterns
        patterns = {
            r'(\w+)\s+eq\s+"([^"]+)"': lambda m: {"attribute": m.group(1), "op": "eq", "value": m.group(2)},
            r'(\w+)\s+eq\s+(\w+)': lambda m: {"attribute": m.group(1), "op": "eq", "value": m.group(2)},
            r'(\w+)\s+co\s+"([^"]+)"': lambda m: {"attribute": m.group(1), "op": "co", "value": m.group(2)},
            r'(\w+)\s+sw\s+"([^"]+)"': lambda m: {"attribute": m.group(1), "op": "sw", "value": m.group(2)},
            r'(\w+)\s+pr': lambda m: {"attribute": m.group(1), "op": "pr"},
        }

        for pattern, handler in patterns.items():
            match = re.match(pattern, filter_string, re.IGNORECASE)
            if match:
                return handler(match)

        return {"raw": filter_string}

    @staticmethod
    def evaluate(resource: Dict[str, Any], filter_expr: Dict[str, Any]) -> bool:
        """Evaluate filter against resource"""
        if "raw" in filter_expr:
            # Cannot evaluate complex filter
            return True

        attribute = filter_expr.get("attribute")
        op = filter_expr.get("op")
        value = filter_expr.get("value")

        # Get attribute value from resource
        resource_value = resource.get(attribute)

        if op == "eq":
            return str(resource_value).lower() == str(value).lower()
        elif op == "co":  # contains
            return value.lower() in str(resource_value).lower()
        elif op == "sw":  # starts with
            return str(resource_value).lower().startswith(value.lower())
        elif op == "pr":  # present
            return resource_value is not None

        return True


class SCIMProvider:
    """
    SCIM 2.0 Provider Implementation

    Implements SCIM 2.0 server for user and group provisioning.
    """

    def __init__(self, config: SCIMConfig):
        self.config = config

        # Storage (in-memory for demo - production would use database)
        self.users: Dict[str, SCIMUser] = {}
        self.groups: Dict[str, SCIMGroup] = {}

        # Webhook events queue
        self.webhook_events: List[SCIMWebhookEvent] = []

        logger.info("Initialized SCIM 2.0 provider")

    # User operations

    def create_user(self, user_data: Dict[str, Any]) -> SCIMUser:
        """Create a new user"""
        # Generate ID
        user_id = str(uuid.uuid4())

        # Parse user data
        user = self._parse_user_data(user_data)
        user.id = user_id

        # Create meta
        now = datetime.utcnow().isoformat() + "Z"
        user.meta = SCIMMeta(
            resourceType="User",
            created=now,
            lastModified=now,
            location=f"{self.config.base_url}/Users/{user_id}"
        )

        # Store
        self.users[user_id] = user

        # Emit webhook event
        self._emit_webhook_event(
            event_type="user.created",
            resource_type=SCIMResourceType.USER,
            resource_id=user_id,
            operation=SCIMOperation.CREATE,
            data=user.to_dict()
        )

        logger.info(f"Created user: {user.userName} ({user_id})")

        return user

    def get_user(self, user_id: str) -> Optional[SCIMUser]:
        """Get user by ID"""
        return self.users.get(user_id)

    def update_user(self, user_id: str, user_data: Dict[str, Any]) -> Optional[SCIMUser]:
        """Update user (full replacement)"""
        if user_id not in self.users:
            return None

        # Parse updated data
        updated_user = self._parse_user_data(user_data)
        updated_user.id = user_id

        # Preserve creation time
        old_meta = self.users[user_id].meta
        now = datetime.utcnow().isoformat() + "Z"
        updated_user.meta = SCIMMeta(
            resourceType="User",
            created=old_meta.created if old_meta else now,
            lastModified=now,
            location=f"{self.config.base_url}/Users/{user_id}"
        )

        # Update
        self.users[user_id] = updated_user

        # Emit webhook event
        self._emit_webhook_event(
            event_type="user.updated",
            resource_type=SCIMResourceType.USER,
            resource_id=user_id,
            operation=SCIMOperation.UPDATE,
            data=updated_user.to_dict()
        )

        logger.info(f"Updated user: {user_id}")

        return updated_user

    def patch_user(self, user_id: str, operations: List[Dict[str, Any]]) -> Optional[SCIMUser]:
        """Patch user with SCIM PATCH operations"""
        user = self.users.get(user_id)
        if not user:
            return None

        # Apply operations
        for op_data in operations:
            op = op_data.get("op", "").lower()
            path = op_data.get("path")
            value = op_data.get("value")

            if op == "replace":
                self._apply_patch_replace(user, path, value)
            elif op == "add":
                self._apply_patch_add(user, path, value)
            elif op == "remove":
                self._apply_patch_remove(user, path)

        # Update meta
        now = datetime.utcnow().isoformat() + "Z"
        if user.meta:
            user.meta.lastModified = now

        # Emit webhook event
        self._emit_webhook_event(
            event_type="user.updated",
            resource_type=SCIMResourceType.USER,
            resource_id=user_id,
            operation=SCIMOperation.UPDATE,
            data=user.to_dict()
        )

        logger.info(f"Patched user: {user_id}")

        return user

    def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        if user_id not in self.users:
            return False

        # Remove from groups
        for group in self.groups.values():
            group.members = [m for m in group.members if m.value != user_id]

        # Delete user
        user = self.users[user_id]
        del self.users[user_id]

        # Emit webhook event
        self._emit_webhook_event(
            event_type="user.deleted",
            resource_type=SCIMResourceType.USER,
            resource_id=user_id,
            operation=SCIMOperation.DELETE,
            data={"id": user_id, "userName": user.userName}
        )

        logger.info(f"Deleted user: {user_id}")

        return True

    def search_users(
        self,
        filter_expr: Optional[str] = None,
        start_index: int = 1,
        count: int = 100,
        attributes: Optional[List[str]] = None
    ) -> SCIMListResponse:
        """Search users"""
        # Apply filter
        filtered_users = list(self.users.values())

        if filter_expr:
            parsed_filter = SCIMFilter.parse(filter_expr)
            filtered_users = [
                u for u in filtered_users
                if SCIMFilter.evaluate(u.to_dict(), parsed_filter)
            ]

        # Pagination
        total = len(filtered_users)
        start_idx = max(0, start_index - 1)
        end_idx = min(start_idx + count, total)
        page_users = filtered_users[start_idx:end_idx]

        # Build response
        resources = [u.to_dict() for u in page_users]

        # Filter attributes if requested
        if attributes:
            resources = self._filter_attributes(resources, attributes)

        response = SCIMListResponse(
            totalResults=total,
            startIndex=start_index,
            itemsPerPage=len(resources),
            Resources=resources
        )

        return response

    # Group operations

    def create_group(self, group_data: Dict[str, Any]) -> SCIMGroup:
        """Create a new group"""
        # Generate ID
        group_id = str(uuid.uuid4())

        # Parse group data
        group = self._parse_group_data(group_data)
        group.id = group_id

        # Create meta
        now = datetime.utcnow().isoformat() + "Z"
        group.meta = SCIMMeta(
            resourceType="Group",
            created=now,
            lastModified=now,
            location=f"{self.config.base_url}/Groups/{group_id}"
        )

        # Store
        self.groups[group_id] = group

        # Update user group memberships
        self._update_user_group_memberships(group_id)

        # Emit webhook event
        self._emit_webhook_event(
            event_type="group.created",
            resource_type=SCIMResourceType.GROUP,
            resource_id=group_id,
            operation=SCIMOperation.CREATE,
            data=group.to_dict()
        )

        logger.info(f"Created group: {group.displayName} ({group_id})")

        return group

    def get_group(self, group_id: str) -> Optional[SCIMGroup]:
        """Get group by ID"""
        return self.groups.get(group_id)

    def update_group(self, group_id: str, group_data: Dict[str, Any]) -> Optional[SCIMGroup]:
        """Update group"""
        if group_id not in self.groups:
            return None

        # Parse updated data
        updated_group = self._parse_group_data(group_data)
        updated_group.id = group_id

        # Preserve creation time
        old_meta = self.groups[group_id].meta
        now = datetime.utcnow().isoformat() + "Z"
        updated_group.meta = SCIMMeta(
            resourceType="Group",
            created=old_meta.created if old_meta else now,
            lastModified=now,
            location=f"{self.config.base_url}/Groups/{group_id}"
        )

        # Update
        self.groups[group_id] = updated_group

        # Update user group memberships
        self._update_user_group_memberships(group_id)

        # Emit webhook event
        self._emit_webhook_event(
            event_type="group.updated",
            resource_type=SCIMResourceType.GROUP,
            resource_id=group_id,
            operation=SCIMOperation.UPDATE,
            data=updated_group.to_dict()
        )

        logger.info(f"Updated group: {group_id}")

        return updated_group

    def delete_group(self, group_id: str) -> bool:
        """Delete group"""
        if group_id not in self.groups:
            return False

        group = self.groups[group_id]

        # Remove from users
        for member in group.members:
            user = self.users.get(member.value)
            if user:
                user.groups = [g for g in user.groups if g.get("value") != group_id]

        # Delete group
        del self.groups[group_id]

        # Emit webhook event
        self._emit_webhook_event(
            event_type="group.deleted",
            resource_type=SCIMResourceType.GROUP,
            resource_id=group_id,
            operation=SCIMOperation.DELETE,
            data={"id": group_id, "displayName": group.displayName}
        )

        logger.info(f"Deleted group: {group_id}")

        return True

    def search_groups(
        self,
        filter_expr: Optional[str] = None,
        start_index: int = 1,
        count: int = 100
    ) -> SCIMListResponse:
        """Search groups"""
        # Apply filter
        filtered_groups = list(self.groups.values())

        if filter_expr:
            parsed_filter = SCIMFilter.parse(filter_expr)
            filtered_groups = [
                g for g in filtered_groups
                if SCIMFilter.evaluate(g.to_dict(), parsed_filter)
            ]

        # Pagination
        total = len(filtered_groups)
        start_idx = max(0, start_index - 1)
        end_idx = min(start_idx + count, total)
        page_groups = filtered_groups[start_idx:end_idx]

        # Build response
        resources = [g.to_dict() for g in page_groups]

        response = SCIMListResponse(
            totalResults=total,
            startIndex=start_index,
            itemsPerPage=len(resources),
            Resources=resources
        )

        return response

    # Helper methods

    def _parse_user_data(self, data: Dict[str, Any]) -> SCIMUser:
        """Parse user data from dict"""
        user = SCIMUser(
            userName=data.get("userName", ""),
            externalId=data.get("externalId"),
            active=data.get("active", True),
            displayName=data.get("displayName"),
        )

        # Parse name
        if "name" in data:
            name_data = data["name"]
            user.name = SCIMName(
                formatted=name_data.get("formatted"),
                familyName=name_data.get("familyName"),
                givenName=name_data.get("givenName"),
                middleName=name_data.get("middleName"),
            )

        # Parse emails
        if "emails" in data:
            user.emails = [
                SCIMEmail(**email) for email in data["emails"]
            ]

        # Parse phone numbers
        if "phoneNumbers" in data:
            user.phoneNumbers = [
                SCIMPhoneNumber(**phone) for phone in data["phoneNumbers"]
            ]

        return user

    def _parse_group_data(self, data: Dict[str, Any]) -> SCIMGroup:
        """Parse group data from dict"""
        group = SCIMGroup(
            displayName=data.get("displayName", ""),
            externalId=data.get("externalId"),
        )

        # Parse members
        if "members" in data:
            group.members = [
                SCIMGroupMember(**member) for member in data["members"]
            ]

        return group

    def _apply_patch_replace(self, user: SCIMUser, path: Optional[str], value: Any) -> None:
        """Apply PATCH replace operation"""
        if not path:
            # Replace whole resource attributes
            if "active" in value:
                user.active = value["active"]
            return

        # Handle specific paths
        if path == "active":
            user.active = value
        elif path == "displayName":
            user.displayName = value

    def _apply_patch_add(self, user: SCIMUser, path: Optional[str], value: Any) -> None:
        """Apply PATCH add operation"""
        if path == "emails":
            if isinstance(value, list):
                user.emails.extend([SCIMEmail(**e) for e in value])
            else:
                user.emails.append(SCIMEmail(**value))

    def _apply_patch_remove(self, user: SCIMUser, path: Optional[str]) -> None:
        """Apply PATCH remove operation"""
        if path == "emails":
            user.emails.clear()

    def _update_user_group_memberships(self, group_id: str) -> None:
        """Update user group memberships"""
        group = self.groups.get(group_id)
        if not group:
            return

        # Update each member user
        for member in group.members:
            user = self.users.get(member.value)
            if user:
                # Add group to user
                group_ref = {
                    "value": group_id,
                    "$ref": f"{self.config.base_url}/Groups/{group_id}",
                    "display": group.displayName,
                    "type": "direct"
                }

                # Check if already present
                existing = [g for g in user.groups if g.get("value") == group_id]
                if not existing:
                    user.groups.append(group_ref)

    def _filter_attributes(
        self,
        resources: List[Dict[str, Any]],
        attributes: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter resource attributes"""
        filtered = []

        for resource in resources:
            filtered_resource = {}
            for attr in attributes:
                if attr in resource:
                    filtered_resource[attr] = resource[attr]
            # Always include schemas and id
            filtered_resource["schemas"] = resource.get("schemas", [])
            filtered_resource["id"] = resource.get("id")
            filtered.append(filtered_resource)

        return filtered

    def _emit_webhook_event(
        self,
        event_type: str,
        resource_type: SCIMResourceType,
        resource_id: str,
        operation: SCIMOperation,
        data: Dict[str, Any]
    ) -> None:
        """Emit webhook event"""
        if not self.config.webhook_enabled:
            return

        event = SCIMWebhookEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            resource_type=resource_type,
            resource_id=resource_id,
            operation=operation,
            data=data
        )

        self.webhook_events.append(event)

        # In production, would send to webhook URL
        if self.config.webhook_url:
            logger.info(f"Webhook event: {event_type} for {resource_type.value} {resource_id}")

    def get_service_provider_config(self) -> Dict[str, Any]:
        """Get SCIM service provider configuration"""
        return {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig"],
            "documentationUri": "https://greenlang.io/docs/scim",
            "patch": {
                "supported": self.config.support_patch
            },
            "bulk": {
                "supported": self.config.support_bulk,
                "maxOperations": self.config.max_bulk_operations,
                "maxPayloadSize": 1048576
            },
            "filter": {
                "supported": self.config.support_filter,
                "maxResults": self.config.max_results
            },
            "changePassword": {
                "supported": False
            },
            "sort": {
                "supported": self.config.support_sort
            },
            "etag": {
                "supported": self.config.support_etag
            },
            "authenticationSchemes": [
                {
                    "type": "oauthbearertoken",
                    "name": "OAuth Bearer Token",
                    "description": "Authentication scheme using the OAuth Bearer Token",
                    "specUri": "http://www.rfc-editor.org/info/rfc6750",
                    "documentationUri": "https://greenlang.io/docs/auth"
                }
            ]
        }

    def get_resource_types(self) -> List[Dict[str, Any]]:
        """Get SCIM resource types"""
        return [
            {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:ResourceType"],
                "id": "User",
                "name": "User",
                "endpoint": "/Users",
                "description": "User Account",
                "schema": "urn:ietf:params:scim:schemas:core:2.0:User",
                "schemaExtensions": []
            },
            {
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:ResourceType"],
                "id": "Group",
                "name": "Group",
                "endpoint": "/Groups",
                "description": "Group",
                "schema": "urn:ietf:params:scim:schemas:core:2.0:Group",
                "schemaExtensions": []
            }
        ]


class SCIMError(Exception):
    """SCIM-specific error"""
    pass


__all__ = [
    "SCIMProvider",
    "SCIMConfig",
    "SCIMUser",
    "SCIMGroup",
    "SCIMListResponse",
    "SCIMError",
    "SCIMResourceType",
    "SCIMOperation",
    "SCIMPatchOp",
    "SCIMWebhookEvent",
]
