"""
GreenLang Authentication and Multi-tenancy Support
"""

from .tenant import TenantManager, TenantContext, Tenant, TenantQuota, TenantIsolation
from .rbac import Role, Permission, RBACManager, AccessControl
from .auth import AuthManager, AuthToken, APIKey, ServiceAccount
from .audit import AuditLogger, AuditEvent, AuditTrail

# Enterprise SSO Providers
from .saml_provider import (
    SAMLProvider, SAMLConfig, SAMLUser, SAMLSession, SAMLError,
    create_okta_config, create_azure_config, create_onelogin_config
)
from .oauth_provider import (
    OAuthProvider, OAuthConfig, OAuthUser, OAuthTokens, OAuthSession, OAuthError,
    create_google_config, create_github_config, create_azure_config as create_azure_oauth_config
)
from .ldap_provider import (
    LDAPProvider, LDAPConfig, LDAPUser, LDAPGroup, LDAPError,
    create_openldap_config, create_active_directory_config
)
from .mfa import (
    MFAManager, MFAConfig, MFAMethod, MFAStatus, MFAEnrollment,
    TOTPDevice, SMSDevice, BackupCode, MFAError
)
from .scim_provider import (
    SCIMProvider, SCIMConfig, SCIMUser, SCIMGroup, SCIMListResponse, SCIMError
)

# Phase 4: Advanced Access Control (RBAC/ABAC)
from .permissions import (
    PermissionAction, ResourceType, PermissionEffect, PermissionCondition,
    Permission as AdvancedPermission, EvaluationResult, PermissionEvaluator,
    PermissionStore, create_permission, parse_permission_string
)
from .roles import (
    BuiltInRole, Role as HierarchicalRole, RoleAssignment,
    RoleHierarchy, RoleManager
)
from .abac import (
    AttributeProvider, UserAttributeProvider, ResourceAttributeProvider,
    EnvironmentAttributeProvider, PolicyEffect, ConditionOperator,
    PolicyCondition, ABACPolicy, PolicyEvaluationResult, ABACEvaluator,
    OPAIntegration, create_policy
)
from .delegation import (
    DelegationStatus, DelegationType, DelegationConstraint,
    PermissionDelegation, DelegationManager,
    create_temporary_delegation, create_limited_use_delegation
)
from .temporal_access import (
    RecurrenceType, DayOfWeek, TimeWindow, RecurrencePattern,
    TemporalPermission, TemporalAccessManager,
    create_business_hours_permission, create_weekend_permission,
    create_temporary_permission
)
from .permission_audit import (
    PermissionChangeType, AuditSeverity as PermissionAuditSeverity,
    PermissionAuditEvent, PermissionAuditLogger,
    get_permission_audit_logger
)

__all__ = [
    # Multi-tenancy
    "TenantManager",
    "TenantContext",
    "Tenant",
    "TenantQuota",
    "TenantIsolation",
    # RBAC
    "Role",
    "Permission",
    "RBACManager",
    "AccessControl",
    # Auth
    "AuthManager",
    "AuthToken",
    "APIKey",
    "ServiceAccount",
    # Audit
    "AuditLogger",
    "AuditEvent",
    "AuditTrail",
    # SAML
    "SAMLProvider",
    "SAMLConfig",
    "SAMLUser",
    "SAMLSession",
    "SAMLError",
    "create_okta_config",
    "create_azure_config",
    "create_onelogin_config",
    # OAuth/OIDC
    "OAuthProvider",
    "OAuthConfig",
    "OAuthUser",
    "OAuthTokens",
    "OAuthSession",
    "OAuthError",
    "create_google_config",
    "create_github_config",
    "create_azure_oauth_config",
    # LDAP
    "LDAPProvider",
    "LDAPConfig",
    "LDAPUser",
    "LDAPGroup",
    "LDAPError",
    "create_openldap_config",
    "create_active_directory_config",
    # MFA
    "MFAManager",
    "MFAConfig",
    "MFAMethod",
    "MFAStatus",
    "MFAEnrollment",
    "TOTPDevice",
    "SMSDevice",
    "BackupCode",
    "MFAError",
    # SCIM
    "SCIMProvider",
    "SCIMConfig",
    "SCIMUser",
    "SCIMGroup",
    "SCIMListResponse",
    "SCIMError",
    # Advanced Permissions (Phase 4)
    "PermissionAction",
    "ResourceType",
    "PermissionEffect",
    "PermissionCondition",
    "AdvancedPermission",
    "EvaluationResult",
    "PermissionEvaluator",
    "PermissionStore",
    "create_permission",
    "parse_permission_string",
    # Role Hierarchy
    "BuiltInRole",
    "HierarchicalRole",
    "RoleAssignment",
    "RoleHierarchy",
    "RoleManager",
    # ABAC
    "AttributeProvider",
    "UserAttributeProvider",
    "ResourceAttributeProvider",
    "EnvironmentAttributeProvider",
    "PolicyEffect",
    "ConditionOperator",
    "PolicyCondition",
    "ABACPolicy",
    "PolicyEvaluationResult",
    "ABACEvaluator",
    "OPAIntegration",
    "create_policy",
    # Delegation
    "DelegationStatus",
    "DelegationType",
    "DelegationConstraint",
    "PermissionDelegation",
    "DelegationManager",
    "create_temporary_delegation",
    "create_limited_use_delegation",
    # Temporal Access
    "RecurrenceType",
    "DayOfWeek",
    "TimeWindow",
    "RecurrencePattern",
    "TemporalPermission",
    "TemporalAccessManager",
    "create_business_hours_permission",
    "create_weekend_permission",
    "create_temporary_permission",
    # Permission Audit
    "PermissionChangeType",
    "PermissionAuditSeverity",
    "PermissionAuditEvent",
    "PermissionAuditLogger",
    "get_permission_audit_logger",
]
