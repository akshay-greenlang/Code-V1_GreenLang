# -*- coding: utf-8 -*-
"""
Multi-Tenant Onboarding Workflow
===================================

6-phase tenant provisioning workflow for CSRD Enterprise Pack. Handles
new tenant setup from registration through health verification, with
full rollback support if any phase fails.

Phases:
    1. Registration: Tenant record creation, unique ID generation, plan assignment
    2. SSO Configuration: SAML/OIDC metadata parsing, IdP integration, test auth
    3. Branding Setup: Logo/color/domain validation, white-label template rendering
    4. Data Residency: Region assignment, storage provisioning, compliance check
    5. Feature Activation: License-based feature flag enablement, agent activation
    6. Health Verification: End-to-end smoke test, connectivity, readiness gate

Author: GreenLang Team
Version: 3.0.0
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class TenantPlan(str, Enum):
    """Available tenant subscription plans."""

    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class SSOProtocol(str, Enum):
    """Supported SSO protocols."""

    SAML_2_0 = "SAML_2.0"
    OIDC = "OIDC"
    NONE = "NONE"


class DataRegion(str, Enum):
    """Available data residency regions."""

    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    EU_NORTH_1 = "eu-north-1"
    US_EAST_1 = "us-east-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"


class IsolationLevel(str, Enum):
    """Tenant isolation levels."""

    SHARED = "shared"
    NAMESPACE = "namespace"
    CLUSTER = "cluster"
    PHYSICAL = "physical"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration in seconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")
    rollback_action: Optional[str] = Field(
        None, description="Rollback action to undo this phase if needed"
    )


class SSOMetadata(BaseModel):
    """SSO identity provider metadata."""

    protocol: SSOProtocol = Field(default=SSOProtocol.SAML_2_0, description="SSO protocol")
    entity_id: str = Field(default="", description="IdP entity ID")
    sso_url: str = Field(default="", description="SSO login URL")
    certificate: str = Field(default="", description="X.509 certificate (base64)")
    metadata_xml: str = Field(default="", description="Raw SAML metadata XML")
    oidc_client_id: str = Field(default="", description="OIDC client ID")
    oidc_discovery_url: str = Field(default="", description="OIDC discovery URL")


class BrandingConfig(BaseModel):
    """White-label branding configuration."""

    logo_url: str = Field(default="", description="URL to tenant logo image")
    primary_color: str = Field(default="#1a73e8", description="Primary brand hex color")
    secondary_color: str = Field(default="#ffffff", description="Secondary brand hex color")
    accent_color: str = Field(default="#34a853", description="Accent hex color")
    custom_domain: str = Field(default="", description="Custom domain for white-label portal")
    company_name: str = Field(default="", description="Company display name")
    favicon_url: str = Field(default="", description="URL to favicon")
    email_footer: str = Field(default="", description="Custom email footer text")

    @field_validator("primary_color", "secondary_color", "accent_color")
    @classmethod
    def validate_hex_color(cls, v: str) -> str:
        """Validate hex color format."""
        if v and not v.startswith("#"):
            raise ValueError(f"Color must be hex format (#RRGGBB), got: {v}")
        if v and len(v) not in (4, 7):
            raise ValueError(f"Color must be #RGB or #RRGGBB, got: {v}")
        return v


class TenantRequest(BaseModel):
    """Input for tenant onboarding workflow."""

    organization_name: str = Field(..., description="Organization display name")
    admin_email: str = Field(..., description="Primary admin email address")
    admin_name: str = Field(default="", description="Primary admin full name")
    plan: TenantPlan = Field(default=TenantPlan.ENTERPRISE, description="Subscription plan")
    region: DataRegion = Field(default=DataRegion.EU_WEST_1, description="Preferred data region")
    isolation_level: IsolationLevel = Field(
        default=IsolationLevel.NAMESPACE, description="Tenant isolation level"
    )
    sso_metadata: Optional[SSOMetadata] = Field(None, description="SSO configuration")
    branding: Optional[BrandingConfig] = Field(None, description="White-label branding config")
    requested_features: List[str] = Field(
        default_factory=list, description="Specific features to enable"
    )
    entity_count_estimate: int = Field(
        default=1, ge=1, le=10000, description="Expected number of entities"
    )
    max_users: int = Field(default=50, ge=1, description="Maximum concurrent users")
    custom_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("admin_email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email format validation."""
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError(f"Invalid email format: {v}")
        return v


class OnboardingResult(BaseModel):
    """Complete result from the tenant onboarding workflow."""

    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(default="multi_tenant_onboarding")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    tenant_id: str = Field(default="", description="Generated tenant ID")
    phases: List[PhaseResult] = Field(default_factory=list, description="Per-phase results")
    total_duration_seconds: float = Field(default=0.0, description="Total duration in seconds")
    portal_url: str = Field(default="", description="Tenant portal URL")
    api_key: str = Field(default="", description="Generated API key (masked)")
    activated_features: List[str] = Field(
        default_factory=list, description="Features activated for tenant"
    )
    health_check_passed: bool = Field(default=False, description="Health verification result")
    provenance_hash: str = Field(default="", description="SHA-256 of complete output")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class MultiTenantOnboardingWorkflow:
    """
    6-phase tenant provisioning workflow with rollback support.

    Manages the complete lifecycle of onboarding a new tenant to the
    CSRD Enterprise platform, from registration through health verification.
    If any phase fails, all completed phases are rolled back in reverse order.

    Attributes:
        workflow_id: Unique execution identifier.
        config: Optional EnterprisePackConfig for feature resolution.
        _completed_phases: Ordered list of completed phases for rollback.

    Example:
        >>> workflow = MultiTenantOnboardingWorkflow()
        >>> request = TenantRequest(
        ...     organization_name="Acme Corp",
        ...     admin_email="admin@acme.com",
        ...     plan=TenantPlan.ENTERPRISE,
        ... )
        >>> result = await workflow.execute(request)
        >>> assert result.status == WorkflowStatus.COMPLETED
        >>> assert result.tenant_id != ""
    """

    PHASE_ORDER = [
        "registration",
        "sso_configuration",
        "branding_setup",
        "data_residency",
        "feature_activation",
        "health_verification",
    ]

    def __init__(self, config: Optional[Any] = None) -> None:
        """
        Initialize the multi-tenant onboarding workflow.

        Args:
            config: Optional EnterprisePackConfig for feature and plan resolution.
        """
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._completed_phases: List[PhaseResult] = []
        self._context: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, tenant_request: TenantRequest) -> OnboardingResult:
        """
        Execute the full 6-phase onboarding workflow.

        If any phase fails, triggers rollback of all previously completed
        phases in reverse order.

        Args:
            tenant_request: Validated tenant onboarding request.

        Returns:
            OnboardingResult with tenant ID, portal URL, and phase results.

        Raises:
            ValueError: If tenant request validation fails.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting tenant onboarding workflow %s for org=%s plan=%s region=%s",
            self.workflow_id, tenant_request.organization_name,
            tenant_request.plan.value, tenant_request.region.value,
        )

        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        phase_handlers = {
            "registration": self._phase_1_registration,
            "sso_configuration": self._phase_2_sso_configuration,
            "branding_setup": self._phase_3_branding_setup,
            "data_residency": self._phase_4_data_residency,
            "feature_activation": self._phase_5_feature_activation,
            "health_verification": self._phase_6_health_verification,
        }

        try:
            for phase_name in self.PHASE_ORDER:
                handler = phase_handlers[phase_name]
                phase_start = datetime.utcnow()

                result = await handler(tenant_request)
                result.duration_seconds = (datetime.utcnow() - phase_start).total_seconds()
                phase_results.append(result)

                if result.status == PhaseStatus.FAILED:
                    self.logger.error(
                        "Phase '%s' failed: %s. Initiating rollback.",
                        phase_name, result.errors,
                    )
                    overall_status = WorkflowStatus.FAILED
                    # Rollback completed phases in reverse
                    await self._rollback(phase_results[:-1])
                    overall_status = WorkflowStatus.ROLLED_BACK
                    break

                self._completed_phases.append(result)

            if overall_status == WorkflowStatus.RUNNING:
                overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.critical(
                "Onboarding workflow %s failed: %s", self.workflow_id, str(exc),
                exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="workflow_error",
                status=PhaseStatus.FAILED,
                errors=[str(exc)],
                provenance_hash=self._hash_data({"error": str(exc)}),
            ))
            await self._rollback(self._completed_phases)
            overall_status = WorkflowStatus.ROLLED_BACK

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        tenant_id = self._context.get("tenant_id", "")
        portal_url = self._context.get("portal_url", "")
        api_key = self._context.get("api_key_masked", "")
        activated = self._context.get("activated_features", [])
        health_passed = self._context.get("health_passed", False)

        provenance = self._hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in phase_results],
        })

        self.logger.info(
            "Onboarding workflow %s finished status=%s tenant=%s in %.1fs",
            self.workflow_id, overall_status.value, tenant_id, total_duration,
        )

        return OnboardingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            tenant_id=tenant_id,
            phases=phase_results,
            total_duration_seconds=total_duration,
            portal_url=portal_url,
            api_key=api_key,
            activated_features=activated,
            health_check_passed=health_passed,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Rollback
    # -------------------------------------------------------------------------

    async def _rollback(self, completed_phases: List[PhaseResult]) -> None:
        """
        Roll back completed phases in reverse order.

        Each phase stores a rollback_action descriptor that indicates
        what undo operation to perform.

        Args:
            completed_phases: Phases to rollback in reverse order.
        """
        self.logger.warning(
            "Rolling back %d completed phases for workflow %s",
            len(completed_phases), self.workflow_id,
        )

        for phase in reversed(completed_phases):
            if phase.status != PhaseStatus.COMPLETED:
                continue

            action = phase.rollback_action or phase.phase_name
            try:
                await self._execute_rollback_action(action, phase)
                phase.status = PhaseStatus.ROLLED_BACK
                self.logger.info(
                    "Rolled back phase '%s' (action=%s)", phase.phase_name, action
                )
            except Exception as exc:
                self.logger.error(
                    "Rollback failed for phase '%s': %s",
                    phase.phase_name, str(exc), exc_info=True,
                )

    async def _execute_rollback_action(
        self, action: str, phase: PhaseResult
    ) -> None:
        """Execute a specific rollback action for a phase."""
        rollback_map = {
            "registration": self._rollback_registration,
            "sso_configuration": self._rollback_sso,
            "branding_setup": self._rollback_branding,
            "data_residency": self._rollback_data_residency,
            "feature_activation": self._rollback_features,
            "health_verification": self._rollback_health,
        }
        handler = rollback_map.get(action)
        if handler:
            await handler(phase)
        else:
            self.logger.warning("No rollback handler for action: %s", action)

    async def _rollback_registration(self, phase: PhaseResult) -> None:
        """Undo tenant registration by marking record as deleted."""
        tenant_id = phase.outputs.get("tenant_id", "")
        self.logger.info("Rolling back registration for tenant %s", tenant_id)
        # In production: mark tenant record as deleted/inactive

    async def _rollback_sso(self, phase: PhaseResult) -> None:
        """Undo SSO configuration by removing IdP integration."""
        self.logger.info("Rolling back SSO configuration")

    async def _rollback_branding(self, phase: PhaseResult) -> None:
        """Undo branding setup by removing custom assets."""
        self.logger.info("Rolling back branding configuration")

    async def _rollback_data_residency(self, phase: PhaseResult) -> None:
        """Undo data residency by deprovision storage resources."""
        self.logger.info("Rolling back data residency provisioning")

    async def _rollback_features(self, phase: PhaseResult) -> None:
        """Undo feature activation by disabling all tenant features."""
        self.logger.info("Rolling back feature activation")

    async def _rollback_health(self, phase: PhaseResult) -> None:
        """No rollback needed for health verification (read-only)."""
        self.logger.info("No rollback needed for health verification")

    # -------------------------------------------------------------------------
    # Phase 1: Registration
    # -------------------------------------------------------------------------

    async def _phase_1_registration(
        self, request: TenantRequest
    ) -> PhaseResult:
        """
        Create tenant record, generate unique ID, and assign plan.

        Validates the organization name is unique, generates a UUID-based
        tenant identifier, creates the initial tenant record with plan
        assignment and admin user provisioning.

        Steps:
            1. Validate organization name uniqueness
            2. Generate tenant ID and API key
            3. Create tenant record in registry
            4. Provision admin user with initial credentials
        """
        phase_name = "registration"
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Step 1: Validate uniqueness
        is_unique = await self._check_org_uniqueness(request.organization_name)
        if not is_unique:
            errors.append(f"Organization name already exists: {request.organization_name}")
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                outputs=outputs,
                errors=errors,
                provenance_hash=self._hash_data({"error": "not_unique"}),
            )

        # Step 2: Generate identifiers
        tenant_id = f"tenant-{uuid.uuid4().hex[:12]}"
        api_key = f"gl_ent_{uuid.uuid4().hex}"
        api_key_masked = f"gl_ent_****{api_key[-8:]}"

        outputs["tenant_id"] = tenant_id
        outputs["api_key_masked"] = api_key_masked
        outputs["plan"] = request.plan.value
        outputs["organization_name"] = request.organization_name

        # Step 3: Create tenant record
        tenant_record = await self._create_tenant_record(
            tenant_id, request.organization_name, request.plan,
            request.admin_email, request.max_users,
        )
        outputs["record_created"] = tenant_record.get("created", False)

        # Step 4: Provision admin user
        admin_user = await self._provision_admin_user(
            tenant_id, request.admin_email, request.admin_name
        )
        outputs["admin_user_id"] = admin_user.get("user_id", "")
        outputs["admin_provisioned"] = admin_user.get("provisioned", False)

        self._context["tenant_id"] = tenant_id
        self._context["api_key_masked"] = api_key_masked

        provenance = self._hash_data(outputs)
        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
            rollback_action="registration",
        )

    # -------------------------------------------------------------------------
    # Phase 2: SSO Configuration
    # -------------------------------------------------------------------------

    async def _phase_2_sso_configuration(
        self, request: TenantRequest
    ) -> PhaseResult:
        """
        Configure SSO integration with tenant identity provider.

        Parses SAML/OIDC metadata, validates certificates, establishes
        trust relationship with the IdP, and performs a test authentication
        flow.

        Steps:
            1. Parse SSO metadata (SAML XML or OIDC discovery)
            2. Validate X.509 certificate chain
            3. Register service provider with IdP
            4. Execute test authentication flow
        """
        phase_name = "sso_configuration"
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        tenant_id = self._context.get("tenant_id", "")

        sso = request.sso_metadata
        if sso is None or sso.protocol == SSOProtocol.NONE:
            outputs["sso_enabled"] = False
            outputs["auth_method"] = "email_password"
            warnings.append("No SSO configured; using email/password authentication")
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.COMPLETED,
                outputs=outputs,
                warnings=warnings,
                provenance_hash=self._hash_data(outputs),
                rollback_action="sso_configuration",
            )

        # Step 1: Parse SSO metadata
        if sso.protocol == SSOProtocol.SAML_2_0:
            parsed = self._parse_saml_metadata(sso.metadata_xml)
            outputs["sso_protocol"] = "SAML_2.0"
            outputs["idp_entity_id"] = parsed.get("entity_id", sso.entity_id)
            outputs["sso_url"] = parsed.get("sso_url", sso.sso_url)
        elif sso.protocol == SSOProtocol.OIDC:
            parsed = await self._discover_oidc(sso.oidc_discovery_url)
            outputs["sso_protocol"] = "OIDC"
            outputs["oidc_issuer"] = parsed.get("issuer", "")
            outputs["oidc_authorization_endpoint"] = parsed.get("authorization_endpoint", "")

        # Step 2: Validate certificate
        if sso.certificate:
            cert_valid = self._validate_certificate(sso.certificate)
            outputs["certificate_valid"] = cert_valid
            if not cert_valid:
                warnings.append("SSO certificate validation warning: check expiry")
        else:
            outputs["certificate_valid"] = True

        # Step 3: Register SP
        sp_registration = await self._register_service_provider(
            tenant_id, sso.protocol, outputs
        )
        outputs["sp_registered"] = sp_registration.get("registered", False)
        outputs["sp_entity_id"] = sp_registration.get("sp_entity_id", "")

        # Step 4: Test authentication
        test_auth = await self._test_sso_authentication(tenant_id, sso.protocol)
        outputs["test_auth_success"] = test_auth.get("success", False)
        outputs["sso_enabled"] = True

        if not test_auth.get("success", False):
            warnings.append("SSO test authentication did not succeed; verify IdP configuration")

        provenance = self._hash_data(outputs)
        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=provenance,
            rollback_action="sso_configuration",
        )

    # -------------------------------------------------------------------------
    # Phase 3: Branding Setup
    # -------------------------------------------------------------------------

    async def _phase_3_branding_setup(
        self, request: TenantRequest
    ) -> PhaseResult:
        """
        Configure white-label branding for the tenant portal.

        Validates logo image dimensions and format, verifies custom domain
        DNS configuration, applies color theme, and renders branded templates.

        Steps:
            1. Validate and upload logo asset
            2. Validate color scheme accessibility (WCAG contrast)
            3. Configure custom domain DNS (CNAME verification)
            4. Render branded portal templates
        """
        phase_name = "branding_setup"
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        tenant_id = self._context.get("tenant_id", "")

        branding = request.branding
        if branding is None:
            branding = BrandingConfig(company_name=request.organization_name)

        # Step 1: Logo validation
        if branding.logo_url:
            logo_valid = await self._validate_logo(branding.logo_url)
            outputs["logo_uploaded"] = logo_valid.get("valid", False)
            outputs["logo_dimensions"] = logo_valid.get("dimensions", "")
            if not logo_valid.get("valid", False):
                warnings.append(f"Logo validation issue: {logo_valid.get('reason', '')}")
        else:
            outputs["logo_uploaded"] = False
            outputs["logo_note"] = "Default GreenLang logo will be used"

        # Step 2: Color accessibility check
        contrast_check = self._check_color_contrast(
            branding.primary_color, branding.secondary_color
        )
        outputs["wcag_contrast_ratio"] = contrast_check.get("ratio", 0.0)
        outputs["wcag_compliant"] = contrast_check.get("aa_compliant", True)
        if not contrast_check.get("aa_compliant", True):
            warnings.append(
                f"Color contrast ratio {contrast_check.get('ratio', 0):.1f} "
                f"does not meet WCAG AA requirements (4.5:1)"
            )

        # Step 3: Custom domain
        if branding.custom_domain:
            domain_check = await self._verify_custom_domain(branding.custom_domain, tenant_id)
            outputs["custom_domain"] = branding.custom_domain
            outputs["domain_verified"] = domain_check.get("verified", False)
            outputs["cname_target"] = domain_check.get("cname_target", "")
            if not domain_check.get("verified", False):
                warnings.append(
                    f"Custom domain CNAME not yet configured. "
                    f"Point {branding.custom_domain} to {domain_check.get('cname_target', '')}"
                )
        else:
            outputs["custom_domain"] = ""
            outputs["portal_subdomain"] = f"{tenant_id}.greenlang.io"

        self._context["portal_url"] = (
            f"https://{branding.custom_domain}"
            if branding.custom_domain and outputs.get("domain_verified")
            else f"https://{tenant_id}.greenlang.io"
        )

        # Step 4: Render templates
        render_result = await self._render_branded_templates(tenant_id, branding)
        outputs["templates_rendered"] = render_result.get("count", 0)
        outputs["branding_applied"] = True

        provenance = self._hash_data(outputs)
        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=provenance,
            rollback_action="branding_setup",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Data Residency
    # -------------------------------------------------------------------------

    async def _phase_4_data_residency(
        self, request: TenantRequest
    ) -> PhaseResult:
        """
        Assign data residency region and provision storage resources.

        Validates the requested region complies with the tenant's regulatory
        requirements (GDPR, Schrems II), provisions database schema, object
        storage bucket, and Redis namespace.

        Steps:
            1. Validate region compliance (GDPR/Schrems II for EU tenants)
            2. Provision database schema with tenant isolation
            3. Create object storage bucket with encryption
            4. Set up Redis namespace for caching
        """
        phase_name = "data_residency"
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        tenant_id = self._context.get("tenant_id", "")

        # Step 1: Region compliance
        compliance_check = await self._check_region_compliance(
            request.region, request.organization_name
        )
        outputs["region"] = request.region.value
        outputs["gdpr_compliant"] = compliance_check.get("gdpr_compliant", True)
        outputs["schrems_ii_compliant"] = compliance_check.get("schrems_ii_compliant", True)

        if not compliance_check.get("gdpr_compliant", True):
            errors.append(
                f"Region {request.region.value} does not meet GDPR requirements "
                f"for organization in {compliance_check.get('jurisdiction', 'unknown')}"
            )
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                outputs=outputs,
                errors=errors,
                provenance_hash=self._hash_data(outputs),
            )

        # Step 2: Database schema provisioning
        db_provision = await self._provision_database_schema(
            tenant_id, request.region, request.isolation_level
        )
        outputs["db_schema"] = db_provision.get("schema_name", "")
        outputs["db_provisioned"] = db_provision.get("provisioned", False)
        outputs["isolation_level"] = request.isolation_level.value

        # Step 3: Object storage
        storage_provision = await self._provision_object_storage(
            tenant_id, request.region
        )
        outputs["storage_bucket"] = storage_provision.get("bucket_name", "")
        outputs["storage_encrypted"] = storage_provision.get("encrypted", True)
        outputs["encryption_algorithm"] = storage_provision.get("algorithm", "AES-256-GCM")

        # Step 4: Redis namespace
        redis_provision = await self._provision_redis_namespace(
            tenant_id, request.region
        )
        outputs["redis_namespace"] = redis_provision.get("namespace", "")
        outputs["redis_provisioned"] = redis_provision.get("provisioned", False)

        provenance = self._hash_data(outputs)
        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=provenance,
            rollback_action="data_residency",
        )

    # -------------------------------------------------------------------------
    # Phase 5: Feature Activation
    # -------------------------------------------------------------------------

    async def _phase_5_feature_activation(
        self, request: TenantRequest
    ) -> PhaseResult:
        """
        Activate features and agents based on subscription plan.

        Maps the tenant's subscription plan to available features, activates
        feature flags, enables agent access, and configures rate limits.

        Steps:
            1. Resolve plan-based feature entitlements
            2. Activate feature flags for tenant
            3. Enable agent access permissions
            4. Configure rate limits and quotas
        """
        phase_name = "feature_activation"
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        tenant_id = self._context.get("tenant_id", "")

        # Step 1: Resolve feature entitlements
        plan_features = self._resolve_plan_features(request.plan)
        requested = set(request.requested_features)
        available = set(plan_features)
        unavailable = requested - available
        if unavailable:
            warnings.append(
                f"Features not available in {request.plan.value} plan: {sorted(unavailable)}"
            )

        features_to_activate = list(available | (requested & available))
        outputs["plan_features"] = plan_features
        outputs["requested_features"] = list(requested)
        outputs["features_to_activate"] = features_to_activate

        # Step 2: Activate feature flags
        activation_result = await self._activate_feature_flags(
            tenant_id, features_to_activate
        )
        outputs["flags_activated"] = activation_result.get("activated_count", 0)
        outputs["activated_features"] = activation_result.get("features", [])
        self._context["activated_features"] = activation_result.get("features", [])

        # Step 3: Agent access
        agent_access = await self._enable_agent_access(tenant_id, request.plan)
        outputs["agents_enabled"] = agent_access.get("agent_count", 0)
        outputs["agent_ids"] = agent_access.get("agent_ids", [])

        # Step 4: Rate limits
        rate_limits = await self._configure_rate_limits(
            tenant_id, request.plan, request.max_users
        )
        outputs["rate_limit_rps"] = rate_limits.get("rps", 0)
        outputs["concurrent_users_limit"] = rate_limits.get("concurrent_users", 0)
        outputs["storage_quota_gb"] = rate_limits.get("storage_quota_gb", 0)

        provenance = self._hash_data(outputs)
        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=provenance,
            rollback_action="feature_activation",
        )

    # -------------------------------------------------------------------------
    # Phase 6: Health Verification
    # -------------------------------------------------------------------------

    async def _phase_6_health_verification(
        self, request: TenantRequest
    ) -> PhaseResult:
        """
        End-to-end health verification and readiness gate.

        Runs smoke tests across all provisioned resources to verify the
        tenant environment is fully operational: database connectivity,
        storage access, SSO flow, API endpoint, and agent availability.

        Steps:
            1. Test database connectivity and schema access
            2. Test object storage read/write
            3. Test Redis connectivity
            4. Test API endpoint with tenant API key
            5. Verify at least one agent is reachable
            6. Generate readiness report
        """
        phase_name = "health_verification"
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        tenant_id = self._context.get("tenant_id", "")

        checks: Dict[str, bool] = {}

        # Step 1: Database
        db_ok = await self._check_database_health(tenant_id)
        checks["database"] = db_ok
        if not db_ok:
            errors.append("Database connectivity check failed")

        # Step 2: Object storage
        storage_ok = await self._check_storage_health(tenant_id)
        checks["object_storage"] = storage_ok
        if not storage_ok:
            errors.append("Object storage health check failed")

        # Step 3: Redis
        redis_ok = await self._check_redis_health(tenant_id)
        checks["redis"] = redis_ok
        if not redis_ok:
            warnings.append("Redis connectivity check failed (non-critical)")

        # Step 4: API endpoint
        api_ok = await self._check_api_health(tenant_id)
        checks["api_endpoint"] = api_ok
        if not api_ok:
            errors.append("API endpoint health check failed")

        # Step 5: Agent availability
        agent_ok = await self._check_agent_availability(tenant_id)
        checks["agent_availability"] = agent_ok
        if not agent_ok:
            warnings.append("Agent availability check incomplete")

        outputs["health_checks"] = checks
        outputs["all_passed"] = all(checks.values())
        outputs["critical_passed"] = checks.get("database", False) and checks.get("api_endpoint", False)

        self._context["health_passed"] = outputs["critical_passed"]

        # Step 6: Readiness report
        outputs["readiness_score"] = sum(1 for v in checks.values() if v) / len(checks) * 100.0
        outputs["ready_for_use"] = outputs["critical_passed"]

        status = PhaseStatus.COMPLETED if outputs["critical_passed"] else PhaseStatus.FAILED
        if not outputs["critical_passed"]:
            errors.append("Critical health checks did not pass")

        provenance = self._hash_data(outputs)
        return PhaseResult(
            phase_name=phase_name,
            status=status,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=provenance,
            rollback_action="health_verification",
        )

    # -------------------------------------------------------------------------
    # Agent Simulation Stubs
    # -------------------------------------------------------------------------

    async def _check_org_uniqueness(self, org_name: str) -> bool:
        """Check organization name is unique in tenant registry."""
        return True

    async def _create_tenant_record(
        self, tenant_id: str, org_name: str, plan: TenantPlan,
        admin_email: str, max_users: int
    ) -> Dict[str, Any]:
        """Create tenant record in the registry."""
        return {"created": True, "tenant_id": tenant_id}

    async def _provision_admin_user(
        self, tenant_id: str, email: str, name: str
    ) -> Dict[str, Any]:
        """Provision the initial admin user."""
        return {"user_id": f"user-{uuid.uuid4().hex[:8]}", "provisioned": True}

    def _parse_saml_metadata(self, xml_str: str) -> Dict[str, Any]:
        """Parse SAML metadata XML and extract IdP configuration."""
        if not xml_str:
            return {"entity_id": "", "sso_url": ""}
        try:
            root = ElementTree.fromstring(xml_str)
            entity_id = root.attrib.get("entityID", "")
            sso_url = ""
            for elem in root.iter():
                if "SingleSignOnService" in elem.tag:
                    sso_url = elem.attrib.get("Location", "")
                    break
            return {"entity_id": entity_id, "sso_url": sso_url}
        except ElementTree.ParseError:
            return {"entity_id": "", "sso_url": "", "parse_error": True}

    async def _discover_oidc(self, discovery_url: str) -> Dict[str, Any]:
        """Discover OIDC configuration from well-known endpoint."""
        return {
            "issuer": discovery_url.replace("/.well-known/openid-configuration", ""),
            "authorization_endpoint": f"{discovery_url}/authorize",
            "token_endpoint": f"{discovery_url}/token",
        }

    def _validate_certificate(self, cert_b64: str) -> bool:
        """Validate X.509 certificate (basic check)."""
        return len(cert_b64) > 100

    async def _register_service_provider(
        self, tenant_id: str, protocol: SSOProtocol, metadata: Dict
    ) -> Dict[str, Any]:
        """Register GreenLang as SP with the tenant IdP."""
        return {
            "registered": True,
            "sp_entity_id": f"https://greenlang.io/sso/{tenant_id}",
        }

    async def _test_sso_authentication(
        self, tenant_id: str, protocol: SSOProtocol
    ) -> Dict[str, Any]:
        """Execute test SSO authentication flow."""
        return {"success": True, "response_time_ms": 250}

    async def _validate_logo(self, url: str) -> Dict[str, Any]:
        """Validate logo image dimensions and format."""
        return {"valid": True, "dimensions": "512x512", "format": "PNG"}

    def _check_color_contrast(
        self, primary: str, secondary: str
    ) -> Dict[str, Any]:
        """Check WCAG color contrast ratio between primary and secondary."""
        return {"ratio": 7.5, "aa_compliant": True, "aaa_compliant": True}

    async def _verify_custom_domain(
        self, domain: str, tenant_id: str
    ) -> Dict[str, Any]:
        """Verify custom domain CNAME configuration."""
        return {
            "verified": False,
            "cname_target": f"{tenant_id}.cdn.greenlang.io",
        }

    async def _render_branded_templates(
        self, tenant_id: str, branding: BrandingConfig
    ) -> Dict[str, Any]:
        """Render white-label portal templates with branding."""
        return {"count": 12, "templates": ["login", "dashboard", "report", "settings"]}

    async def _check_region_compliance(
        self, region: DataRegion, org_name: str
    ) -> Dict[str, Any]:
        """Check regulatory compliance for data residency region."""
        eu_regions = {DataRegion.EU_WEST_1, DataRegion.EU_CENTRAL_1, DataRegion.EU_NORTH_1}
        return {
            "gdpr_compliant": region in eu_regions,
            "schrems_ii_compliant": region in eu_regions,
            "jurisdiction": "EU" if region in eu_regions else "non-EU",
        }

    async def _provision_database_schema(
        self, tenant_id: str, region: DataRegion, isolation: IsolationLevel
    ) -> Dict[str, Any]:
        """Provision isolated database schema for tenant."""
        return {
            "schema_name": f"gl_tenant_{tenant_id.replace('-', '_')}",
            "provisioned": True,
            "isolation": isolation.value,
        }

    async def _provision_object_storage(
        self, tenant_id: str, region: DataRegion
    ) -> Dict[str, Any]:
        """Provision encrypted object storage bucket."""
        return {
            "bucket_name": f"gl-enterprise-{tenant_id}",
            "encrypted": True,
            "algorithm": "AES-256-GCM",
            "region": region.value,
        }

    async def _provision_redis_namespace(
        self, tenant_id: str, region: DataRegion
    ) -> Dict[str, Any]:
        """Provision Redis namespace for tenant caching."""
        return {"namespace": f"gl:{tenant_id}", "provisioned": True}

    def _resolve_plan_features(self, plan: TenantPlan) -> List[str]:
        """Resolve available features based on subscription plan."""
        base = [
            "annual_reporting", "data_collection", "emissions_calculation",
            "materiality_assessment", "audit_preparation",
        ]
        pro = base + [
            "consolidated_reporting", "cross_framework", "scenario_analysis",
            "continuous_compliance", "benchmarking",
        ]
        enterprise = pro + [
            "enterprise_reporting", "ai_quality", "narrative_gen",
            "iot_integration", "predictive_compliance", "supply_chain_assessment",
            "regulatory_filing", "custom_workflows", "auditor_collaboration",
            "multi_tenant", "white_label", "api_graphql", "carbon_credits",
        ]
        return {
            TenantPlan.STARTER: base,
            TenantPlan.PROFESSIONAL: pro,
            TenantPlan.ENTERPRISE: enterprise,
        }.get(plan, base)

    async def _activate_feature_flags(
        self, tenant_id: str, features: List[str]
    ) -> Dict[str, Any]:
        """Activate feature flags for tenant."""
        return {"activated_count": len(features), "features": features}

    async def _enable_agent_access(
        self, tenant_id: str, plan: TenantPlan
    ) -> Dict[str, Any]:
        """Enable agent access permissions based on plan."""
        counts = {TenantPlan.STARTER: 47, TenantPlan.PROFESSIONAL: 90, TenantPlan.ENTERPRISE: 135}
        return {"agent_count": counts.get(plan, 47), "agent_ids": []}

    async def _configure_rate_limits(
        self, tenant_id: str, plan: TenantPlan, max_users: int
    ) -> Dict[str, Any]:
        """Configure rate limits and quotas for tenant."""
        limits = {
            TenantPlan.STARTER: {"rps": 100, "storage_quota_gb": 50},
            TenantPlan.PROFESSIONAL: {"rps": 500, "storage_quota_gb": 500},
            TenantPlan.ENTERPRISE: {"rps": 5000, "storage_quota_gb": 5000},
        }
        plan_limits = limits.get(plan, limits[TenantPlan.STARTER])
        return {**plan_limits, "concurrent_users": max_users}

    async def _check_database_health(self, tenant_id: str) -> bool:
        """Check database connectivity for tenant."""
        return True

    async def _check_storage_health(self, tenant_id: str) -> bool:
        """Check object storage health for tenant."""
        return True

    async def _check_redis_health(self, tenant_id: str) -> bool:
        """Check Redis connectivity for tenant."""
        return True

    async def _check_api_health(self, tenant_id: str) -> bool:
        """Check API endpoint health for tenant."""
        return True

    async def _check_agent_availability(self, tenant_id: str) -> bool:
        """Check agent availability for tenant."""
        return True

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _hash_data(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
