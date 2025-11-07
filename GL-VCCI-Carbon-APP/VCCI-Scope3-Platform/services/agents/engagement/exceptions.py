"""
Custom exceptions for Supplier Engagement Agent.

Covers consent violations, campaign errors, portal issues, and integration failures.
"""


class EngagementAgentError(Exception):
    """Base exception for all Supplier Engagement Agent errors."""
    pass


# Consent-related exceptions
class ConsentError(EngagementAgentError):
    """Base exception for consent-related errors."""
    pass


class ConsentNotGrantedError(ConsentError):
    """Raised when attempting to contact supplier without valid consent."""

    def __init__(self, supplier_id: str, jurisdiction: str):
        self.supplier_id = supplier_id
        self.jurisdiction = jurisdiction
        super().__init__(
            f"Cannot contact supplier {supplier_id} - consent not granted under {jurisdiction} regulations"
        )


class OptOutViolationError(ConsentError):
    """Raised when attempting to contact opted-out supplier."""

    def __init__(self, supplier_id: str, opt_out_date: str):
        self.supplier_id = supplier_id
        self.opt_out_date = opt_out_date
        super().__init__(
            f"Cannot contact supplier {supplier_id} - opted out on {opt_out_date}"
        )


class JurisdictionNotSupportedError(ConsentError):
    """Raised when jurisdiction rules are not implemented."""

    def __init__(self, country_code: str):
        self.country_code = country_code
        super().__init__(f"Jurisdiction {country_code} not supported")


# Campaign-related exceptions
class CampaignError(EngagementAgentError):
    """Base exception for campaign-related errors."""
    pass


class CampaignNotFoundError(CampaignError):
    """Raised when campaign doesn't exist."""

    def __init__(self, campaign_id: str):
        self.campaign_id = campaign_id
        super().__init__(f"Campaign {campaign_id} not found")


class CampaignAlreadyActiveError(CampaignError):
    """Raised when attempting to start already-active campaign."""

    def __init__(self, campaign_id: str):
        self.campaign_id = campaign_id
        super().__init__(f"Campaign {campaign_id} is already active")


class InvalidEmailSequenceError(CampaignError):
    """Raised when email sequence is invalid."""

    def __init__(self, reason: str):
        super().__init__(f"Invalid email sequence: {reason}")


# Portal-related exceptions
class PortalError(EngagementAgentError):
    """Base exception for portal-related errors."""
    pass


class AuthenticationError(PortalError):
    """Raised when authentication fails."""

    def __init__(self, supplier_id: str, reason: str = "Invalid credentials"):
        self.supplier_id = supplier_id
        super().__init__(f"Authentication failed for {supplier_id}: {reason}")


class UploadValidationError(PortalError):
    """Raised when uploaded data fails validation."""

    def __init__(self, errors: list):
        self.errors = errors
        super().__init__(f"Data validation failed: {len(errors)} errors found")


class FileFormatError(PortalError):
    """Raised when uploaded file format is unsupported."""

    def __init__(self, file_type: str, supported_types: list):
        self.file_type = file_type
        self.supported_types = supported_types
        super().__init__(
            f"Unsupported file type {file_type}. Supported: {', '.join(supported_types)}"
        )


# Integration-related exceptions
class IntegrationError(EngagementAgentError):
    """Base exception for integration errors."""
    pass


class EmailServiceError(IntegrationError):
    """Raised when email service fails."""

    def __init__(self, service_name: str, reason: str):
        self.service_name = service_name
        super().__init__(f"{service_name} error: {reason}")


class EmailDeliveryError(IntegrationError):
    """Raised when email delivery fails."""

    def __init__(self, supplier_id: str, email: str, reason: str):
        self.supplier_id = supplier_id
        self.email = email
        super().__init__(
            f"Failed to deliver email to {supplier_id} ({email}): {reason}"
        )


class TemplateRenderError(IntegrationError):
    """Raised when email template rendering fails."""

    def __init__(self, template_name: str, reason: str):
        self.template_name = template_name
        super().__init__(f"Failed to render template {template_name}: {reason}")


# Data-related exceptions
class DataError(EngagementAgentError):
    """Base exception for data-related errors."""
    pass


class SupplierNotFoundError(DataError):
    """Raised when supplier doesn't exist."""

    def __init__(self, supplier_id: str):
        self.supplier_id = supplier_id
        super().__init__(f"Supplier {supplier_id} not found")


class InvalidDataQualityError(DataError):
    """Raised when data quality score is below threshold."""

    def __init__(self, score: float, threshold: float):
        self.score = score
        self.threshold = threshold
        super().__init__(
            f"Data quality score {score:.2f} below threshold {threshold:.2f}"
        )
