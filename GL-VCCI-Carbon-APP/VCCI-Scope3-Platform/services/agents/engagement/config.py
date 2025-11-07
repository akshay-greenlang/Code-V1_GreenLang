"""
Configuration for Supplier Engagement Agent.

Includes consent rules, campaign settings, portal configuration, and email service settings.
"""
from typing import Dict, Any, List
from datetime import timedelta


# Jurisdiction configurations
EU_COUNTRIES = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE"
]

CCPA_JURISDICTIONS = ["US-CA"]  # California

# Consent settings
CONSENT_CONFIG = {
    "default_status": "pending",
    "retention_period_days": 730,  # 2 years (GDPR Article 17)
    "opt_out_grace_period_days": 1,  # Immediate opt-out honor
    "re_consent_interval_months": 24,  # Re-consent every 2 years
}

# GDPR settings
GDPR_CONFIG = {
    "requires_explicit_consent": True,
    "lawful_basis_allowed": ["consent", "contract", "legitimate_interest"],
    "data_portability_required": True,
    "right_to_erasure": True,
    "dpa_required": True,
}

# CCPA settings
CCPA_CONFIG = {
    "opt_out_model": True,
    "honor_opt_out_days": 15,  # Must honor within 15 days
    "privacy_policy_required": True,
    "do_not_sell_link_required": True,
}

# CAN-SPAM settings
CAN_SPAM_CONFIG = {
    "opt_out_model": True,
    "honor_opt_out_days": 10,  # 10 business days
    "physical_address_required": True,
    "unsubscribe_link_required": True,
    "subject_line_truthful": True,
}

# Campaign settings
CAMPAIGN_CONFIG = {
    "default_response_rate_target": 0.50,  # 50%
    "max_touches_per_sequence": 6,
    "min_touch_interval_days": 7,
    "max_campaign_duration_days": 90,
    "auto_pause_on_opt_out": True,
}

# Email sequence defaults (4-touch sequence)
DEFAULT_EMAIL_SEQUENCE = {
    "sequence_id": "default_4touch",
    "name": "Standard 4-Touch Sequence",
    "touches": [
        {
            "touch_number": 1,
            "day_offset": 0,
            "template": "touch_1_introduction",
            "subject": "Partner with us on carbon transparency"
        },
        {
            "touch_number": 2,
            "day_offset": 14,
            "template": "touch_2_reminder",
            "subject": "Your action needed: Carbon data request"
        },
        {
            "touch_number": 3,
            "day_offset": 35,
            "template": "touch_3_final_reminder",
            "subject": "Final reminder: Carbon transparency program"
        },
        {
            "touch_number": 4,
            "day_offset": 42,
            "template": "touch_4_thank_you",
            "subject": "Thank you or alternative next steps"
        }
    ]
}

# Portal settings
PORTAL_CONFIG = {
    "session_duration_hours": 24,
    "magic_link_expiry_minutes": 15,
    "max_file_size_mb": 50,
    "supported_file_types": ["csv", "xlsx", "json", "xml"],
    "max_upload_retries": 3,
    "validation_timeout_seconds": 30,
}

# Upload validation rules
VALIDATION_CONFIG = {
    "required_fields": ["supplier_id", "product_id", "emission_factor", "unit"],
    "optional_fields": ["activity_data", "uncertainty", "data_quality"],
    "min_data_quality_score": 0.60,  # 60% DQI threshold
    "allow_partial_submissions": True,
    "schema_version": "v1.0",
}

# Gamification settings
GAMIFICATION_CONFIG = {
    "leaderboard_top_n": 10,
    "badges": {
        "early_adopter": {
            "criteria": "First 10 suppliers to submit data",
            "points": 100
        },
        "data_champion": {
            "criteria": "Data quality score >= 0.90",
            "points": 150
        },
        "complete_profile": {
            "criteria": "100% field completion",
            "points": 75
        },
        "quality_leader": {
            "criteria": "Highest DQI in cohort",
            "points": 200
        },
        "fast_responder": {
            "criteria": "Response within 7 days",
            "points": 50
        }
    },
    "leaderboard_refresh_hours": 24,
}

# Email service settings
EMAIL_SERVICE_CONFIG = {
    "default_provider": "sendgrid",  # sendgrid | mailgun | aws_ses
    "from_email": "sustainability@company.com",
    "from_name": "Sustainability Team",
    "reply_to": "sustainability-reply@company.com",
    "max_retries": 3,
    "retry_delay_seconds": 60,
    "rate_limit_per_minute": 100,
    "batch_size": 50,
}

# SendGrid configuration (stub)
SENDGRID_CONFIG = {
    "api_key": "SENDGRID_API_KEY_PLACEHOLDER",
    "endpoint": "https://api.sendgrid.com/v3/mail/send",
    "tracking": {
        "open_tracking": True,
        "click_tracking": True,
        "subscription_tracking": True,
    }
}

# Mailgun configuration (stub)
MAILGUN_CONFIG = {
    "api_key": "MAILGUN_API_KEY_PLACEHOLDER",
    "domain": "mg.company.com",
    "endpoint": "https://api.mailgun.net/v3",
    "tracking": {
        "clicks": True,
        "opens": True,
    }
}

# AWS SES configuration (stub)
AWS_SES_CONFIG = {
    "access_key_id": "AWS_ACCESS_KEY_PLACEHOLDER",
    "secret_access_key": "AWS_SECRET_KEY_PLACEHOLDER",
    "region": "us-east-1",
    "configuration_set": "supplier-engagement",
}

# Localization settings
SUPPORTED_LANGUAGES = ["en", "de", "fr", "es", "zh"]

LANGUAGE_NAMES = {
    "en": "English",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "zh": "中文"
}

# Analytics settings
ANALYTICS_CONFIG = {
    "refresh_interval_minutes": 60,
    "retention_days": 365,
    "export_formats": ["json", "csv", "xlsx"],
    "dashboard_metrics": [
        "email_open_rate",
        "portal_visit_rate",
        "data_submission_rate",
        "avg_time_to_response",
        "avg_data_quality_score"
    ]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console", "file"],
    "file_path": "logs/engagement_agent.log",
    "max_bytes": 10485760,  # 10MB
    "backup_count": 5,
}

# Database configuration (for consent registry, campaigns, etc.)
DATABASE_CONFIG = {
    "type": "sqlite",  # sqlite | postgresql | mysql
    "path": "data/engagement.db",
    "connection_pool_size": 10,
    "timeout_seconds": 30,
}

# API configuration (for supplier portal)
API_CONFIG = {
    "base_url": "https://portal.company.com",
    "api_version": "v1",
    "rate_limit_per_hour": 1000,
    "cors_allowed_origins": ["https://portal.company.com"],
}

# Security settings
SECURITY_CONFIG = {
    "encryption_key": "ENCRYPTION_KEY_PLACEHOLDER",
    "jwt_secret": "JWT_SECRET_PLACEHOLDER",
    "jwt_expiry_hours": 24,
    "password_min_length": 12,
    "require_2fa": False,
}


def get_jurisdiction_config(country_code: str) -> Dict[str, Any]:
    """
    Get configuration for specific jurisdiction.

    Args:
        country_code: ISO 3166-1 country code (e.g., 'DE', 'US-CA')

    Returns:
        Jurisdiction-specific configuration
    """
    if country_code in EU_COUNTRIES:
        return {
            "jurisdiction": "GDPR",
            "regulation": "General Data Protection Regulation (EU) 2016/679",
            "config": GDPR_CONFIG,
            "requires_opt_in": True,
            "opt_out_grace_days": 1,
        }
    elif country_code in CCPA_JURISDICTIONS:
        return {
            "jurisdiction": "CCPA",
            "regulation": "California Consumer Privacy Act",
            "config": CCPA_CONFIG,
            "requires_opt_in": False,
            "opt_out_grace_days": 15,
        }
    else:
        # Default to CAN-SPAM (US federal law)
        return {
            "jurisdiction": "CAN-SPAM",
            "regulation": "Controlling the Assault of Non-Solicited Pornography And Marketing Act",
            "config": CAN_SPAM_CONFIG,
            "requires_opt_in": False,
            "opt_out_grace_days": 10,
        }


def get_email_service_config(provider: str = "sendgrid") -> Dict[str, Any]:
    """
    Get email service provider configuration.

    Args:
        provider: Email service provider (sendgrid, mailgun, aws_ses)

    Returns:
        Provider-specific configuration
    """
    configs = {
        "sendgrid": SENDGRID_CONFIG,
        "mailgun": MAILGUN_CONFIG,
        "aws_ses": AWS_SES_CONFIG,
    }
    return configs.get(provider, SENDGRID_CONFIG)
