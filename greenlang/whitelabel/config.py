# -*- coding: utf-8 -*-
"""
White-Label Configuration for GreenLang

This module provides white-label branding support for partners,
allowing them to customize the UI with their own branding.

Features:
- Custom branding (logo, colors, fonts)
- Custom domain support
- SSL certificate management
- Theme customization
- Partner portal branding
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import logging

from sqlalchemy import Column, String, DateTime, Boolean, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship
from pydantic import BaseModel, HttpUrl, validator
from fastapi import FastAPI, HTTPException, Depends, status
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)
Base = declarative_base()


class DomainStatus(str, Enum):
    """Custom domain status"""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    SUSPENDED = "SUSPENDED"


class ThemeMode(str, Enum):
    """Theme mode"""
    LIGHT = "LIGHT"
    DARK = "DARK"
    AUTO = "AUTO"


# Database Models
class WhiteLabelConfigModel(Base):
    """White-label configuration model"""
    __tablename__ = "whitelabel_configs"

    id = Column(String, primary_key=True)
    partner_id = Column(String, ForeignKey("partners.id"), unique=True, nullable=False)

    # Branding
    brand_name = Column(String, nullable=False)
    logo_url = Column(String, nullable=True)
    favicon_url = Column(String, nullable=True)
    logo_dark_url = Column(String, nullable=True)  # For dark mode

    # Colors
    primary_color = Column(String, default="#1E40AF")  # Blue
    secondary_color = Column(String, default="#10B981")  # Green
    accent_color = Column(String, default="#F59E0B")  # Amber
    background_color = Column(String, default="#FFFFFF")
    text_color = Column(String, default="#1F2937")
    error_color = Column(String, default="#EF4444")
    success_color = Column(String, default="#10B981")
    warning_color = Column(String, default="#F59E0B")

    # Typography
    font_family = Column(String, default="Inter, sans-serif")
    heading_font = Column(String, default="Inter, sans-serif")

    # Custom domain
    custom_domain = Column(String, nullable=True, unique=True)
    domain_status = Column(String, default=DomainStatus.PENDING)
    ssl_enabled = Column(Boolean, default=False)
    ssl_cert_path = Column(String, nullable=True)

    # Contact & Legal
    support_email = Column(String, nullable=True)
    support_url = Column(String, nullable=True)
    terms_url = Column(String, nullable=True)
    privacy_url = Column(String, nullable=True)
    company_name = Column(String, nullable=True)
    company_address = Column(Text, nullable=True)

    # Theme settings
    theme_mode = Column(String, default=ThemeMode.LIGHT)
    custom_css = Column(Text, nullable=True)
    custom_js = Column(Text, nullable=True)

    # Feature flags
    show_powered_by = Column(Boolean, default=True)
    allow_user_signup = Column(Boolean, default=True)
    enable_analytics = Column(Boolean, default=True)

    # Additional customization
    metadata = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DomainConfigModel(Base):
    """Custom domain configuration"""
    __tablename__ = "domain_configs"

    id = Column(String, primary_key=True)
    partner_id = Column(String, ForeignKey("partners.id"), nullable=False)
    domain = Column(String, unique=True, nullable=False)
    status = Column(String, default=DomainStatus.PENDING)

    # DNS configuration
    cname_target = Column(String, nullable=False)
    dns_verified = Column(Boolean, default=False)
    dns_verified_at = Column(DateTime, nullable=True)

    # SSL configuration
    ssl_enabled = Column(Boolean, default=False)
    ssl_provider = Column(String, default="letsencrypt")
    ssl_cert_issued_at = Column(DateTime, nullable=True)
    ssl_cert_expires_at = Column(DateTime, nullable=True)
    ssl_auto_renew = Column(Boolean, default=True)

    # Metadata
    verification_token = Column(String, nullable=True)
    metadata = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Pydantic Models
class ColorScheme(BaseModel):
    """Color scheme configuration"""
    primary_color: str = "#1E40AF"
    secondary_color: str = "#10B981"
    accent_color: str = "#F59E0B"
    background_color: str = "#FFFFFF"
    text_color: str = "#1F2937"
    error_color: str = "#EF4444"
    success_color: str = "#10B981"
    warning_color: str = "#F59E0B"

    @validator('*')
    def validate_color(cls, v):
        if not v.startswith('#') or len(v) not in [4, 7]:
            raise ValueError('Invalid hex color format')
        return v


class Typography(BaseModel):
    """Typography configuration"""
    font_family: str = "Inter, sans-serif"
    heading_font: str = "Inter, sans-serif"
    font_size_base: str = "16px"
    line_height: str = "1.5"


class WhiteLabelCreate(BaseModel):
    """Schema for creating white-label config"""
    brand_name: str
    logo_url: Optional[HttpUrl] = None
    favicon_url: Optional[HttpUrl] = None
    colors: Optional[ColorScheme] = None
    typography: Optional[Typography] = None
    custom_domain: Optional[str] = None
    support_email: Optional[str] = None
    terms_url: Optional[HttpUrl] = None
    privacy_url: Optional[HttpUrl] = None


class WhiteLabelUpdate(BaseModel):
    """Schema for updating white-label config"""
    brand_name: Optional[str] = None
    logo_url: Optional[HttpUrl] = None
    favicon_url: Optional[HttpUrl] = None
    logo_dark_url: Optional[HttpUrl] = None
    colors: Optional[ColorScheme] = None
    typography: Optional[Typography] = None
    custom_domain: Optional[str] = None
    support_email: Optional[str] = None
    support_url: Optional[HttpUrl] = None
    terms_url: Optional[HttpUrl] = None
    privacy_url: Optional[HttpUrl] = None
    company_name: Optional[str] = None
    company_address: Optional[str] = None
    theme_mode: Optional[ThemeMode] = None
    custom_css: Optional[str] = None
    show_powered_by: Optional[bool] = None


class WhiteLabelResponse(BaseModel):
    """Schema for white-label response"""
    id: str
    partner_id: str
    brand_name: str
    logo_url: Optional[str]
    favicon_url: Optional[str]
    logo_dark_url: Optional[str]
    colors: ColorScheme
    typography: Typography
    custom_domain: Optional[str]
    domain_status: Optional[str]
    ssl_enabled: bool
    support_email: Optional[str]
    terms_url: Optional[str]
    privacy_url: Optional[str]
    theme_mode: str
    show_powered_by: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DomainConfigCreate(BaseModel):
    """Schema for creating domain config"""
    domain: str
    cname_target: str = "partners.greenlang.com"

    @validator('domain')
    def validate_domain(cls, v):
        # Basic domain validation
        if not v or '.' not in v:
            raise ValueError('Invalid domain format')
        return v.lower()


class DomainConfigResponse(BaseModel):
    """Schema for domain config response"""
    id: str
    partner_id: str
    domain: str
    status: DomainStatus
    cname_target: str
    dns_verified: bool
    ssl_enabled: bool
    ssl_cert_expires_at: Optional[datetime]
    verification_token: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# Dataclasses
@dataclass
class WhiteLabelConfig:
    """White-label configuration dataclass"""
    id: str
    partner_id: str
    brand_name: str
    logo_url: Optional[str]
    primary_color: str
    secondary_color: str
    custom_domain: Optional[str]
    support_email: Optional[str]
    terms_url: Optional[str]
    privacy_url: Optional[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class ThemeConfig:
    """Theme configuration for rendering"""
    colors: Dict[str, str]
    typography: Dict[str, str]
    custom_css: Optional[str] = None
    logo_url: Optional[str] = None
    brand_name: str = "GreenLang"


class WhiteLabelManager:
    """
    Manages white-label configurations
    """

    def __init__(self, db: Session):
        self.db = db

    def create_config(
        self,
        partner_id: str,
        config_data: WhiteLabelCreate
    ) -> WhiteLabelConfigModel:
        """
        Create white-label configuration for partner

        Args:
            partner_id: Partner ID
            config_data: Configuration data

        Returns:
            WhiteLabelConfigModel
        """
        import uuid

        # Check if config already exists
        existing = self.db.query(WhiteLabelConfigModel).filter(
            WhiteLabelConfigModel.partner_id == partner_id
        ).first()

        if existing:
            raise ValueError(f"White-label config already exists for partner {partner_id}")

        # Create config
        config = WhiteLabelConfigModel(
            id=f"wl_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:16]}",
            partner_id=partner_id,
            brand_name=config_data.brand_name,
            logo_url=str(config_data.logo_url) if config_data.logo_url else None,
            favicon_url=str(config_data.favicon_url) if config_data.favicon_url else None,
            support_email=config_data.support_email,
            terms_url=str(config_data.terms_url) if config_data.terms_url else None,
            privacy_url=str(config_data.privacy_url) if config_data.privacy_url else None,
            custom_domain=config_data.custom_domain
        )

        # Apply colors if provided
        if config_data.colors:
            config.primary_color = config_data.colors.primary_color
            config.secondary_color = config_data.colors.secondary_color
            config.accent_color = config_data.colors.accent_color
            config.background_color = config_data.colors.background_color
            config.text_color = config_data.colors.text_color

        # Apply typography if provided
        if config_data.typography:
            config.font_family = config_data.typography.font_family
            config.heading_font = config_data.typography.heading_font

        self.db.add(config)
        self.db.commit()
        self.db.refresh(config)

        logger.info(f"Created white-label config for partner {partner_id}")

        return config

    def get_config(self, partner_id: str) -> Optional[WhiteLabelConfigModel]:
        """Get white-label config for partner"""
        return self.db.query(WhiteLabelConfigModel).filter(
            WhiteLabelConfigModel.partner_id == partner_id
        ).first()

    def update_config(
        self,
        partner_id: str,
        updates: WhiteLabelUpdate
    ) -> WhiteLabelConfigModel:
        """Update white-label configuration"""
        config = self.get_config(partner_id)

        if not config:
            raise ValueError(f"White-label config not found for partner {partner_id}")

        # Update fields
        if updates.brand_name:
            config.brand_name = updates.brand_name
        if updates.logo_url:
            config.logo_url = str(updates.logo_url)
        if updates.favicon_url:
            config.favicon_url = str(updates.favicon_url)
        if updates.logo_dark_url:
            config.logo_dark_url = str(updates.logo_dark_url)

        # Update colors
        if updates.colors:
            config.primary_color = updates.colors.primary_color
            config.secondary_color = updates.colors.secondary_color
            config.accent_color = updates.colors.accent_color
            config.background_color = updates.colors.background_color
            config.text_color = updates.colors.text_color
            config.error_color = updates.colors.error_color
            config.success_color = updates.colors.success_color
            config.warning_color = updates.colors.warning_color

        # Update typography
        if updates.typography:
            config.font_family = updates.typography.font_family
            config.heading_font = updates.typography.heading_font

        # Update other fields
        if updates.support_email:
            config.support_email = updates.support_email
        if updates.terms_url:
            config.terms_url = str(updates.terms_url)
        if updates.privacy_url:
            config.privacy_url = str(updates.privacy_url)
        if updates.company_name:
            config.company_name = updates.company_name
        if updates.company_address:
            config.company_address = updates.company_address
        if updates.theme_mode:
            config.theme_mode = updates.theme_mode
        if updates.custom_css is not None:
            config.custom_css = updates.custom_css
        if updates.show_powered_by is not None:
            config.show_powered_by = updates.show_powered_by

        config.updated_at = DeterministicClock.utcnow()
        self.db.commit()
        self.db.refresh(config)

        logger.info(f"Updated white-label config for partner {partner_id}")

        return config

    def generate_theme_css(self, partner_id: str) -> str:
        """
        Generate CSS for partner's theme

        Args:
            partner_id: Partner ID

        Returns:
            CSS string
        """
        config = self.get_config(partner_id)

        if not config:
            return ""

        css = f"""
        :root {{
            --color-primary: {config.primary_color};
            --color-secondary: {config.secondary_color};
            --color-accent: {config.accent_color};
            --color-background: {config.background_color};
            --color-text: {config.text_color};
            --color-error: {config.error_color};
            --color-success: {config.success_color};
            --color-warning: {config.warning_color};
            --font-family: {config.font_family};
            --heading-font: {config.heading_font};
        }}

        body {{
            font-family: var(--font-family);
            background-color: var(--color-background);
            color: var(--color-text);
        }}

        h1, h2, h3, h4, h5, h6 {{
            font-family: var(--heading-font);
        }}

        .btn-primary {{
            background-color: var(--color-primary);
            border-color: var(--color-primary);
        }}

        .btn-secondary {{
            background-color: var(--color-secondary);
            border-color: var(--color-secondary);
        }}

        .text-primary {{ color: var(--color-primary); }}
        .bg-primary {{ background-color: var(--color-primary); }}

        {config.custom_css or ''}
        """

        return css.strip()

    def configure_custom_domain(
        self,
        partner_id: str,
        domain: str
    ) -> DomainConfigModel:
        """
        Configure custom domain for partner

        Args:
            partner_id: Partner ID
            domain: Custom domain

        Returns:
            DomainConfigModel
        """
        import uuid
        import secrets

        # Create domain config
        domain_config = DomainConfigModel(
            id=f"dom_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:16]}",
            partner_id=partner_id,
            domain=domain.lower(),
            cname_target="partners.greenlang.com",
            verification_token=secrets.token_hex(16)
        )

        self.db.add(domain_config)
        self.db.commit()
        self.db.refresh(domain_config)

        logger.info(f"Created domain config for partner {partner_id}: {domain}")

        return domain_config

    def verify_domain(self, domain_id: str) -> bool:
        """
        Verify DNS configuration for custom domain

        Args:
            domain_id: Domain config ID

        Returns:
            True if verified
        """
        import socket

        domain_config = self.db.query(DomainConfigModel).filter(
            DomainConfigModel.id == domain_id
        ).first()

        if not domain_config:
            return False

        try:
            # Check CNAME record
            cname = socket.getfqdn(domain_config.domain)
            if domain_config.cname_target in cname:
                domain_config.dns_verified = True
                domain_config.dns_verified_at = DeterministicClock.utcnow()
                domain_config.status = DomainStatus.ACTIVE
                self.db.commit()

                logger.info(f"Domain verified: {domain_config.domain}")
                return True
        except Exception as e:
            logger.error(f"Error verifying domain {domain_config.domain}: {e}")

        return False


if __name__ == "__main__":
    # Example usage
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("postgresql://localhost/greenlang_partners")
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    manager = WhiteLabelManager(db)

    # Create config
    config_data = WhiteLabelCreate(
        brand_name="Acme Carbon Analytics",
        logo_url="https://example.com/logo.png",
        colors=ColorScheme(
            primary_color="#FF5733",
            secondary_color="#10B981"
        ),
        support_email="support@acme.com"
    )

    config = manager.create_config("partner_123", config_data)
    print(f"Created config: {config.id}")

    # Generate CSS
    css = manager.generate_theme_css("partner_123")
    print(css)
