"""
License Management System

Manages license keys, activation, validation, and usage tracking
for paid marketplace agents.
"""

import uuid
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from sqlalchemy.orm import Session
from sqlalchemy import and_

from greenlang.marketplace.models import AgentPurchase, MarketplaceAgent, AgentInstall

logger = logging.getLogger(__name__)


class LicenseType(str, Enum):
    """License types"""
    PERSONAL = "personal"  # 1 user
    TEAM = "team"  # 5 users
    ENTERPRISE = "enterprise"  # Unlimited


class LicenseStatus(str, Enum):
    """License status"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


@dataclass
class LicenseKey:
    """License key data"""
    key: str
    agent_id: str
    user_id: str
    license_type: LicenseType
    max_activations: int
    current_activations: int
    expires_at: Optional[datetime]
    status: LicenseStatus


@dataclass
class LicenseActivation:
    """License activation data"""
    activation_id: str
    license_key: str
    machine_id: str
    activated_at: datetime
    last_used_at: datetime


@dataclass
class LicenseValidation:
    """License validation result"""
    valid: bool
    license: Optional[LicenseKey]
    errors: List[str]


class LicenseGenerator:
    """
    Generate and validate license keys.

    License key format: PPPP-AAAA-UUUU-SSSS
    - PPPP: Product/Agent code
    - AAAA: User code
    - UUUU: Unique ID
    - SSSS: Signature
    """

    SECRET_KEY = b"greenlang_marketplace_secret"  # In production, use env variable

    @staticmethod
    def generate(agent_id: str, user_id: str) -> str:
        """
        Generate license key.

        Args:
            agent_id: Agent UUID
            user_id: User UUID

        Returns:
            License key string
        """
        # Create unique identifier
        unique_id = uuid.uuid4().hex[:8]

        # Combine data
        data = f"{agent_id[:8]}{user_id[:8]}{unique_id}"

        # Generate signature
        signature = hmac.new(
            LicenseGenerator.SECRET_KEY,
            data.encode(),
            hashlib.sha256
        ).hexdigest()[:8]

        # Format as XXXX-XXXX-XXXX-XXXX
        key = f"{agent_id[:4]}-{user_id[:4]}-{unique_id[:4]}-{signature[:4]}"

        return key.upper()

    @staticmethod
    def verify_signature(key: str) -> bool:
        """
        Verify license key signature.

        Args:
            key: License key

        Returns:
            True if signature is valid
        """
        try:
            parts = key.split('-')
            if len(parts) != 4:
                return False

            # Reconstruct data
            data = ''.join(parts[:3])

            # Verify signature
            expected_sig = hmac.new(
                LicenseGenerator.SECRET_KEY,
                data.encode(),
                hashlib.sha256
            ).hexdigest()[:8]

            return parts[3].lower() == expected_sig[:4].lower()

        except Exception as e:
            logger.error(f"Error verifying license signature: {e}")
            return False


class LicenseValidator:
    """
    Validate license keys.

    Checks activation limits, expiration, and status.
    """

    def __init__(self, session: Session):
        self.session = session

    def validate(
        self,
        license_key: str,
        machine_id: Optional[str] = None
    ) -> LicenseValidation:
        """
        Validate license key.

        Args:
            license_key: License key to validate
            machine_id: Optional machine ID for activation check

        Returns:
            Validation result
        """
        errors = []

        # Verify signature
        if not LicenseGenerator.verify_signature(license_key):
            errors.append("Invalid license key signature")
            return LicenseValidation(valid=False, license=None, errors=errors)

        # Find purchase with this license key
        purchase = self.session.query(AgentPurchase).filter(
            AgentPurchase.license_key == license_key
        ).first()

        if not purchase:
            errors.append("License key not found")
            return LicenseValidation(valid=False, license=None, errors=errors)

        # Check status
        if purchase.status == "refunded":
            errors.append("License has been refunded")
            return LicenseValidation(valid=False, license=None, errors=errors)

        # Check expiration (for subscriptions)
        if purchase.subscription_period_end:
            if datetime.utcnow() > purchase.subscription_period_end:
                # Check grace period (7 days)
                grace_end = purchase.subscription_period_end + timedelta(days=7)
                if datetime.utcnow() > grace_end:
                    errors.append("License expired")
                    return LicenseValidation(valid=False, license=None, errors=errors)
                else:
                    errors.append("License in grace period - please renew")

        # Check activation limits
        if machine_id:
            active_installs = self.session.query(AgentInstall).filter(
                and_(
                    AgentInstall.agent_id == purchase.agent_id,
                    AgentInstall.user_id == purchase.user_id,
                    AgentInstall.active == True
                )
            ).count()

            max_activations = self._get_max_activations(purchase)

            if active_installs >= max_activations:
                if not self._is_already_activated(purchase, machine_id):
                    errors.append(
                        f"Maximum activations reached ({max_activations})"
                    )
                    return LicenseValidation(valid=False, license=None, errors=errors)

        # Build license data
        license_data = LicenseKey(
            key=license_key,
            agent_id=str(purchase.agent_id),
            user_id=str(purchase.user_id),
            license_type=LicenseType.PERSONAL,  # Could be stored in purchase metadata
            max_activations=self._get_max_activations(purchase),
            current_activations=self._get_current_activations(purchase),
            expires_at=purchase.subscription_period_end,
            status=self._get_license_status(purchase)
        )

        return LicenseValidation(
            valid=True,
            license=license_data,
            errors=errors  # May contain warnings
        )

    def _get_max_activations(self, purchase: AgentPurchase) -> int:
        """Get maximum allowed activations"""
        # Default: 1 for personal, could be configured in metadata
        return 1

    def _get_current_activations(self, purchase: AgentPurchase) -> int:
        """Get current number of activations"""
        return self.session.query(AgentInstall).filter(
            and_(
                AgentInstall.agent_id == purchase.agent_id,
                AgentInstall.user_id == purchase.user_id,
                AgentInstall.active == True
            )
        ).count()

    def _is_already_activated(
        self,
        purchase: AgentPurchase,
        machine_id: str
    ) -> bool:
        """Check if this machine is already activated"""
        install = self.session.query(AgentInstall).filter(
            and_(
                AgentInstall.agent_id == purchase.agent_id,
                AgentInstall.user_id == purchase.user_id,
                AgentInstall.installation_id == machine_id,
                AgentInstall.active == True
            )
        ).first()

        return install is not None

    def _get_license_status(self, purchase: AgentPurchase) -> LicenseStatus:
        """Determine license status"""
        if purchase.status == "refunded":
            return LicenseStatus.REVOKED

        if purchase.subscription_period_end:
            if datetime.utcnow() > purchase.subscription_period_end:
                return LicenseStatus.EXPIRED

        return LicenseStatus.ACTIVE


class LicenseManager:
    """
    Main license manager.

    Handles license activation, deactivation, and tracking.
    """

    def __init__(self, session: Session):
        self.session = session
        self.validator = LicenseValidator(session)

    def activate_license(
        self,
        license_key: str,
        machine_id: str,
        agent_id: str
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Activate license on a machine.

        Args:
            license_key: License key
            machine_id: Unique machine identifier
            agent_id: Agent UUID

        Returns:
            Tuple of (success, activation_id, errors)
        """
        errors = []

        # Validate license
        validation = self.validator.validate(license_key, machine_id)

        if not validation.valid:
            return False, None, validation.errors

        try:
            # Get purchase
            purchase = self.session.query(AgentPurchase).filter(
                AgentPurchase.license_key == license_key
            ).first()

            if not purchase:
                return False, None, ["License key not found"]

            # Check if already activated on this machine
            existing_install = self.session.query(AgentInstall).filter(
                and_(
                    AgentInstall.agent_id == agent_id,
                    AgentInstall.user_id == purchase.user_id,
                    AgentInstall.installation_id == machine_id
                )
            ).first()

            if existing_install:
                # Reactivate if inactive
                if not existing_install.active:
                    existing_install.active = True
                    self.session.commit()

                return True, str(existing_install.id), []

            # Create new installation record
            install = AgentInstall(
                user_id=purchase.user_id,
                agent_id=agent_id,
                version="latest",  # Should be specified
                installation_id=machine_id,
                active=True
            )

            self.session.add(install)
            self.session.commit()

            logger.info(
                f"Activated license {license_key} on machine {machine_id}"
            )

            return True, str(install.id), []

        except Exception as e:
            logger.error(f"Error activating license: {e}", exc_info=True)
            self.session.rollback()
            errors.append(f"Activation failed: {str(e)}")
            return False, None, errors

    def deactivate_license(
        self,
        license_key: str,
        machine_id: str
    ) -> Tuple[bool, List[str]]:
        """
        Deactivate license on a machine.

        Args:
            license_key: License key
            machine_id: Machine identifier

        Returns:
            Tuple of (success, errors)
        """
        errors = []

        try:
            # Get purchase
            purchase = self.session.query(AgentPurchase).filter(
                AgentPurchase.license_key == license_key
            ).first()

            if not purchase:
                return False, ["License key not found"]

            # Find installation
            install = self.session.query(AgentInstall).filter(
                and_(
                    AgentInstall.user_id == purchase.user_id,
                    AgentInstall.installation_id == machine_id,
                    AgentInstall.active == True
                )
            ).first()

            if not install:
                return False, ["No active installation found"]

            # Deactivate
            install.active = False
            install.uninstalled_at = datetime.utcnow()

            self.session.commit()

            logger.info(
                f"Deactivated license {license_key} on machine {machine_id}"
            )

            return True, []

        except Exception as e:
            logger.error(f"Error deactivating license: {e}", exc_info=True)
            self.session.rollback()
            errors.append(f"Deactivation failed: {str(e)}")
            return False, errors

    def get_activations(self, license_key: str) -> List[Dict[str, Any]]:
        """
        Get all activations for a license.

        Args:
            license_key: License key

        Returns:
            List of activation records
        """
        purchase = self.session.query(AgentPurchase).filter(
            AgentPurchase.license_key == license_key
        ).first()

        if not purchase:
            return []

        installs = self.session.query(AgentInstall).filter(
            and_(
                AgentInstall.user_id == purchase.user_id,
                AgentInstall.agent_id == purchase.agent_id
            )
        ).all()

        return [
            {
                "activation_id": str(install.id),
                "machine_id": install.installation_id,
                "active": install.active,
                "activated_at": install.installed_at.isoformat(),
                "last_used_at": install.last_used_at.isoformat() if install.last_used_at else None
            }
            for install in installs
        ]

    def revoke_license(
        self,
        license_key: str,
        reason: str
    ) -> Tuple[bool, List[str]]:
        """
        Revoke a license (admin action).

        Args:
            license_key: License key to revoke
            reason: Revocation reason

        Returns:
            Tuple of (success, errors)
        """
        errors = []

        try:
            purchase = self.session.query(AgentPurchase).filter(
                AgentPurchase.license_key == license_key
            ).first()

            if not purchase:
                return False, ["License key not found"]

            # Update purchase status
            purchase.status = "refunded"  # Or create a 'revoked' status
            purchase.refund_reason = f"License revoked: {reason}"

            # Deactivate all installations
            installs = self.session.query(AgentInstall).filter(
                and_(
                    AgentInstall.user_id == purchase.user_id,
                    AgentInstall.agent_id == purchase.agent_id,
                    AgentInstall.active == True
                )
            ).all()

            for install in installs:
                install.active = False
                install.uninstalled_at = datetime.utcnow()

            self.session.commit()

            logger.warning(f"Revoked license {license_key}: {reason}")

            return True, []

        except Exception as e:
            logger.error(f"Error revoking license: {e}", exc_info=True)
            self.session.rollback()
            errors.append(f"Revocation failed: {str(e)}")
            return False, errors
