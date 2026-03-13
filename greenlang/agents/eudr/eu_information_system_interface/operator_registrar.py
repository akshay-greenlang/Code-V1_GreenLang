# -*- coding: utf-8 -*-
"""
Operator Registrar Engine - AGENT-EUDR-036: EU Information System Interface

Engine 2: Manages operator registration and lifecycle in the EU Information
System per EUDR Articles 4-6, including EORI validation, registration
submission, status management, renewal processing, and competent authority
mapping.

Responsibilities:
    - Register new operators/traders in the EU Information System
    - Validate EORI numbers against format requirements
    - Manage operator registration lifecycle (pending -> active -> expired)
    - Track registration expiry and trigger renewal notices
    - Map operators to Member State competent authorities
    - Maintain operator profile data for DDS submissions

Zero-Hallucination Guarantees:
    - EORI validation uses deterministic regex patterns
    - Expiry calculations use standard date arithmetic
    - No LLM involvement in registration logic
    - Complete provenance trail for every registration

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-036 (GL-EUDR-EUIS-036)
Regulation: EU 2023/1115 (EUDR) Articles 4, 5, 6, 33
Status: Production Ready
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .config import EUInformationSystemInterfaceConfig, get_config
from .models import (
    CompetentAuthority,
    OperatorRegistration,
    OperatorType,
    RegistrationStatus,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


class OperatorRegistrar:
    """Manages operator registration in the EU Information System.

    Handles the complete operator registration lifecycle including
    validation, submission, status management, and renewal processing
    per EUDR Articles 4-6.

    Attributes:
        _config: Agent configuration instance.
        _provenance: Provenance tracker for audit trail.

    Example:
        >>> registrar = OperatorRegistrar()
        >>> reg = await registrar.register_operator(
        ...     operator_id="OP001",
        ...     eori_number="DE123456789012",
        ...     operator_type="operator",
        ...     company_name="Example GmbH",
        ...     member_state="DE",
        ... )
        >>> assert reg.registration_status == RegistrationStatus.PENDING
    """

    def __init__(
        self,
        config: Optional[EUInformationSystemInterfaceConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize OperatorRegistrar.

        Args:
            config: Agent configuration. Uses get_config() if None.
            provenance: Provenance tracker instance.
        """
        self._config = config or get_config()
        self._provenance = provenance or ProvenanceTracker()
        logger.info(
            "OperatorRegistrar initialized: eori_pattern=%s, "
            "expiry_days=%d, renewal_notice=%d",
            self._config.eori_format_pattern,
            self._config.registration_expiry_days,
            self._config.registration_renewal_notice_days,
        )

    async def register_operator(
        self,
        operator_id: str,
        eori_number: str,
        operator_type: str,
        company_name: str,
        member_state: str,
        address: str = "",
        contact_email: str = "",
    ) -> OperatorRegistration:
        """Register a new operator in the EU Information System.

        Creates a registration record with PENDING status. The actual
        registration with the EU IS is performed via the API client.

        Args:
            operator_id: GreenLang operator identifier.
            eori_number: EORI number (validated against format).
            operator_type: Operator classification (operator/trader/sme_*).
            company_name: Legal entity name.
            member_state: EU Member State code (2-letter).
            address: Registered address (optional).
            contact_email: Contact email (optional).

        Returns:
            OperatorRegistration in PENDING status.

        Raises:
            ValueError: If EORI format is invalid or member state unknown.
        """
        start = time.monotonic()
        registration_id = f"reg-{uuid.uuid4().hex[:12]}"

        logger.info(
            "Registering operator: id=%s, eori=%s, type=%s, state=%s",
            operator_id, eori_number, operator_type, member_state,
        )

        # Validate EORI format
        if not self._validate_eori(eori_number):
            raise ValueError(
                f"Invalid EORI format: '{eori_number}'. "
                f"Expected pattern: {self._config.eori_format_pattern}"
            )

        # Validate member state
        try:
            authority = CompetentAuthority(member_state)
        except ValueError as e:
            raise ValueError(
                f"Unknown member state: '{member_state}'. "
                f"Must be a valid EU Member State code."
            ) from e

        # Validate operator type
        try:
            op_type = OperatorType(operator_type)
        except ValueError as e:
            raise ValueError(
                f"Invalid operator type: '{operator_type}'. "
                f"Must be one of: {[t.value for t in OperatorType]}"
            ) from e

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=self._config.registration_expiry_days)

        registration = OperatorRegistration(
            registration_id=registration_id,
            operator_id=operator_id,
            eori_number=eori_number,
            operator_type=op_type,
            company_name=company_name,
            member_state=authority,
            address=address,
            contact_email=contact_email,
            registration_status=RegistrationStatus.PENDING,
            registered_at=now,
            expires_at=expires_at,
        )

        # Compute provenance hash
        provenance_data = {
            "registration_id": registration_id,
            "operator_id": operator_id,
            "eori_number": eori_number,
            "member_state": member_state,
            "registered_at": now.isoformat(),
        }
        registration.provenance_hash = self._provenance.compute_hash(
            provenance_data
        )

        # Record provenance entry
        self._provenance.create_entry(
            step="register_operator",
            source=f"operator:{operator_id}",
            input_hash=self._provenance.compute_hash(
                {"operator_id": operator_id}
            ),
            output_hash=registration.provenance_hash,
        )

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Registration %s created for operator %s in %.1fms "
            "(expires=%s)",
            registration_id, operator_id, elapsed_ms,
            expires_at.isoformat(),
        )

        return registration

    async def update_registration_status(
        self,
        registration: OperatorRegistration,
        new_status: str,
        eu_system_id: Optional[str] = None,
    ) -> OperatorRegistration:
        """Update operator registration status.

        Args:
            registration: Registration to update.
            new_status: New status value.
            eu_system_id: Optional EU IS system ID (set on acceptance).

        Returns:
            Updated OperatorRegistration.

        Raises:
            ValueError: If status transition is invalid.
        """
        try:
            status = RegistrationStatus(new_status)
        except ValueError as e:
            raise ValueError(
                f"Invalid registration status: '{new_status}'"
            ) from e

        # Validate state transition
        valid_transitions = {
            RegistrationStatus.PENDING: {
                RegistrationStatus.ACTIVE,
                RegistrationStatus.REVOKED,
            },
            RegistrationStatus.ACTIVE: {
                RegistrationStatus.SUSPENDED,
                RegistrationStatus.EXPIRED,
                RegistrationStatus.REVOKED,
            },
            RegistrationStatus.SUSPENDED: {
                RegistrationStatus.ACTIVE,
                RegistrationStatus.REVOKED,
            },
            RegistrationStatus.EXPIRED: {
                RegistrationStatus.ACTIVE,
            },
        }

        allowed = valid_transitions.get(
            registration.registration_status, set()
        )
        if status not in allowed:
            raise ValueError(
                f"Invalid transition from "
                f"'{registration.registration_status.value}' to "
                f"'{status.value}'. Allowed: "
                f"{[s.value for s in allowed]}"
            )

        old_status = registration.registration_status.value
        registration.registration_status = status

        if eu_system_id:
            registration.eu_system_id = eu_system_id

        # Update provenance
        registration.provenance_hash = self._provenance.compute_hash({
            "registration_id": registration.registration_id,
            "old_status": old_status,
            "new_status": status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(
            "Registration %s status changed: %s -> %s",
            registration.registration_id, old_status, status.value,
        )

        return registration

    async def check_renewal_eligibility(
        self,
        registration: OperatorRegistration,
    ) -> Dict[str, Any]:
        """Check if a registration is due for renewal.

        Evaluates registration expiry against the configured
        renewal notice period.

        Args:
            registration: Registration to check.

        Returns:
            Dictionary with renewal eligibility details.
        """
        now = datetime.now(timezone.utc)
        expires_at = registration.expires_at

        if expires_at is None:
            return {
                "registration_id": registration.registration_id,
                "needs_renewal": False,
                "reason": "No expiry date set",
            }

        # Make expires_at timezone-aware if it is not
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        days_until_expiry = (expires_at - now).days
        notice_threshold = self._config.registration_renewal_notice_days

        needs_renewal = days_until_expiry <= notice_threshold
        is_expired = days_until_expiry <= 0

        result = {
            "registration_id": registration.registration_id,
            "operator_id": registration.operator_id,
            "current_status": registration.registration_status.value,
            "expires_at": expires_at.isoformat(),
            "days_until_expiry": days_until_expiry,
            "needs_renewal": needs_renewal,
            "is_expired": is_expired,
            "renewal_notice_days": notice_threshold,
            "checked_at": now.isoformat(),
        }

        if needs_renewal:
            logger.warning(
                "Registration %s needs renewal: %d days until expiry",
                registration.registration_id, days_until_expiry,
            )

        return result

    async def renew_registration(
        self,
        registration: OperatorRegistration,
    ) -> OperatorRegistration:
        """Renew an operator registration.

        Extends the expiry date by the configured expiry period.

        Args:
            registration: Registration to renew.

        Returns:
            Updated OperatorRegistration with new expiry date.

        Raises:
            ValueError: If registration is revoked and cannot be renewed.
        """
        if registration.registration_status == RegistrationStatus.REVOKED:
            raise ValueError(
                f"Registration {registration.registration_id} is revoked "
                f"and cannot be renewed."
            )

        now = datetime.now(timezone.utc)
        new_expiry = now + timedelta(
            days=self._config.registration_expiry_days
        )

        old_expiry = registration.expires_at
        registration.expires_at = new_expiry
        registration.registration_status = RegistrationStatus.ACTIVE

        # Update provenance
        registration.provenance_hash = self._provenance.compute_hash({
            "registration_id": registration.registration_id,
            "action": "renew",
            "old_expiry": str(old_expiry),
            "new_expiry": new_expiry.isoformat(),
            "renewed_at": now.isoformat(),
        })

        logger.info(
            "Registration %s renewed: new_expiry=%s",
            registration.registration_id, new_expiry.isoformat(),
        )

        return registration

    def _validate_eori(self, eori: str) -> bool:
        """Validate EORI number against configured pattern.

        Args:
            eori: EORI number string.

        Returns:
            True if EORI matches the expected format.
        """
        pattern = self._config.eori_format_pattern
        return bool(re.match(pattern, eori))

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status.

        Returns:
            Dictionary with engine status and configuration details.
        """
        return {
            "engine": "OperatorRegistrar",
            "status": "available",
            "config": {
                "eori_pattern": self._config.eori_format_pattern,
                "expiry_days": self._config.registration_expiry_days,
                "renewal_notice_days": self._config.registration_renewal_notice_days,
                "max_operators": self._config.max_operators_per_account,
            },
        }
