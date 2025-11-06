"""
Jurisdiction-specific rules for GDPR, CCPA, and CAN-SPAM compliance.

Implements consent requirements for different regulatory frameworks.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

from ..config import EU_COUNTRIES, CCPA_JURISDICTIONS, get_jurisdiction_config
from ..models import ConsentStatus, LawfulBasis
from ..exceptions import JurisdictionNotSupportedError


class JurisdictionType(str, Enum):
    """Supported jurisdictions."""
    GDPR = "GDPR"
    CCPA = "CCPA"
    CAN_SPAM = "CAN_SPAM"


class JurisdictionRules:
    """
    Base class for jurisdiction-specific consent rules.

    Each jurisdiction implements different requirements for lawful
    data processing and communications.
    """

    def __init__(self, jurisdiction_type: JurisdictionType):
        self.jurisdiction_type = jurisdiction_type
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load jurisdiction-specific configuration."""
        raise NotImplementedError

    def can_contact(
        self,
        consent_status: ConsentStatus,
        lawful_basis: LawfulBasis,
        opt_out_date: Optional[datetime] = None
    ) -> bool:
        """
        Check if supplier can be contacted under this jurisdiction.

        Args:
            consent_status: Current consent status
            lawful_basis: GDPR lawful basis
            opt_out_date: Date of opt-out (if applicable)

        Returns:
            True if contact is allowed, False otherwise
        """
        raise NotImplementedError

    def requires_opt_in(self) -> bool:
        """Check if jurisdiction requires opt-in (vs opt-out)."""
        raise NotImplementedError

    def opt_out_grace_period(self) -> timedelta:
        """Get grace period for honoring opt-out requests."""
        raise NotImplementedError

    def validate_lawful_basis(self, lawful_basis: LawfulBasis) -> bool:
        """Validate if lawful basis is acceptable for this jurisdiction."""
        raise NotImplementedError


class GDPRRules(JurisdictionRules):
    """
    GDPR (General Data Protection Regulation) compliance rules.

    Key requirements:
    - Explicit consent required for marketing communications
    - Lawful basis: consent, contract, legitimate interest
    - Right to erasure (Article 17)
    - Data portability (Article 20)
    - DPA required for data processors
    """

    def __init__(self):
        super().__init__(JurisdictionType.GDPR)

    def _load_config(self) -> Dict[str, Any]:
        """Load GDPR configuration."""
        return get_jurisdiction_config("DE")  # Use Germany as GDPR reference

    def can_contact(
        self,
        consent_status: ConsentStatus,
        lawful_basis: LawfulBasis,
        opt_out_date: Optional[datetime] = None
    ) -> bool:
        """
        Check if contact is allowed under GDPR.

        GDPR requires explicit consent for marketing, but allows
        legitimate interest for contractual communications.
        """
        # Always honor opt-out
        if consent_status == ConsentStatus.OPTED_OUT:
            return False

        # Check if opt-out is within grace period
        if opt_out_date:
            grace_period = self.opt_out_grace_period()
            if datetime.utcnow() < opt_out_date + grace_period:
                return False

        # For marketing: requires explicit consent
        if lawful_basis == LawfulBasis.CONSENT:
            return consent_status == ConsentStatus.OPTED_IN

        # For contractual: legitimate interest is acceptable
        if lawful_basis in [LawfulBasis.CONTRACT, LawfulBasis.LEGITIMATE_INTEREST]:
            return consent_status != ConsentStatus.OPTED_OUT

        return False

    def requires_opt_in(self) -> bool:
        """GDPR requires opt-in for marketing communications."""
        return True

    def opt_out_grace_period(self) -> timedelta:
        """GDPR requires immediate opt-out honor."""
        return timedelta(days=1)

    def validate_lawful_basis(self, lawful_basis: LawfulBasis) -> bool:
        """Validate lawful basis under GDPR."""
        allowed = [
            LawfulBasis.CONSENT,
            LawfulBasis.CONTRACT,
            LawfulBasis.LEGITIMATE_INTEREST
        ]
        return lawful_basis in allowed

    def requires_dpa(self) -> bool:
        """GDPR requires Data Processing Agreement."""
        return True


class CCPARules(JurisdictionRules):
    """
    CCPA (California Consumer Privacy Act) compliance rules.

    Key requirements:
    - Opt-out model (vs opt-in)
    - Right to know what data is collected
    - Right to delete personal information
    - Right to opt-out of data sale
    - Must honor opt-out within 15 days
    """

    def __init__(self):
        super().__init__(JurisdictionType.CCPA)

    def _load_config(self) -> Dict[str, Any]:
        """Load CCPA configuration."""
        return get_jurisdiction_config("US-CA")

    def can_contact(
        self,
        consent_status: ConsentStatus,
        lawful_basis: LawfulBasis,
        opt_out_date: Optional[datetime] = None
    ) -> bool:
        """
        Check if contact is allowed under CCPA.

        CCPA uses opt-out model - can contact unless opted out.
        """
        # Always honor opt-out
        if consent_status == ConsentStatus.OPTED_OUT:
            return False

        # Check if opt-out is within grace period
        if opt_out_date:
            grace_period = self.opt_out_grace_period()
            if datetime.utcnow() < opt_out_date + grace_period:
                return False

        # Opt-out model: can contact unless opted out
        return consent_status != ConsentStatus.OPTED_OUT

    def requires_opt_in(self) -> bool:
        """CCPA uses opt-out model."""
        return False

    def opt_out_grace_period(self) -> timedelta:
        """CCPA requires opt-out honor within 15 days."""
        return timedelta(days=15)

    def validate_lawful_basis(self, lawful_basis: LawfulBasis) -> bool:
        """CCPA doesn't restrict lawful basis like GDPR."""
        return True

    def requires_privacy_notice(self) -> bool:
        """CCPA requires privacy notice at collection."""
        return True

    def requires_do_not_sell_link(self) -> bool:
        """CCPA requires 'Do Not Sell My Personal Information' link."""
        return True


class CANSPAMRules(JurisdictionRules):
    """
    CAN-SPAM Act compliance rules (US federal law).

    Key requirements:
    - Opt-out model
    - Truthful subject lines and headers
    - Physical postal address required
    - Unsubscribe link required in every email
    - Must honor opt-out within 10 business days
    """

    def __init__(self):
        super().__init__(JurisdictionType.CAN_SPAM)

    def _load_config(self) -> Dict[str, Any]:
        """Load CAN-SPAM configuration."""
        return get_jurisdiction_config("US")

    def can_contact(
        self,
        consent_status: ConsentStatus,
        lawful_basis: LawfulBasis,
        opt_out_date: Optional[datetime] = None
    ) -> bool:
        """
        Check if contact is allowed under CAN-SPAM.

        CAN-SPAM uses opt-out model - can contact unless opted out.
        """
        # Always honor opt-out
        if consent_status == ConsentStatus.OPTED_OUT:
            return False

        # Check if opt-out is within grace period (10 business days)
        if opt_out_date:
            grace_period = self.opt_out_grace_period()
            if datetime.utcnow() < opt_out_date + grace_period:
                return False

        # Opt-out model: can contact unless opted out
        return consent_status != ConsentStatus.OPTED_OUT

    def requires_opt_in(self) -> bool:
        """CAN-SPAM uses opt-out model."""
        return False

    def opt_out_grace_period(self) -> timedelta:
        """CAN-SPAM requires opt-out honor within 10 business days."""
        return timedelta(days=10)

    def validate_lawful_basis(self, lawful_basis: LawfulBasis) -> bool:
        """CAN-SPAM doesn't have lawful basis requirements like GDPR."""
        return True

    def requires_physical_address(self) -> bool:
        """CAN-SPAM requires physical postal address in emails."""
        return True

    def requires_unsubscribe_link(self) -> bool:
        """CAN-SPAM requires unsubscribe link in every email."""
        return True

    def requires_truthful_subject(self) -> bool:
        """CAN-SPAM requires truthful subject lines."""
        return True


class JurisdictionManager:
    """
    Manages jurisdiction-specific rules and determines applicable regulations.
    """

    def __init__(self):
        self.rules_map = {
            JurisdictionType.GDPR: GDPRRules(),
            JurisdictionType.CCPA: CCPARules(),
            JurisdictionType.CAN_SPAM: CANSPAMRules(),
        }

    def get_jurisdiction(self, country_code: str) -> JurisdictionType:
        """
        Determine applicable jurisdiction based on country code.

        Args:
            country_code: ISO 3166-1 country code

        Returns:
            Applicable jurisdiction type
        """
        country_upper = country_code.upper()

        if country_upper in EU_COUNTRIES:
            return JurisdictionType.GDPR
        elif country_upper in CCPA_JURISDICTIONS:
            return JurisdictionType.CCPA
        else:
            # Default to CAN-SPAM for other countries
            return JurisdictionType.CAN_SPAM

    def get_rules(self, country_code: str) -> JurisdictionRules:
        """
        Get jurisdiction rules for country.

        Args:
            country_code: ISO 3166-1 country code

        Returns:
            Jurisdiction-specific rules

        Raises:
            JurisdictionNotSupportedError: If jurisdiction not supported
        """
        jurisdiction = self.get_jurisdiction(country_code)
        rules = self.rules_map.get(jurisdiction)

        if not rules:
            raise JurisdictionNotSupportedError(country_code)

        return rules

    def can_contact(
        self,
        country_code: str,
        consent_status: ConsentStatus,
        lawful_basis: LawfulBasis,
        opt_out_date: Optional[datetime] = None
    ) -> bool:
        """
        Check if supplier can be contacted based on jurisdiction.

        Args:
            country_code: Supplier country
            consent_status: Current consent status
            lawful_basis: GDPR lawful basis
            opt_out_date: Date of opt-out (if applicable)

        Returns:
            True if contact is allowed, False otherwise
        """
        rules = self.get_rules(country_code)
        return rules.can_contact(consent_status, lawful_basis, opt_out_date)

    def get_required_email_elements(self, country_code: str) -> Dict[str, bool]:
        """
        Get required email elements for jurisdiction.

        Args:
            country_code: Supplier country

        Returns:
            Dictionary of required elements
        """
        rules = self.get_rules(country_code)
        elements = {
            "unsubscribe_link": True,  # Required by all
            "privacy_policy_link": False,
            "physical_address": False,
            "dpa_link": False,
            "do_not_sell_link": False,
        }

        if isinstance(rules, GDPRRules):
            elements["privacy_policy_link"] = True
            elements["dpa_link"] = rules.requires_dpa()

        elif isinstance(rules, CCPARules):
            elements["privacy_policy_link"] = True
            elements["do_not_sell_link"] = rules.requires_do_not_sell_link()

        elif isinstance(rules, CANSPAMRules):
            elements["physical_address"] = rules.requires_physical_address()

        return elements
