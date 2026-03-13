# -*- coding: utf-8 -*-
"""
Unit tests for OperatorRegistrar engine - AGENT-EUDR-036

Tests operator registration, status updates, renewal, and EORI validation.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import timedelta, timezone, datetime

import pytest

from greenlang.agents.eudr.eu_information_system_interface.config import (
    EUInformationSystemInterfaceConfig,
)
from greenlang.agents.eudr.eu_information_system_interface.operator_registrar import (
    OperatorRegistrar,
)
from greenlang.agents.eudr.eu_information_system_interface.models import (
    CompetentAuthority,
    OperatorType,
    RegistrationStatus,
)
from greenlang.agents.eudr.eu_information_system_interface.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def registrar() -> OperatorRegistrar:
    """Create an OperatorRegistrar instance."""
    config = EUInformationSystemInterfaceConfig()
    return OperatorRegistrar(config=config, provenance=ProvenanceTracker())


class TestRegisterOperator:
    """Test OperatorRegistrar.register_operator()."""

    @pytest.mark.asyncio
    async def test_register_basic(self, registrar):
        reg = await registrar.register_operator(
            operator_id="op-001",
            eori_number="DE123456789012",
            operator_type="operator",
            company_name="Green Trading GmbH",
            member_state="DE",
        )
        assert reg.registration_id.startswith("reg-")
        assert reg.operator_id == "op-001"
        assert reg.eori_number == "DE123456789012"
        assert reg.operator_type == OperatorType.OPERATOR
        assert reg.member_state == CompetentAuthority.DE
        assert reg.registration_status == RegistrationStatus.PENDING

    @pytest.mark.asyncio
    async def test_register_trader(self, registrar):
        reg = await registrar.register_operator(
            operator_id="op-002",
            eori_number="NL987654321098",
            operator_type="trader",
            company_name="Euro Timber NV",
            member_state="NL",
        )
        assert reg.operator_type == OperatorType.TRADER
        assert reg.member_state == CompetentAuthority.NL

    @pytest.mark.asyncio
    async def test_register_sme_operator(self, registrar):
        reg = await registrar.register_operator(
            operator_id="op-003",
            eori_number="FR123456789012",
            operator_type="sme_operator",
            company_name="Petit Cafe SARL",
            member_state="FR",
        )
        assert reg.operator_type == OperatorType.SME_OPERATOR

    @pytest.mark.asyncio
    async def test_register_with_address_and_email(self, registrar):
        reg = await registrar.register_operator(
            operator_id="op-001",
            eori_number="DE123456789012",
            operator_type="operator",
            company_name="Green Trading GmbH",
            member_state="DE",
            address="Musterstr. 123, Berlin",
            contact_email="compliance@greentrading.de",
        )
        assert reg.address == "Musterstr. 123, Berlin"
        assert reg.contact_email == "compliance@greentrading.de"

    @pytest.mark.asyncio
    async def test_register_sets_expiry(self, registrar):
        reg = await registrar.register_operator(
            operator_id="op-001",
            eori_number="DE123456789012",
            operator_type="operator",
            company_name="Test GmbH",
            member_state="DE",
        )
        assert reg.expires_at is not None
        assert reg.registered_at is not None

    @pytest.mark.asyncio
    async def test_register_has_provenance(self, registrar):
        reg = await registrar.register_operator(
            operator_id="op-001",
            eori_number="DE123456789012",
            operator_type="operator",
            company_name="Test GmbH",
            member_state="DE",
        )
        assert len(reg.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_register_invalid_eori_raises(self, registrar):
        with pytest.raises(ValueError, match="Invalid EORI"):
            await registrar.register_operator(
                operator_id="op-001",
                eori_number="invalid",
                operator_type="operator",
                company_name="Test GmbH",
                member_state="DE",
            )

    @pytest.mark.asyncio
    async def test_register_invalid_member_state_raises(self, registrar):
        with pytest.raises(ValueError, match="Unknown member state"):
            await registrar.register_operator(
                operator_id="op-001",
                eori_number="DE123456789012",
                operator_type="operator",
                company_name="Test GmbH",
                member_state="XX",
            )

    @pytest.mark.asyncio
    async def test_register_invalid_operator_type_raises(self, registrar):
        with pytest.raises(ValueError, match="Invalid operator type"):
            await registrar.register_operator(
                operator_id="op-001",
                eori_number="DE123456789012",
                operator_type="invalid_type",
                company_name="Test GmbH",
                member_state="DE",
            )


class TestUpdateRegistrationStatus:
    """Test OperatorRegistrar.update_registration_status()."""

    @pytest.mark.asyncio
    async def test_pending_to_active(self, registrar, sample_registration):
        updated = await registrar.update_registration_status(
            sample_registration, "active", eu_system_id="EUIS-DE-001",
        )
        assert updated.registration_status == RegistrationStatus.ACTIVE
        assert updated.eu_system_id == "EUIS-DE-001"

    @pytest.mark.asyncio
    async def test_active_to_suspended(self, registrar, active_registration):
        updated = await registrar.update_registration_status(
            active_registration, "suspended",
        )
        assert updated.registration_status == RegistrationStatus.SUSPENDED

    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self, registrar, sample_registration):
        with pytest.raises(ValueError, match="Invalid transition"):
            await registrar.update_registration_status(
                sample_registration, "expired",
            )

    @pytest.mark.asyncio
    async def test_invalid_status_raises(self, registrar, sample_registration):
        with pytest.raises(ValueError, match="Invalid registration status"):
            await registrar.update_registration_status(
                sample_registration, "nonexistent",
            )


class TestCheckRenewalEligibility:
    """Test OperatorRegistrar.check_renewal_eligibility()."""

    @pytest.mark.asyncio
    async def test_not_due_for_renewal(self, registrar, active_registration):
        result = await registrar.check_renewal_eligibility(active_registration)
        assert result["needs_renewal"] is False

    @pytest.mark.asyncio
    async def test_due_for_renewal(self, registrar, active_registration):
        active_registration.expires_at = datetime.now(timezone.utc) + timedelta(days=10)
        result = await registrar.check_renewal_eligibility(active_registration)
        assert result["needs_renewal"] is True

    @pytest.mark.asyncio
    async def test_expired(self, registrar, active_registration):
        active_registration.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        result = await registrar.check_renewal_eligibility(active_registration)
        assert result["is_expired"] is True

    @pytest.mark.asyncio
    async def test_no_expiry_date(self, registrar, sample_registration):
        sample_registration.expires_at = None
        result = await registrar.check_renewal_eligibility(sample_registration)
        assert result["needs_renewal"] is False


class TestRenewRegistration:
    """Test OperatorRegistrar.renew_registration()."""

    @pytest.mark.asyncio
    async def test_renew_active(self, registrar, active_registration):
        old_expiry = active_registration.expires_at
        renewed = await registrar.renew_registration(active_registration)
        assert renewed.expires_at > old_expiry
        assert renewed.registration_status == RegistrationStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_renew_revoked_raises(self, registrar, sample_registration):
        sample_registration.registration_status = RegistrationStatus.REVOKED
        with pytest.raises(ValueError, match="revoked"):
            await registrar.renew_registration(sample_registration)


class TestHealthCheck:
    """Test OperatorRegistrar.health_check()."""

    @pytest.mark.asyncio
    async def test_health_check(self, registrar):
        health = await registrar.health_check()
        assert health["engine"] == "OperatorRegistrar"
        assert health["status"] == "available"
