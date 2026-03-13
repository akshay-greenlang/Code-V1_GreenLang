# -*- coding: utf-8 -*-
"""
Unit tests for ValueCalculator engine - AGENT-EUDR-039

Tests CIF/FOB value calculations, currency conversion (EUR/USD/GBP/JPY),
tariff rate application, duty computation, VAT calculation,
exchange rate handling, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.customs_declaration_support.config import CustomsDeclarationConfig
from greenlang.agents.eudr.customs_declaration_support.value_calculator import ValueCalculator
from greenlang.agents.eudr.customs_declaration_support.models import (
    CurrencyCode, DutyCalculation, IncotermsType, TariffCalculation,
    TariffLineItem,
)


@pytest.fixture
def config():
    return CustomsDeclarationConfig()


@pytest.fixture
def calculator(config):
    return ValueCalculator(config=config)


# ====================================================================
# CIF/FOB Calculation Tests
# ====================================================================


class TestCIFCalculation:
    @pytest.mark.asyncio
    async def test_cif_includes_insurance_and_freight(self, calculator):
        result = await calculator.calculate_customs_value(
            fob_value=Decimal("10000.00"),
            freight_cost=Decimal("500.00"),
            insurance_cost=Decimal("100.00"),
            incoterms="CIF",
        )
        assert result == Decimal("10600.00")

    @pytest.mark.asyncio
    async def test_fob_excludes_insurance_and_freight(self, calculator):
        result = await calculator.calculate_customs_value(
            fob_value=Decimal("10000.00"),
            freight_cost=Decimal("500.00"),
            insurance_cost=Decimal("100.00"),
            incoterms="FOB",
        )
        assert result == Decimal("10000.00")

    @pytest.mark.asyncio
    async def test_cif_zero_extras(self, calculator):
        result = await calculator.calculate_customs_value(
            fob_value=Decimal("25000.00"),
            freight_cost=Decimal("0.00"),
            insurance_cost=Decimal("0.00"),
            incoterms="CIF",
        )
        assert result == Decimal("25000.00")

    @pytest.mark.asyncio
    async def test_exw_calculation(self, calculator):
        result = await calculator.calculate_customs_value(
            fob_value=Decimal("8000.00"),
            freight_cost=Decimal("1000.00"),
            insurance_cost=Decimal("200.00"),
            incoterms="EXW",
        )
        # EXW typically needs all costs added for customs value
        assert result >= Decimal("8000.00")


# ====================================================================
# Currency Conversion Tests
# ====================================================================


class TestCurrencyConversion:
    @pytest.mark.asyncio
    async def test_eur_to_eur_no_conversion(self, calculator):
        result = await calculator.convert_currency(
            amount=Decimal("1000.00"),
            from_currency="EUR",
            to_currency="EUR",
        )
        assert result == Decimal("1000.00")

    @pytest.mark.asyncio
    async def test_usd_to_eur_conversion(self, calculator):
        result = await calculator.convert_currency(
            amount=Decimal("1000.00"),
            from_currency="USD",
            to_currency="EUR",
        )
        assert isinstance(result, Decimal)
        assert result > Decimal("0")

    @pytest.mark.asyncio
    async def test_gbp_to_eur_conversion(self, calculator):
        result = await calculator.convert_currency(
            amount=Decimal("1000.00"),
            from_currency="GBP",
            to_currency="EUR",
        )
        assert isinstance(result, Decimal)
        assert result > Decimal("0")

    @pytest.mark.asyncio
    async def test_jpy_to_eur_conversion(self, calculator):
        result = await calculator.convert_currency(
            amount=Decimal("100000.00"),
            from_currency="JPY",
            to_currency="EUR",
        )
        assert isinstance(result, Decimal)
        assert result > Decimal("0")

    @pytest.mark.asyncio
    async def test_zero_amount_conversion(self, calculator):
        result = await calculator.convert_currency(
            amount=Decimal("0.00"),
            from_currency="USD",
            to_currency="EUR",
        )
        assert result == Decimal("0.00") or result == Decimal("0")

    @pytest.mark.asyncio
    async def test_negative_amount_raises(self, calculator):
        with pytest.raises(ValueError, match="negative"):
            await calculator.convert_currency(
                amount=Decimal("-100.00"),
                from_currency="USD",
                to_currency="EUR",
            )

    @pytest.mark.asyncio
    async def test_unsupported_currency_raises(self, calculator):
        with pytest.raises(ValueError, match="currency"):
            await calculator.convert_currency(
                amount=Decimal("1000.00"),
                from_currency="XYZ",
                to_currency="EUR",
            )

    @pytest.mark.asyncio
    async def test_conversion_precision(self, calculator):
        result = await calculator.convert_currency(
            amount=Decimal("1000.00"),
            from_currency="USD",
            to_currency="EUR",
        )
        # Result should have reasonable precision
        assert result == result.quantize(Decimal("0.01")) or \
               result == result.quantize(Decimal("0.0001"))


class TestGetExchangeRate:
    @pytest.mark.asyncio
    async def test_eur_to_eur_rate_is_one(self, calculator):
        rate = await calculator.get_exchange_rate("EUR", "EUR")
        assert rate == Decimal("1.0") or rate == Decimal("1")

    @pytest.mark.asyncio
    async def test_rate_is_positive(self, calculator):
        rate = await calculator.get_exchange_rate("USD", "EUR")
        assert rate > Decimal("0")

    @pytest.mark.asyncio
    async def test_rate_is_decimal(self, calculator):
        rate = await calculator.get_exchange_rate("GBP", "EUR")
        assert isinstance(rate, Decimal)


# ====================================================================
# Tariff Calculation Tests
# ====================================================================


class TestTariffCalculation:
    @pytest.mark.asyncio
    async def test_calculate_tariff_basic(self, calculator):
        result = await calculator.calculate_tariff(
            declaration_id="DECL-001",
            cn_code="18010000",
            customs_value=Decimal("25000.00"),
            quantity=Decimal("10000.00"),
            origin_country="CI",
        )
        assert isinstance(result, TariffCalculation)
        assert result.declaration_id == "DECL-001"

    @pytest.mark.asyncio
    async def test_zero_duty_for_raw_cocoa(self, calculator):
        result = await calculator.calculate_tariff(
            declaration_id="DECL-001",
            cn_code="18010000",
            customs_value=Decimal("25000.00"),
            quantity=Decimal("10000.00"),
            origin_country="CI",
        )
        assert result.total_duty_amount == Decimal("0.00") or result.total_duty_amount == Decimal("0")

    @pytest.mark.asyncio
    async def test_vat_applied(self, calculator):
        result = await calculator.calculate_tariff(
            declaration_id="DECL-001",
            cn_code="18010000",
            customs_value=Decimal("25000.00"),
            quantity=Decimal("10000.00"),
            origin_country="CI",
        )
        assert result.total_vat_amount >= Decimal("0")

    @pytest.mark.asyncio
    async def test_total_payable_non_negative(self, calculator):
        result = await calculator.calculate_tariff(
            declaration_id="DECL-001",
            cn_code="18010000",
            customs_value=Decimal("25000.00"),
            quantity=Decimal("10000.00"),
            origin_country="CI",
        )
        assert result.total_payable >= Decimal("0")

    @pytest.mark.asyncio
    async def test_currency_default_eur(self, calculator):
        result = await calculator.calculate_tariff(
            declaration_id="DECL-001",
            cn_code="18010000",
            customs_value=Decimal("25000.00"),
            quantity=Decimal("10000.00"),
            origin_country="CI",
        )
        assert result.currency == CurrencyCode.EUR

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, calculator):
        result = await calculator.calculate_tariff(
            declaration_id="DECL-001",
            cn_code="18010000",
            customs_value=Decimal("25000.00"),
            quantity=Decimal("10000.00"),
            origin_country="CI",
        )
        assert len(result.provenance_hash) == 64


class TestDutyCalculation:
    @pytest.mark.asyncio
    async def test_calculate_duty_basic(self, calculator):
        result = await calculator.calculate_duty(
            cn_code="18032000",
            customs_value=Decimal("50000.00"),
            origin_country="CI",
        )
        assert isinstance(result, DutyCalculation)
        assert result.cn_code == "18032000"

    @pytest.mark.asyncio
    async def test_duty_amount_matches_rate(self, calculator):
        result = await calculator.calculate_duty(
            cn_code="18032000",
            customs_value=Decimal("50000.00"),
            origin_country="CI",
        )
        expected = result.customs_value * result.duty_rate / 100
        assert result.duty_amount == expected or abs(result.duty_amount - expected) < Decimal("0.01")

    @pytest.mark.asyncio
    async def test_preferential_tariff(self, calculator):
        result = await calculator.calculate_duty(
            cn_code="18010000",
            customs_value=Decimal("25000.00"),
            origin_country="CI",
            preferential_origin="CI",
        )
        # If preferential rate applies, it should be <= standard rate
        if result.preferential_rate is not None:
            assert result.preferential_rate <= result.duty_rate

    @pytest.mark.asyncio
    async def test_anti_dumping_duty(self, calculator):
        result = await calculator.calculate_duty(
            cn_code="18032000",
            customs_value=Decimal("50000.00"),
            origin_country="CI",
        )
        assert isinstance(result.anti_dumping_duty, Decimal)
        assert result.anti_dumping_duty >= Decimal("0")


class TestLineItemCalculation:
    @pytest.mark.asyncio
    async def test_calculate_line_item(self, calculator):
        item = await calculator.calculate_line_item(
            line_number=1,
            cn_code="18010000",
            quantity=Decimal("10000.00"),
            unit_price=Decimal("2.50"),
            origin_country="CI",
        )
        assert isinstance(item, TariffLineItem)
        assert item.line_number == 1
        assert item.total_value == Decimal("25000.00")

    @pytest.mark.asyncio
    async def test_line_item_duty_computed(self, calculator):
        item = await calculator.calculate_line_item(
            line_number=1,
            cn_code="18010000",
            quantity=Decimal("1000.00"),
            unit_price=Decimal("3.00"),
            origin_country="CI",
        )
        assert isinstance(item.duty_amount, Decimal)
        assert item.duty_amount >= Decimal("0")

    @pytest.mark.asyncio
    async def test_line_item_vat_computed(self, calculator):
        item = await calculator.calculate_line_item(
            line_number=1,
            cn_code="18010000",
            quantity=Decimal("1000.00"),
            unit_price=Decimal("3.00"),
            origin_country="CI",
        )
        assert isinstance(item.vat_amount, Decimal)
        assert item.vat_amount >= Decimal("0")


# ====================================================================
# Health Check Tests
# ====================================================================


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_returns_dict(self, calculator):
        health = await calculator.health_check()
        assert isinstance(health, dict)
        assert health["engine"] == "ValueCalculator"

    @pytest.mark.asyncio
    async def test_status_healthy(self, calculator):
        health = await calculator.health_check()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_calculations_performed_zero(self, calculator):
        health = await calculator.health_check()
        assert health["calculations_performed"] == 0

    @pytest.mark.asyncio
    async def test_supported_currencies(self, calculator):
        health = await calculator.health_check()
        assert health["supported_currencies"] >= 4
