# -*- coding: utf-8 -*-
"""
Tests for MassBalanceEngine - AGENT-EUDR-009 Engine 4: Mass Balance Accounting

Comprehensive test suite covering:
- Input/output recording with running balance (F4.1, F4.3-F4.5)
- Conversion factor application (F4.6)
- Loss/waste accounting within tolerance (F4.7)
- Overdraft detection and alerts (F4.9)
- Credit period management (F4.2)
- Period-end reconciliation with variance (F4.10)
- Carry-forward with expiry (F4.8)
- Multi-commodity facility balance

Test count: 60+ tests
Coverage target: >= 85% of MassBalanceEngine module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-009 Chain of Custody Agent (GL-EUDR-COC-009)
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.chain_of_custody.conftest import (
    EUDR_COMMODITIES,
    CONVERSION_FACTORS,
    LOSS_TOLERANCES,
    CREDIT_PERIODS,
    FAC_ID_MILL_ID,
    FAC_ID_REFINERY_ID,
    FAC_ID_PROC_GH,
    FAC_ID_FACTORY_DE,
    FAC_ID_WAREHOUSE_NL,
    BATCH_ID_PALM_MILL_ID,
    BATCH_ID_PALM_CPO_ID,
    make_mass_balance_entry,
    make_batch,
    assert_mass_conservation,
)


# ===========================================================================
# 1. Input/Output Recording (F4.1, F4.3-F4.5)
# ===========================================================================


class TestInputOutputRecording:
    """Test recording inputs and outputs to the mass balance ledger."""

    def test_record_input(self, mass_balance_engine):
        """Record a compliant input to the ledger."""
        entry = make_mass_balance_entry(
            facility_id=FAC_ID_MILL_ID,
            commodity="palm_oil",
            entry_type="input",
            quantity_kg=10000.0,
            compliance_status="compliant",
        )
        result = mass_balance_engine.record_entry(entry)
        assert result is not None
        assert result["entry_type"] == "input"
        assert result["quantity_kg"] == 10000.0

    def test_record_output(self, mass_balance_engine):
        """Record an output from the ledger."""
        # First add input
        inp = make_mass_balance_entry(entry_type="input", quantity_kg=10000.0)
        mass_balance_engine.record_entry(inp)
        # Then record output
        out = make_mass_balance_entry(entry_type="output", quantity_kg=5000.0)
        result = mass_balance_engine.record_entry(out)
        assert result["entry_type"] == "output"

    def test_running_balance_calculated(self, mass_balance_engine):
        """Running balance is inputs minus outputs."""
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="input", quantity_kg=10000.0)
        )
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="output", quantity_kg=3000.0)
        )
        balance = mass_balance_engine.get_balance(FAC_ID_MILL_ID, "palm_oil")
        assert balance["current_balance_kg"] == pytest.approx(7000.0)

    def test_balance_starts_at_zero(self, mass_balance_engine):
        """New facility starts with zero balance."""
        balance = mass_balance_engine.get_balance("FAC-NEW-001", "cocoa")
        assert balance["current_balance_kg"] == 0.0

    def test_multiple_inputs_accumulate(self, mass_balance_engine):
        """Multiple inputs accumulate in the balance."""
        for i in range(5):
            mass_balance_engine.record_entry(
                make_mass_balance_entry(entry_type="input", quantity_kg=2000.0,
                                        entry_id=f"MBE-ACC-{i}")
            )
        balance = mass_balance_engine.get_balance(FAC_ID_MILL_ID, "palm_oil")
        assert balance["current_balance_kg"] == pytest.approx(10000.0)

    def test_invalid_entry_type_raises(self, mass_balance_engine):
        """Invalid entry type raises ValueError."""
        entry = make_mass_balance_entry(entry_type="invalid")
        with pytest.raises(ValueError):
            mass_balance_engine.record_entry(entry)

    def test_negative_quantity_raises(self, mass_balance_engine):
        """Negative quantity raises ValueError."""
        entry = make_mass_balance_entry(quantity_kg=-100.0)
        with pytest.raises(ValueError):
            mass_balance_engine.record_entry(entry)

    def test_zero_quantity_raises(self, mass_balance_engine):
        """Zero quantity raises ValueError."""
        entry = make_mass_balance_entry(quantity_kg=0.0)
        with pytest.raises(ValueError):
            mass_balance_engine.record_entry(entry)

    def test_entry_generates_provenance_hash(self, mass_balance_engine):
        """Each ledger entry has a provenance hash."""
        entry = make_mass_balance_entry(entry_type="input", quantity_kg=5000.0)
        result = mass_balance_engine.record_entry(entry)
        assert result.get("provenance_hash") is not None
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_record_all_commodities(self, mass_balance_engine, commodity):
        """Input/output recording works for all 7 EUDR commodities."""
        entry = make_mass_balance_entry(commodity=commodity, entry_type="input",
                                         quantity_kg=1000.0)
        result = mass_balance_engine.record_entry(entry)
        assert result["commodity"] == commodity


# ===========================================================================
# 2. Conversion Factor Application (F4.6)
# ===========================================================================


class TestConversionFactors:
    """Test conversion factor application for yield ratios."""

    def test_cocoa_beans_to_nibs_factor(self, mass_balance_engine):
        """Cocoa beans to nibs uses 0.87 yield ratio."""
        result = mass_balance_engine.apply_conversion(
            input_commodity_form="beans",
            output_commodity_form="nibs",
            commodity="cocoa",
            input_quantity_kg=10000.0,
        )
        assert result["expected_output_kg"] == pytest.approx(8700.0)

    def test_palm_ffb_to_cpo_factor(self, mass_balance_engine):
        """Palm FFB to CPO uses ~0.22 yield ratio."""
        result = mass_balance_engine.apply_conversion(
            input_commodity_form="ffb",
            output_commodity_form="cpo",
            commodity="palm_oil",
            input_quantity_kg=50000.0,
        )
        assert result["expected_output_kg"] == pytest.approx(11000.0, rel=0.05)

    def test_coffee_cherry_to_green_factor(self, mass_balance_engine):
        """Coffee cherry to green uses ~0.18 yield ratio."""
        result = mass_balance_engine.apply_conversion(
            input_commodity_form="cherry",
            output_commodity_form="green",
            commodity="coffee",
            input_quantity_kg=8000.0,
        )
        assert result["expected_output_kg"] == pytest.approx(1440.0, rel=0.05)

    def test_soya_beans_to_oil_factor(self, mass_balance_engine):
        """Soya beans to oil uses ~0.19 yield ratio."""
        result = mass_balance_engine.apply_conversion(
            input_commodity_form="beans",
            output_commodity_form="oil",
            commodity="soya",
            input_quantity_kg=100000.0,
        )
        assert result["expected_output_kg"] == pytest.approx(19000.0, rel=0.05)

    @pytest.mark.parametrize("conv_key,conv_data", list(CONVERSION_FACTORS.items()))
    def test_all_conversion_factors(self, mass_balance_engine, conv_key, conv_data):
        """All conversion factors from reference data are available."""
        factor = mass_balance_engine.get_conversion_factor(conv_key)
        assert factor == pytest.approx(conv_data["yield_ratio"], rel=0.01)

    def test_unknown_conversion_raises(self, mass_balance_engine):
        """Unknown conversion raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            mass_balance_engine.apply_conversion(
                "unknown_form", "also_unknown", "cocoa", 1000.0
            )

    def test_conversion_deterministic(self, mass_balance_engine):
        """Conversion calculations are deterministic (zero-hallucination)."""
        r1 = mass_balance_engine.apply_conversion("beans", "nibs", "cocoa", 10000.0)
        r2 = mass_balance_engine.apply_conversion("beans", "nibs", "cocoa", 10000.0)
        assert r1["expected_output_kg"] == r2["expected_output_kg"]


# ===========================================================================
# 3. Loss/Waste Accounting (F4.7)
# ===========================================================================


class TestLossWasteAccounting:
    """Test loss and waste accounting within tolerance."""

    def test_loss_within_tolerance_accepted(self, mass_balance_engine):
        """Loss within commodity tolerance is accepted."""
        result = mass_balance_engine.validate_loss(
            commodity="cocoa",
            input_kg=10000.0,
            output_kg=9850.0,
            waste_kg=100.0,
        )
        assert result["within_tolerance"] is True

    def test_loss_exceeding_tolerance_flagged(self, mass_balance_engine):
        """Loss exceeding commodity tolerance is flagged."""
        result = mass_balance_engine.validate_loss(
            commodity="cocoa",
            input_kg=10000.0,
            output_kg=9000.0,
            waste_kg=500.0,
        )
        assert result["within_tolerance"] is False

    @pytest.mark.parametrize("commodity,tolerance", list(LOSS_TOLERANCES.items()))
    def test_loss_tolerances_per_commodity(self, mass_balance_engine, commodity, tolerance):
        """Each commodity has its defined loss tolerance."""
        tol = mass_balance_engine.get_loss_tolerance(commodity)
        assert tol == pytest.approx(tolerance, abs=0.005)

    def test_zero_loss_accepted(self, mass_balance_engine):
        """Zero loss (perfect processing) is always accepted."""
        result = mass_balance_engine.validate_loss(
            commodity="cocoa",
            input_kg=1000.0,
            output_kg=1000.0,
            waste_kg=0.0,
        )
        assert result["within_tolerance"] is True

    def test_waste_recorded_separately(self, mass_balance_engine):
        """Waste is tracked as a separate line item in the ledger."""
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="input", quantity_kg=10000.0)
        )
        result = mass_balance_engine.record_waste(
            facility_id=FAC_ID_MILL_ID,
            commodity="palm_oil",
            waste_kg=500.0,
            reason="Processing residue",
        )
        assert result["waste_kg"] == 500.0


# ===========================================================================
# 4. Overdraft Detection (F4.9)
# ===========================================================================


class TestOverdraftDetection:
    """Test overdraft detection when outputs exceed inputs."""

    def test_overdraft_detected(self, mass_balance_engine):
        """Output exceeding input triggers overdraft alert."""
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="input", quantity_kg=5000.0,
                                    entry_id="MBE-OD-IN")
        )
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="output", quantity_kg=3000.0,
                                    entry_id="MBE-OD-OUT1")
        )
        result = mass_balance_engine.check_overdraft(FAC_ID_MILL_ID, "palm_oil")
        assert result["overdraft"] is False

        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="output", quantity_kg=3000.0,
                                    entry_id="MBE-OD-OUT2")
        )
        result = mass_balance_engine.check_overdraft(FAC_ID_MILL_ID, "palm_oil")
        assert result["overdraft"] is True

    def test_no_overdraft_when_balanced(self, mass_balance_engine):
        """No overdraft when outputs equal inputs."""
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="input", quantity_kg=5000.0,
                                    entry_id="MBE-BAL-IN")
        )
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="output", quantity_kg=5000.0,
                                    entry_id="MBE-BAL-OUT")
        )
        result = mass_balance_engine.check_overdraft(FAC_ID_MILL_ID, "palm_oil")
        assert result["overdraft"] is False

    def test_overdraft_amount_calculated(self, mass_balance_engine):
        """Overdraft amount is correctly calculated."""
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="input", quantity_kg=3000.0,
                                    entry_id="MBE-ODA-IN")
        )
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="output", quantity_kg=3500.0,
                                    entry_id="MBE-ODA-OUT")
        )
        result = mass_balance_engine.check_overdraft(FAC_ID_MILL_ID, "palm_oil")
        assert result["overdraft"] is True
        assert result["overdraft_kg"] == pytest.approx(500.0)


# ===========================================================================
# 5. Credit Period Management (F4.2)
# ===========================================================================


class TestCreditPeriodManagement:
    """Test credit period management for mass balance."""

    @pytest.mark.parametrize("standard,months", list(CREDIT_PERIODS.items()))
    def test_credit_periods_per_standard(self, mass_balance_engine, standard, months):
        """Each certification standard has its defined credit period."""
        period = mass_balance_engine.get_credit_period(standard)
        assert period == months

    def test_credit_period_default(self, mass_balance_engine):
        """Default credit period when no standard specified."""
        period = mass_balance_engine.get_credit_period("DEFAULT")
        assert period in (3, 12)

    def test_expired_credits_not_counted(self, mass_balance_engine):
        """Credits past their expiry date are not counted in balance."""
        now = datetime.now(timezone.utc)
        expired_entry = make_mass_balance_entry(
            entry_type="input", quantity_kg=5000.0,
            credit_period_start=(now - timedelta(days=400)).isoformat(),
            entry_id="MBE-EXP-IN",
        )
        mass_balance_engine.record_entry(expired_entry)
        balance = mass_balance_engine.get_balance(
            FAC_ID_MILL_ID, "palm_oil",
            exclude_expired=True,
        )
        assert balance["current_balance_kg"] == 0.0


# ===========================================================================
# 6. Period-End Reconciliation (F4.10)
# ===========================================================================


class TestPeriodEndReconciliation:
    """Test period-end reconciliation with variance reporting."""

    def test_reconciliation_balanced(self, mass_balance_engine):
        """Reconciliation with balanced ledger shows zero variance."""
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="input", quantity_kg=10000.0,
                                    entry_id="MBE-REC-IN")
        )
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="output", quantity_kg=10000.0,
                                    entry_id="MBE-REC-OUT")
        )
        result = mass_balance_engine.reconcile(FAC_ID_MILL_ID, "palm_oil")
        assert result["variance_kg"] == pytest.approx(0.0)
        assert result["variance_pct"] == pytest.approx(0.0)

    def test_reconciliation_with_variance(self, mass_balance_engine):
        """Reconciliation reports variance when inputs != outputs."""
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="input", quantity_kg=10000.0,
                                    entry_id="MBE-RECV-IN")
        )
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="output", quantity_kg=9500.0,
                                    entry_id="MBE-RECV-OUT")
        )
        result = mass_balance_engine.reconcile(FAC_ID_MILL_ID, "palm_oil")
        assert result["variance_kg"] == pytest.approx(500.0)
        assert result["variance_pct"] == pytest.approx(5.0)

    def test_reconciliation_generates_report(self, mass_balance_engine):
        """Reconciliation generates a structured report."""
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="input", quantity_kg=5000.0,
                                    entry_id="MBE-RPT-IN")
        )
        result = mass_balance_engine.reconcile(FAC_ID_MILL_ID, "palm_oil")
        assert "total_input_kg" in result
        assert "total_output_kg" in result
        assert "variance_kg" in result


# ===========================================================================
# 7. Carry-Forward (F4.8)
# ===========================================================================


class TestCarryForward:
    """Test balance carry-forward between periods."""

    def test_carry_forward_basic(self, mass_balance_engine):
        """Remaining balance carries forward to next period."""
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="input", quantity_kg=10000.0,
                                    entry_id="MBE-CF-IN")
        )
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="output", quantity_kg=7000.0,
                                    entry_id="MBE-CF-OUT")
        )
        result = mass_balance_engine.carry_forward(FAC_ID_MILL_ID, "palm_oil")
        assert result["carried_forward_kg"] == pytest.approx(3000.0)

    def test_carry_forward_zero_balance(self, mass_balance_engine):
        """Zero balance results in zero carry-forward."""
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="input", quantity_kg=5000.0,
                                    entry_id="MBE-CFZ-IN")
        )
        mass_balance_engine.record_entry(
            make_mass_balance_entry(entry_type="output", quantity_kg=5000.0,
                                    entry_id="MBE-CFZ-OUT")
        )
        result = mass_balance_engine.carry_forward(FAC_ID_MILL_ID, "palm_oil")
        assert result["carried_forward_kg"] == pytest.approx(0.0)


# ===========================================================================
# 8. Multi-Commodity Facility Balance
# ===========================================================================


class TestMultiCommodityBalance:
    """Test balance tracking per commodity at multi-commodity facilities."""

    def test_separate_balances_per_commodity(self, mass_balance_engine):
        """Each commodity has its own balance at a facility."""
        mass_balance_engine.record_entry(
            make_mass_balance_entry(
                facility_id=FAC_ID_WAREHOUSE_NL, commodity="cocoa",
                entry_type="input", quantity_kg=5000.0, entry_id="MBE-MC-COC")
        )
        mass_balance_engine.record_entry(
            make_mass_balance_entry(
                facility_id=FAC_ID_WAREHOUSE_NL, commodity="coffee",
                entry_type="input", quantity_kg=3000.0, entry_id="MBE-MC-COF")
        )
        cocoa_balance = mass_balance_engine.get_balance(FAC_ID_WAREHOUSE_NL, "cocoa")
        coffee_balance = mass_balance_engine.get_balance(FAC_ID_WAREHOUSE_NL, "coffee")
        assert cocoa_balance["current_balance_kg"] == pytest.approx(5000.0)
        assert coffee_balance["current_balance_kg"] == pytest.approx(3000.0)

    def test_output_does_not_affect_other_commodity(self, mass_balance_engine):
        """Output from one commodity does not reduce another's balance."""
        mass_balance_engine.record_entry(
            make_mass_balance_entry(
                facility_id=FAC_ID_WAREHOUSE_NL, commodity="cocoa",
                entry_type="input", quantity_kg=5000.0, entry_id="MBE-ISO-COC-IN")
        )
        mass_balance_engine.record_entry(
            make_mass_balance_entry(
                facility_id=FAC_ID_WAREHOUSE_NL, commodity="coffee",
                entry_type="input", quantity_kg=3000.0, entry_id="MBE-ISO-COF-IN")
        )
        mass_balance_engine.record_entry(
            make_mass_balance_entry(
                facility_id=FAC_ID_WAREHOUSE_NL, commodity="cocoa",
                entry_type="output", quantity_kg=2000.0, entry_id="MBE-ISO-COC-OUT")
        )
        coffee_balance = mass_balance_engine.get_balance(FAC_ID_WAREHOUSE_NL, "coffee")
        assert coffee_balance["current_balance_kg"] == pytest.approx(3000.0)

    def test_balance_history(self, mass_balance_engine):
        """Balance history shows all entries for a facility-commodity pair."""
        for i in range(3):
            mass_balance_engine.record_entry(
                make_mass_balance_entry(
                    facility_id=FAC_ID_MILL_ID, commodity="palm_oil",
                    entry_type="input", quantity_kg=1000.0 * (i + 1),
                    entry_id=f"MBE-HIST-{i}")
            )
        history = mass_balance_engine.get_history(FAC_ID_MILL_ID, "palm_oil")
        assert len(history) >= 3
