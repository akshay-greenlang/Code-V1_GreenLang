# -*- coding: utf-8 -*-
"""
Reference Data Package - AGENT-EUDR-011: Mass Balance Calculator Agent

Provides built-in reference datasets for mass balance calculations:
    - conversion_factors: Commodity yield ratios and acceptable ranges
      for every major processing step across the seven EUDR commodity
      classes (cocoa, oil_palm, coffee, soya, rubber, wood, cattle)
    - loss_tolerances: Maximum loss percentages for processing, transport,
      and storage operations, plus by-product recovery rates
    - credit_period_rules: Standard-specific credit period durations,
      carry-forward limits, grace periods, overdraft policies, and
      reconciliation requirements

These datasets enable deterministic, zero-hallucination mass balance
validation without external API dependencies.  All data is version-tracked
and provenance-auditable.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011 (Mass Balance Calculator)
Agent ID: GL-EUDR-MBC-011
"""

from greenlang.agents.eudr.mass_balance_calculator.reference_data.conversion_factors import (
    COMMODITY_PROCESS_INDEX,
    CONVERSION_FACTORS,
    TOTAL_COMMODITIES,
    TOTAL_CONVERSION_FACTORS,
    compute_deviation_pct,
    get_all_commodities,
    get_all_factors,
    get_all_processes,
    get_expected_yield,
    get_factor_range,
    get_reference_factor,
    get_source,
    is_within_range,
)
from greenlang.agents.eudr.mass_balance_calculator.reference_data.credit_period_rules import (
    CARRY_FORWARD_RULES,
    CREDIT_PERIOD_RULES,
    GRACE_PERIOD_RULES,
    OVERDRAFT_POLICIES,
    RECONCILIATION_REQUIREMENTS,
    TOTAL_STANDARDS,
    get_all_rules,
    get_all_standard_ids,
    get_carry_forward_rules,
    get_grace_period,
    get_grace_period_calendar_days,
    get_max_carry_forward_pct,
    get_overdraft_tolerance,
    get_period_duration_days,
    get_period_duration_months,
    get_period_rules,
    get_reconciliation_requirements,
    get_resolution_deadline_hours,
    get_standards_for_commodity,
    is_carry_forward_allowed,
    is_zero_overdraft,
)
from greenlang.agents.eudr.mass_balance_calculator.reference_data.loss_tolerances import (
    BYPRODUCT_RECOVERY_RATES,
    PROCESSING_LOSS_TOLERANCES,
    STORAGE_LOSS_TOLERANCES,
    TOTAL_BYPRODUCT_RATES,
    TOTAL_PROCESSING_TOLERANCES,
    TOTAL_STORAGE_TOLERANCES,
    TOTAL_TRANSPORT_TOLERANCES,
    TRANSPORT_LOSS_TOLERANCES,
    get_all_byproducts,
    get_all_tolerances,
    get_byproduct_recovery,
    get_expected_loss,
    get_loss_tolerance,
    get_max_loss,
    get_storage_tolerance,
    get_transport_tolerance,
    is_within_tolerance,
)

__all__ = [
    # -- conversion_factors --
    "CONVERSION_FACTORS",
    "COMMODITY_PROCESS_INDEX",
    "TOTAL_CONVERSION_FACTORS",
    "TOTAL_COMMODITIES",
    "get_reference_factor",
    "get_all_factors",
    "get_factor_range",
    "is_within_range",
    "get_expected_yield",
    "get_source",
    "get_all_commodities",
    "get_all_processes",
    "compute_deviation_pct",
    # -- loss_tolerances --
    "PROCESSING_LOSS_TOLERANCES",
    "TRANSPORT_LOSS_TOLERANCES",
    "STORAGE_LOSS_TOLERANCES",
    "BYPRODUCT_RECOVERY_RATES",
    "TOTAL_PROCESSING_TOLERANCES",
    "TOTAL_TRANSPORT_TOLERANCES",
    "TOTAL_STORAGE_TOLERANCES",
    "TOTAL_BYPRODUCT_RATES",
    "get_loss_tolerance",
    "is_within_tolerance",
    "get_all_tolerances",
    "get_expected_loss",
    "get_max_loss",
    "get_transport_tolerance",
    "get_storage_tolerance",
    "get_byproduct_recovery",
    "get_all_byproducts",
    # -- credit_period_rules --
    "CREDIT_PERIOD_RULES",
    "CARRY_FORWARD_RULES",
    "OVERDRAFT_POLICIES",
    "RECONCILIATION_REQUIREMENTS",
    "GRACE_PERIOD_RULES",
    "TOTAL_STANDARDS",
    "get_period_rules",
    "get_all_rules",
    "get_carry_forward_rules",
    "get_grace_period",
    "get_grace_period_calendar_days",
    "get_overdraft_tolerance",
    "get_period_duration_months",
    "get_period_duration_days",
    "get_reconciliation_requirements",
    "get_max_carry_forward_pct",
    "is_carry_forward_allowed",
    "is_zero_overdraft",
    "get_all_standard_ids",
    "get_standards_for_commodity",
    "get_resolution_deadline_hours",
]
