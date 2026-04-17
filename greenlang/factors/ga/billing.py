# -*- coding: utf-8 -*-
"""
Billing engine and usage metering for Factors catalog (F101).

Provides tier-based billing plans, API usage metering, overage tracking,
and invoice generation for the Factors product.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BillingTier(str, Enum):
    COMMUNITY = "community"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class BillingPlan:
    """Billing plan definition for a tier."""

    tier: BillingTier
    monthly_price_usd: float
    included_api_calls: int
    overage_per_1k_calls: float
    included_connectors: List[str]
    max_results_per_query: int
    sla_uptime_pct: float
    support_level: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "monthly_price_usd": self.monthly_price_usd,
            "included_api_calls": self.included_api_calls,
            "overage_per_1k_calls": self.overage_per_1k_calls,
            "included_connectors": self.included_connectors,
            "max_results_per_query": self.max_results_per_query,
            "sla_uptime_pct": self.sla_uptime_pct,
            "support_level": self.support_level,
        }


# Default pricing plans
PLANS: Dict[BillingTier, BillingPlan] = {
    BillingTier.COMMUNITY: BillingPlan(
        tier=BillingTier.COMMUNITY,
        monthly_price_usd=0.0,
        included_api_calls=1000,
        overage_per_1k_calls=0.0,  # hard cap
        included_connectors=[],
        max_results_per_query=25,
        sla_uptime_pct=99.0,
        support_level="community",
    ),
    BillingTier.PRO: BillingPlan(
        tier=BillingTier.PRO,
        monthly_price_usd=299.0,
        included_api_calls=50000,
        overage_per_1k_calls=5.0,
        included_connectors=["electricity_maps"],
        max_results_per_query=100,
        sla_uptime_pct=99.5,
        support_level="email",
    ),
    BillingTier.ENTERPRISE: BillingPlan(
        tier=BillingTier.ENTERPRISE,
        monthly_price_usd=999.0,
        included_api_calls=500000,
        overage_per_1k_calls=3.0,
        included_connectors=["iea_statistics", "ecoinvent", "electricity_maps"],
        max_results_per_query=500,
        sla_uptime_pct=99.9,
        support_level="dedicated",
    ),
}


@dataclass
class UsageRecord:
    """Monthly usage record for a tenant."""

    tenant_id: str
    month: str  # "2026-04"
    api_calls: int = 0
    search_calls: int = 0
    match_calls: int = 0
    batch_calls: int = 0
    connector_calls: int = 0

    @property
    def total_calls(self) -> int:
        return self.api_calls


@dataclass
class InvoiceLineItem:
    """A single line item on an invoice."""

    description: str
    quantity: int
    unit_price: float
    total: float


@dataclass
class Invoice:
    """Monthly invoice for a tenant."""

    invoice_id: str
    tenant_id: str
    month: str
    tier: str
    base_price: float
    overage_amount: float
    total_amount: float
    line_items: List[InvoiceLineItem] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invoice_id": self.invoice_id,
            "tenant_id": self.tenant_id,
            "month": self.month,
            "tier": self.tier,
            "base_price": self.base_price,
            "overage_amount": round(self.overage_amount, 2),
            "total_amount": round(self.total_amount, 2),
            "line_items": [
                {"description": li.description, "quantity": li.quantity, "total": li.total}
                for li in self.line_items
            ],
            "created_at": self.created_at,
        }


class UsageMeter:
    """
    Tracks API usage per tenant per month.
    """

    def __init__(self) -> None:
        self._records: Dict[str, UsageRecord] = {}  # key: "tenant_id:month"

    def _key(self, tenant_id: str, month: str) -> str:
        return f"{tenant_id}:{month}"

    def record_call(self, tenant_id: str, call_type: str = "api") -> UsageRecord:
        """Record a single API call."""
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        k = self._key(tenant_id, month)
        if k not in self._records:
            self._records[k] = UsageRecord(tenant_id=tenant_id, month=month)
        rec = self._records[k]
        rec.api_calls += 1
        if call_type == "search":
            rec.search_calls += 1
        elif call_type == "match":
            rec.match_calls += 1
        elif call_type == "batch":
            rec.batch_calls += 1
        elif call_type == "connector":
            rec.connector_calls += 1
        return rec

    def get_usage(self, tenant_id: str, month: str) -> Optional[UsageRecord]:
        return self._records.get(self._key(tenant_id, month))

    def get_current_usage(self, tenant_id: str) -> Optional[UsageRecord]:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        return self.get_usage(tenant_id, month)


class BillingEngine:
    """
    Generates invoices based on usage and billing plans.
    """

    def __init__(self, meter: UsageMeter, plans: Optional[Dict[BillingTier, BillingPlan]] = None) -> None:
        self._meter = meter
        self._plans = plans or PLANS
        self._tenant_tiers: Dict[str, BillingTier] = {}
        self._invoices: List[Invoice] = []

    def set_tenant_tier(self, tenant_id: str, tier: BillingTier) -> None:
        self._tenant_tiers[tenant_id] = tier

    def get_plan(self, tier: BillingTier) -> BillingPlan:
        return self._plans[tier]

    def generate_invoice(self, tenant_id: str, month: str) -> Invoice:
        """Generate an invoice for a tenant's monthly usage."""
        tier = self._tenant_tiers.get(tenant_id, BillingTier.COMMUNITY)
        plan = self._plans[tier]
        usage = self._meter.get_usage(tenant_id, month)
        total_calls = usage.total_calls if usage else 0

        line_items: List[InvoiceLineItem] = []

        # Base subscription
        line_items.append(InvoiceLineItem(
            description=f"{tier.value.title()} plan - monthly subscription",
            quantity=1,
            unit_price=plan.monthly_price_usd,
            total=plan.monthly_price_usd,
        ))

        # Overage calculation
        overage_amount = 0.0
        if total_calls > plan.included_api_calls and plan.overage_per_1k_calls > 0:
            overage_calls = total_calls - plan.included_api_calls
            overage_amount = (overage_calls / 1000) * plan.overage_per_1k_calls
            line_items.append(InvoiceLineItem(
                description=f"API overage: {overage_calls} calls beyond {plan.included_api_calls} included",
                quantity=overage_calls,
                unit_price=plan.overage_per_1k_calls / 1000,
                total=overage_amount,
            ))

        invoice = Invoice(
            invoice_id=f"inv_{uuid.uuid4().hex[:12]}",
            tenant_id=tenant_id,
            month=month,
            tier=tier.value,
            base_price=plan.monthly_price_usd,
            overage_amount=overage_amount,
            total_amount=plan.monthly_price_usd + overage_amount,
            line_items=line_items,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self._invoices.append(invoice)
        logger.info(
            "Invoice generated: %s tenant=%s month=%s total=$%.2f",
            invoice.invoice_id, tenant_id, month, invoice.total_amount,
        )
        return invoice

    def list_invoices(self, tenant_id: Optional[str] = None) -> List[Invoice]:
        if tenant_id:
            return [i for i in self._invoices if i.tenant_id == tenant_id]
        return list(self._invoices)

    def is_within_quota(self, tenant_id: str) -> bool:
        """Check if tenant is within their API call quota."""
        tier = self._tenant_tiers.get(tenant_id, BillingTier.COMMUNITY)
        plan = self._plans[tier]
        usage = self._meter.get_current_usage(tenant_id)
        if not usage:
            return True
        # Community tier has hard cap
        if tier == BillingTier.COMMUNITY:
            return usage.total_calls < plan.included_api_calls
        return True  # Pro/Enterprise allow overage
