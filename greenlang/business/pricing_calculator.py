"""
GreenLang Pricing Calculator Module

Provides comprehensive pricing calculations including:
- Edition pricing
- Add-on pricing
- Bundle discounts
- ROI calculations
- TCO calculations
- Usage-based pricing

Example:
    >>> calculator = PricingCalculator()
    >>> quote = calculator.generate_quote(
    ...     edition="professional",
    ...     add_ons=["GL-CBAM-APP"],
    ...     term_years=2,
    ...     users=75
    ... )
    >>> print(quote.summary())
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json


class PricingTier(Enum):
    """Pricing tier/edition enumeration."""
    ESSENTIALS = "essentials"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class DiscountType(Enum):
    """Types of discounts available."""
    VOLUME = "volume"
    MULTI_YEAR = "multi_year"
    MULTI_PRODUCT = "multi_product"
    STRATEGIC = "strategic"
    ACADEMIC = "academic"
    NONPROFIT = "nonprofit"
    STARTUP = "startup"
    PARTNER = "partner"


@dataclass
class PricingConfig:
    """Configuration for pricing calculations."""

    # Base edition pricing (annual)
    edition_prices: Dict[str, Decimal] = field(default_factory=lambda: {
        "essentials": Decimal("100000"),
        "professional": Decimal("250000"),
        "enterprise": Decimal("500000"),
    })

    # Edition allocations
    edition_allocations: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "essentials": {
            "users": 10,
            "data_points_monthly": 100000,
            "agents": 25,
            "facilities": 5,
            "api_calls_monthly": 100000,
            "storage_gb": 50,
            "reports_monthly": 100,
        },
        "professional": {
            "users": 50,
            "data_points_monthly": 1000000,
            "agents": 100,
            "facilities": 25,
            "api_calls_monthly": 1000000,
            "storage_gb": 500,
            "reports_monthly": 1000,
        },
        "enterprise": {
            "users": -1,  # Unlimited
            "data_points_monthly": -1,
            "agents": 500,
            "facilities": -1,
            "api_calls_monthly": -1,
            "storage_gb": 5000,
            "reports_monthly": -1,
        },
    })

    # Overage pricing
    overage_prices: Dict[str, Decimal] = field(default_factory=lambda: {
        "user": Decimal("1000"),  # Per user per year
        "data_point": Decimal("0.005"),  # Per point
        "agent": Decimal("2000"),  # Per agent per year
        "facility": Decimal("5000"),  # Per facility per year
        "api_call": Decimal("0.00001"),  # Per call
        "storage_gb": Decimal("10"),  # Per GB per month
        "report": Decimal("5"),  # Per report
    })

    # Add-on module pricing
    addon_prices: Dict[str, Decimal] = field(default_factory=lambda: {
        # Application modules
        "GL-CSRD-APP": Decimal("150000"),
        "GL-CBAM-APP": Decimal("100000"),
        "GL-EUDR-APP": Decimal("75000"),
        "GL-ProcessHeat-APP": Decimal("200000"),
        "GL-SmartBuilding-APP": Decimal("150000"),
        "GL-EVFleet-APP": Decimal("100000"),
        "GL-Scope3-APP": Decimal("175000"),
        "GL-ClimateRisk-APP": Decimal("250000"),
        # Feature modules
        "ML-Optimization": Decimal("50000"),
        "Edge-Computing": Decimal("75000"),
        "Real-Time-Streaming": Decimal("40000"),
        "Advanced-Analytics": Decimal("30000"),
        "Custom-Dashboards": Decimal("15000"),
        "API-Write-Access": Decimal("20000"),
        "White-Label": Decimal("50000"),
        # Integration modules
        "SAP-Connector": Decimal("25000"),
        "Oracle-Connector": Decimal("25000"),
        "Microsoft-Connector": Decimal("20000"),
        "Salesforce-Connector": Decimal("15000"),
        "BMS-Connector-Pack": Decimal("30000"),
        "IoT-Platform-Pack": Decimal("25000"),
    })

    # Volume discount tiers
    volume_discounts: List[Tuple[Decimal, Decimal]] = field(default_factory=lambda: [
        (Decimal("100000"), Decimal("0")),
        (Decimal("250000"), Decimal("0.05")),
        (Decimal("500000"), Decimal("0.10")),
        (Decimal("1000000"), Decimal("0.15")),
        (Decimal("2000000"), Decimal("0.20")),
        (Decimal("5000000"), Decimal("0.25")),
    ])

    # Multi-year discounts
    multiyear_discounts: Dict[int, Decimal] = field(default_factory=lambda: {
        1: Decimal("0"),
        2: Decimal("0.10"),
        3: Decimal("0.15"),
        5: Decimal("0.20"),
    })

    # Multi-product discounts
    multiproduct_discounts: Dict[int, Decimal] = field(default_factory=lambda: {
        1: Decimal("0"),
        2: Decimal("0.05"),
        3: Decimal("0.10"),
        4: Decimal("0.12"),
        5: Decimal("0.15"),
    })

    # Special discounts
    special_discounts: Dict[str, Decimal] = field(default_factory=lambda: {
        "academic": Decimal("0.75"),
        "nonprofit": Decimal("0.50"),
        "government": Decimal("0.25"),
        "startup_preseed": Decimal("0.90"),
        "startup_seed": Decimal("0.75"),
        "startup_seriesa": Decimal("0.50"),
        "startup_growth": Decimal("0.25"),
    })

    # Implementation package pricing
    implementation_prices: Dict[str, Decimal] = field(default_factory=lambda: {
        "quickstart": Decimal("20000"),
        "standard": Decimal("50000"),
        "comprehensive": Decimal("120000"),
        "enterprise": Decimal("200000"),
    })

    # Professional services rates
    service_rates: Dict[str, Decimal] = field(default_factory=lambda: {
        "consulting": Decimal("300"),
        "development": Decimal("250"),
        "training_daily": Decimal("2500"),
        "technical_support": Decimal("200"),
        "data_migration": Decimal("175"),
        "custom_reporting": Decimal("150"),
    })


@dataclass
class LineItem:
    """Individual line item in a quote."""
    item_code: str
    description: str
    quantity: int
    unit_price: Decimal
    total_price: Decimal
    category: str
    notes: Optional[str] = None


@dataclass
class Discount:
    """Applied discount on a quote."""
    discount_type: DiscountType
    description: str
    percentage: Decimal
    amount: Decimal


@dataclass
class Quote:
    """Complete pricing quote."""
    quote_id: str
    customer_name: str
    created_date: datetime
    valid_until: date
    edition: PricingTier
    term_years: int
    line_items: List[LineItem]
    discounts: List[Discount]
    subtotal: Decimal
    total_discount: Decimal
    total_price: Decimal
    annual_price: Decimal
    monthly_equivalent: Decimal
    currency: str = "USD"
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a human-readable summary of the quote."""
        lines = [
            f"Quote ID: {self.quote_id}",
            f"Customer: {self.customer_name}",
            f"Edition: {self.edition.value.title()}",
            f"Term: {self.term_years} year(s)",
            "",
            "Line Items:",
            "-" * 60,
        ]

        for item in self.line_items:
            lines.append(
                f"  {item.description:40} ${item.total_price:>12,.2f}"
            )

        lines.extend([
            "-" * 60,
            f"  {'Subtotal':40} ${self.subtotal:>12,.2f}",
        ])

        if self.discounts:
            lines.append("")
            lines.append("Discounts Applied:")
            for discount in self.discounts:
                lines.append(
                    f"  {discount.description:40} -${discount.amount:>11,.2f}"
                )

        lines.extend([
            "-" * 60,
            f"  {'Total Annual Price':40} ${self.annual_price:>12,.2f}",
            f"  {'Monthly Equivalent':40} ${self.monthly_equivalent:>12,.2f}",
            f"  {'Total Contract Value ({} yr)'.format(self.term_years):40} "
            f"${self.total_price:>12,.2f}",
        ])

        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert quote to dictionary for serialization."""
        return {
            "quote_id": self.quote_id,
            "customer_name": self.customer_name,
            "created_date": self.created_date.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "edition": self.edition.value,
            "term_years": self.term_years,
            "line_items": [
                {
                    "item_code": li.item_code,
                    "description": li.description,
                    "quantity": li.quantity,
                    "unit_price": str(li.unit_price),
                    "total_price": str(li.total_price),
                    "category": li.category,
                    "notes": li.notes,
                }
                for li in self.line_items
            ],
            "discounts": [
                {
                    "discount_type": d.discount_type.value,
                    "description": d.description,
                    "percentage": str(d.percentage),
                    "amount": str(d.amount),
                }
                for d in self.discounts
            ],
            "subtotal": str(self.subtotal),
            "total_discount": str(self.total_discount),
            "total_price": str(self.total_price),
            "annual_price": str(self.annual_price),
            "monthly_equivalent": str(self.monthly_equivalent),
            "currency": self.currency,
            "notes": self.notes,
        }


class PricingCalculator:
    """
    Calculator for GreenLang pricing, quotes, and discounts.

    Example:
        >>> calculator = PricingCalculator()
        >>> quote = calculator.generate_quote(
        ...     customer_name="Acme Corp",
        ...     edition="professional",
        ...     add_ons=["GL-CBAM-APP", "GL-ProcessHeat-APP"],
        ...     term_years=3,
        ...     extra_users=25
        ... )
        >>> print(f"Annual: ${quote.annual_price:,.2f}")
    """

    def __init__(self, config: Optional[PricingConfig] = None):
        """Initialize the pricing calculator with optional custom configuration."""
        self.config = config or PricingConfig()

    def _generate_quote_id(self, customer_name: str) -> str:
        """Generate a unique quote ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_input = f"{customer_name}{timestamp}".encode()
        hash_suffix = hashlib.md5(hash_input).hexdigest()[:6].upper()
        return f"GL-Q-{timestamp}-{hash_suffix}"

    def get_edition_price(self, edition: str) -> Decimal:
        """Get base price for an edition."""
        edition_lower = edition.lower()
        if edition_lower not in self.config.edition_prices:
            raise ValueError(f"Unknown edition: {edition}")
        return self.config.edition_prices[edition_lower]

    def get_addon_price(self, addon_code: str) -> Decimal:
        """Get price for an add-on module."""
        if addon_code not in self.config.addon_prices:
            raise ValueError(f"Unknown add-on: {addon_code}")
        return self.config.addon_prices[addon_code]

    def calculate_volume_discount(self, total: Decimal) -> Decimal:
        """Calculate volume discount percentage based on total."""
        discount = Decimal("0")
        for threshold, disc in self.config.volume_discounts:
            if total >= threshold:
                discount = disc
        return discount

    def calculate_multiyear_discount(self, years: int) -> Decimal:
        """Calculate multi-year discount percentage."""
        if years in self.config.multiyear_discounts:
            return self.config.multiyear_discounts[years]
        # For years not explicitly defined, use the highest available
        max_years = max(y for y in self.config.multiyear_discounts.keys() if y <= years)
        return self.config.multiyear_discounts[max_years]

    def calculate_multiproduct_discount(self, product_count: int) -> Decimal:
        """Calculate multi-product discount percentage."""
        if product_count in self.config.multiproduct_discounts:
            return self.config.multiproduct_discounts[product_count]
        # Cap at maximum defined
        max_products = max(self.config.multiproduct_discounts.keys())
        if product_count > max_products:
            return self.config.multiproduct_discounts[max_products]
        return Decimal("0")

    def calculate_overage(
        self,
        edition: str,
        extra_users: int = 0,
        extra_agents: int = 0,
        extra_facilities: int = 0,
    ) -> Decimal:
        """Calculate overage costs for exceeding edition allocations."""
        total = Decimal("0")

        if extra_users > 0:
            total += Decimal(extra_users) * self.config.overage_prices["user"]
        if extra_agents > 0:
            total += Decimal(extra_agents) * self.config.overage_prices["agent"]
        if extra_facilities > 0:
            total += Decimal(extra_facilities) * self.config.overage_prices["facility"]

        return total

    def generate_quote(
        self,
        customer_name: str,
        edition: str,
        add_ons: Optional[List[str]] = None,
        term_years: int = 1,
        extra_users: int = 0,
        extra_agents: int = 0,
        extra_facilities: int = 0,
        implementation_package: Optional[str] = None,
        special_discount_type: Optional[str] = None,
        valid_days: int = 30,
    ) -> Quote:
        """
        Generate a complete pricing quote.

        Args:
            customer_name: Name of the customer
            edition: Edition tier (essentials, professional, enterprise)
            add_ons: List of add-on module codes
            term_years: Contract term in years
            extra_users: Additional users beyond allocation
            extra_agents: Additional agents beyond allocation
            extra_facilities: Additional facilities beyond allocation
            implementation_package: Implementation package name
            special_discount_type: Special discount category
            valid_days: Number of days quote is valid

        Returns:
            Complete Quote object
        """
        add_ons = add_ons or []
        line_items: List[LineItem] = []
        discounts: List[Discount] = []

        # Parse edition
        edition_lower = edition.lower()
        try:
            edition_enum = PricingTier(edition_lower)
        except ValueError:
            raise ValueError(f"Unknown edition: {edition}")

        # Add base edition
        edition_price = self.get_edition_price(edition_lower)
        line_items.append(LineItem(
            item_code=f"GL-{edition_lower.upper()}-001",
            description=f"GreenLang {edition.title()} Edition (Annual)",
            quantity=1,
            unit_price=edition_price,
            total_price=edition_price,
            category="edition",
        ))

        # Add add-ons
        for addon in add_ons:
            addon_price = self.get_addon_price(addon)
            line_items.append(LineItem(
                item_code=addon,
                description=f"{addon} Module (Annual)",
                quantity=1,
                unit_price=addon_price,
                total_price=addon_price,
                category="addon",
            ))

        # Add overages
        if extra_users > 0:
            user_price = self.config.overage_prices["user"]
            line_items.append(LineItem(
                item_code="GL-OVERAGE-USERS",
                description=f"Additional Users ({extra_users})",
                quantity=extra_users,
                unit_price=user_price,
                total_price=user_price * extra_users,
                category="overage",
            ))

        if extra_agents > 0:
            agent_price = self.config.overage_prices["agent"]
            line_items.append(LineItem(
                item_code="GL-OVERAGE-AGENTS",
                description=f"Additional Agents ({extra_agents})",
                quantity=extra_agents,
                unit_price=agent_price,
                total_price=agent_price * extra_agents,
                category="overage",
            ))

        if extra_facilities > 0:
            facility_price = self.config.overage_prices["facility"]
            line_items.append(LineItem(
                item_code="GL-OVERAGE-FACILITIES",
                description=f"Additional Facilities ({extra_facilities})",
                quantity=extra_facilities,
                unit_price=facility_price,
                total_price=facility_price * extra_facilities,
                category="overage",
            ))

        # Add implementation if specified
        if implementation_package:
            impl_price = self.config.implementation_prices.get(
                implementation_package.lower()
            )
            if impl_price:
                line_items.append(LineItem(
                    item_code=f"GL-IMPL-{implementation_package.upper()}",
                    description=f"Implementation - {implementation_package.title()}",
                    quantity=1,
                    unit_price=impl_price,
                    total_price=impl_price,
                    category="services",
                    notes="One-time fee",
                ))

        # Calculate subtotal (annual recurring)
        recurring_subtotal = sum(
            li.total_price for li in line_items
            if li.category != "services"
        )
        services_total = sum(
            li.total_price for li in line_items
            if li.category == "services"
        )
        subtotal = recurring_subtotal

        # Apply discounts
        total_discount_pct = Decimal("0")

        # Volume discount
        volume_discount_pct = self.calculate_volume_discount(subtotal)
        if volume_discount_pct > 0:
            discount_amount = subtotal * volume_discount_pct
            discounts.append(Discount(
                discount_type=DiscountType.VOLUME,
                description=f"Volume Discount ({volume_discount_pct * 100:.0f}%)",
                percentage=volume_discount_pct,
                amount=discount_amount,
            ))
            total_discount_pct += volume_discount_pct

        # Multi-year discount
        if term_years > 1:
            multiyear_discount_pct = self.calculate_multiyear_discount(term_years)
            if multiyear_discount_pct > 0:
                discount_amount = subtotal * multiyear_discount_pct
                discounts.append(Discount(
                    discount_type=DiscountType.MULTI_YEAR,
                    description=f"{term_years}-Year Term Discount ({multiyear_discount_pct * 100:.0f}%)",
                    percentage=multiyear_discount_pct,
                    amount=discount_amount,
                ))
                total_discount_pct += multiyear_discount_pct

        # Multi-product discount (count products including edition)
        product_count = 1 + len(add_ons)
        multiproduct_discount_pct = self.calculate_multiproduct_discount(product_count)
        if multiproduct_discount_pct > 0:
            discount_amount = subtotal * multiproduct_discount_pct
            discounts.append(Discount(
                discount_type=DiscountType.MULTI_PRODUCT,
                description=f"Multi-Product Discount ({multiproduct_discount_pct * 100:.0f}%)",
                percentage=multiproduct_discount_pct,
                amount=discount_amount,
            ))
            total_discount_pct += multiproduct_discount_pct

        # Special discount
        if special_discount_type:
            special_key = special_discount_type.lower().replace(" ", "_")
            if special_key in self.config.special_discounts:
                special_discount_pct = self.config.special_discounts[special_key]
                discount_amount = subtotal * special_discount_pct
                discounts.append(Discount(
                    discount_type=DiscountType.ACADEMIC if "academic" in special_key
                                 else DiscountType.NONPROFIT if "nonprofit" in special_key
                                 else DiscountType.STARTUP if "startup" in special_key
                                 else DiscountType.STRATEGIC,
                    description=f"{special_discount_type.title()} Discount ({special_discount_pct * 100:.0f}%)",
                    percentage=special_discount_pct,
                    amount=discount_amount,
                ))
                total_discount_pct += special_discount_pct

        # Cap total discount at 40% (unless special)
        if not special_discount_type and total_discount_pct > Decimal("0.40"):
            total_discount_pct = Decimal("0.40")

        # Calculate final amounts
        total_discount = sum(d.amount for d in discounts)
        annual_price = (subtotal - total_discount).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        monthly_equivalent = (annual_price / 12).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        total_contract = (annual_price * term_years + services_total).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Generate quote
        now = datetime.now()
        quote = Quote(
            quote_id=self._generate_quote_id(customer_name),
            customer_name=customer_name,
            created_date=now,
            valid_until=date(now.year, now.month, now.day),
            edition=edition_enum,
            term_years=term_years,
            line_items=line_items,
            discounts=discounts,
            subtotal=subtotal,
            total_discount=total_discount,
            total_price=total_contract,
            annual_price=annual_price,
            monthly_equivalent=monthly_equivalent,
            notes=[],
        )

        # Add notes
        if services_total > 0:
            quote.notes.append(
                f"One-time services fee of ${services_total:,.2f} included in total contract value"
            )
        if term_years > 1:
            quote.notes.append(
                f"Annual billing in advance for {term_years}-year term"
            )

        return quote


@dataclass
class ROIResult:
    """Result of ROI calculation."""
    investment_year1: Decimal
    investment_annual: Decimal
    investment_total: Decimal
    benefits_year1: Decimal
    benefits_annual: Decimal
    benefits_total: Decimal
    net_benefit_year1: Decimal
    net_benefit_total: Decimal
    roi_year1: Decimal
    roi_total: Decimal
    payback_months: Decimal
    yearly_breakdown: List[Dict[str, Decimal]]

    def summary(self) -> str:
        """Generate human-readable ROI summary."""
        return f"""
ROI Analysis Summary
{'='*50}
Investment (Year 1):     ${self.investment_year1:>12,.2f}
Investment (5-Year):     ${self.investment_total:>12,.2f}
Benefits (Year 1):       ${self.benefits_year1:>12,.2f}
Benefits (5-Year):       ${self.benefits_total:>12,.2f}
Net Benefit (Year 1):    ${self.net_benefit_year1:>12,.2f}
Net Benefit (5-Year):    ${self.net_benefit_total:>12,.2f}
ROI (Year 1):            {self.roi_year1:>12.0f}%
ROI (5-Year):            {self.roi_total:>12.0f}%
Payback Period:          {self.payback_months:>12.1f} months
{'='*50}
"""


class ROICalculator:
    """
    Calculator for Return on Investment projections.

    Example:
        >>> calculator = ROICalculator()
        >>> result = calculator.calculate_roi(
        ...     annual_license=250000,
        ...     implementation=50000,
        ...     energy_spend=5000000,
        ...     energy_savings_pct=0.20,
        ...     compliance_savings=500000,
        ...     labor_savings=200000,
        ... )
        >>> print(result.summary())
    """

    def calculate_roi(
        self,
        annual_license: float,
        implementation: float = 0,
        training: float = 15000,
        internal_resources: float = 50000,
        energy_spend: float = 0,
        energy_savings_pct: float = 0.20,
        energy_savings_growth: float = 0.10,
        compliance_savings: float = 0,
        labor_savings: float = 0,
        consultant_savings: float = 0,
        carbon_credit_savings: float = 0,
        penalty_avoidance: float = 0,
        years: int = 5,
    ) -> ROIResult:
        """
        Calculate comprehensive ROI projection.

        Args:
            annual_license: Annual license/subscription cost
            implementation: One-time implementation cost
            training: Annual training cost
            internal_resources: Annual internal staff allocation
            energy_spend: Current annual energy spend
            energy_savings_pct: Expected energy savings percentage
            energy_savings_growth: Year-over-year savings growth
            compliance_savings: Annual compliance cost reduction
            labor_savings: Annual labor productivity savings
            consultant_savings: Annual consultant cost reduction
            carbon_credit_savings: Annual carbon credit cost reduction
            penalty_avoidance: Annual penalty risk avoided
            years: Number of years to project

        Returns:
            ROIResult with complete analysis
        """
        yearly_breakdown = []
        total_investment = Decimal("0")
        total_benefits = Decimal("0")

        for year in range(1, years + 1):
            # Calculate costs for this year
            costs = {
                "license": Decimal(str(annual_license)),
                "implementation": Decimal(str(implementation)) if year == 1 else Decimal("0"),
                "training": Decimal(str(training)) if year == 1 else Decimal(str(training * 0.5)),
                "internal_resources": Decimal(str(internal_resources)),
            }
            year_investment = sum(costs.values())

            # Calculate benefits for this year (with growth factor)
            growth_factor = 1 + energy_savings_growth * (year - 1)
            benefits = {
                "energy_savings": Decimal(str(energy_spend * energy_savings_pct * growth_factor)),
                "compliance_savings": Decimal(str(compliance_savings)),
                "labor_savings": Decimal(str(labor_savings * (1 + 0.05 * (year - 1)))),
                "consultant_savings": Decimal(str(consultant_savings)),
                "carbon_credit_savings": Decimal(str(carbon_credit_savings * (1 + 0.10 * (year - 1)))),
                "penalty_avoidance": Decimal(str(penalty_avoidance)),
            }
            year_benefits = sum(benefits.values())

            yearly_breakdown.append({
                "year": year,
                "investment": year_investment,
                "benefits": year_benefits,
                "net_benefit": year_benefits - year_investment,
                **costs,
                **benefits,
            })

            total_investment += year_investment
            total_benefits += year_benefits

        # Calculate summary metrics
        investment_year1 = yearly_breakdown[0]["investment"]
        benefits_year1 = yearly_breakdown[0]["benefits"]
        net_benefit_year1 = benefits_year1 - investment_year1
        net_benefit_total = total_benefits - total_investment

        roi_year1 = (net_benefit_year1 / investment_year1 * 100) if investment_year1 > 0 else Decimal("0")
        roi_total = (net_benefit_total / total_investment * 100) if total_investment > 0 else Decimal("0")

        # Calculate payback period
        cumulative_benefit = Decimal("0")
        payback_months = Decimal("0")
        for breakdown in yearly_breakdown:
            monthly_benefit = breakdown["net_benefit"] / 12
            for month in range(1, 13):
                cumulative_benefit += monthly_benefit
                payback_months += 1
                if cumulative_benefit >= 0:
                    break
            if cumulative_benefit >= 0:
                break

        return ROIResult(
            investment_year1=investment_year1,
            investment_annual=Decimal(str(annual_license + internal_resources + training * 0.5)),
            investment_total=total_investment,
            benefits_year1=benefits_year1,
            benefits_annual=total_benefits / years,
            benefits_total=total_benefits,
            net_benefit_year1=net_benefit_year1,
            net_benefit_total=net_benefit_total,
            roi_year1=roi_year1,
            roi_total=roi_total,
            payback_months=payback_months,
            yearly_breakdown=yearly_breakdown,
        )


@dataclass
class TCOResult:
    """Result of TCO calculation."""
    direct_costs: Dict[str, Decimal]
    indirect_costs: Dict[str, Decimal]
    hidden_costs: Dict[str, Decimal]
    total_by_year: List[Decimal]
    total_tco: Decimal
    average_annual: Decimal
    cost_breakdown: Dict[str, Decimal]

    def summary(self) -> str:
        """Generate human-readable TCO summary."""
        lines = [
            "Total Cost of Ownership Analysis",
            "=" * 50,
            "",
            "Direct Costs:",
        ]
        for key, value in self.direct_costs.items():
            lines.append(f"  {key:30} ${value:>12,.2f}")

        lines.append("\nIndirect Costs:")
        for key, value in self.indirect_costs.items():
            lines.append(f"  {key:30} ${value:>12,.2f}")

        lines.append("\nHidden Costs:")
        for key, value in self.hidden_costs.items():
            lines.append(f"  {key:30} ${value:>12,.2f}")

        lines.extend([
            "",
            "-" * 50,
            f"  {'5-Year Total TCO':30} ${self.total_tco:>12,.2f}",
            f"  {'Average Annual':30} ${self.average_annual:>12,.2f}",
        ])

        return "\n".join(lines)


class TCOCalculator:
    """
    Calculator for Total Cost of Ownership analysis.

    Example:
        >>> calculator = TCOCalculator()
        >>> result = calculator.calculate_tco(
        ...     annual_license=250000,
        ...     implementation=50000,
        ...     internal_fte=0.5,
        ...     fte_cost=150000,
        ... )
        >>> print(result.summary())
    """

    def calculate_tco(
        self,
        annual_license: float,
        implementation: float = 50000,
        training_initial: float = 15000,
        training_ongoing: float = 5000,
        integration_initial: float = 25000,
        integration_ongoing: float = 10000,
        internal_fte: float = 0.25,
        fte_cost: float = 150000,
        change_management: float = 20000,
        learning_curve_cost: float = 10000,
        opportunity_cost: float = 25000,
        years: int = 5,
    ) -> TCOResult:
        """
        Calculate comprehensive TCO analysis.

        Args:
            annual_license: Annual subscription cost
            implementation: One-time implementation cost
            training_initial: Initial training cost
            training_ongoing: Ongoing annual training cost
            integration_initial: Initial integration cost
            integration_ongoing: Ongoing integration maintenance
            internal_fte: FTE allocation for internal resources
            fte_cost: Fully loaded annual FTE cost
            change_management: Change management budget
            learning_curve_cost: Productivity loss during transition
            opportunity_cost: Opportunity cost during transition
            years: Number of years to project

        Returns:
            TCOResult with complete analysis
        """
        direct_costs = {
            "license_total": Decimal(str(annual_license * years)),
            "implementation": Decimal(str(implementation)),
            "training_total": Decimal(str(training_initial + training_ongoing * (years - 1))),
            "integration_total": Decimal(str(integration_initial + integration_ongoing * (years - 1))),
        }

        indirect_costs = {
            "internal_resources": Decimal(str(internal_fte * fte_cost * years)),
            "change_management": Decimal(str(change_management)),
        }

        hidden_costs = {
            "learning_curve": Decimal(str(learning_curve_cost)),
            "opportunity_cost": Decimal(str(opportunity_cost)),
        }

        # Calculate yearly breakdown
        total_by_year = []
        for year in range(1, years + 1):
            year_cost = Decimal(str(annual_license))
            year_cost += Decimal(str(internal_fte * fte_cost))

            if year == 1:
                year_cost += Decimal(str(implementation))
                year_cost += Decimal(str(training_initial))
                year_cost += Decimal(str(integration_initial))
                year_cost += Decimal(str(change_management))
                year_cost += Decimal(str(learning_curve_cost))
                year_cost += Decimal(str(opportunity_cost))
            else:
                year_cost += Decimal(str(training_ongoing))
                year_cost += Decimal(str(integration_ongoing))

            total_by_year.append(year_cost)

        total_tco = sum(direct_costs.values()) + sum(indirect_costs.values()) + sum(hidden_costs.values())

        return TCOResult(
            direct_costs=direct_costs,
            indirect_costs=indirect_costs,
            hidden_costs=hidden_costs,
            total_by_year=total_by_year,
            total_tco=total_tco,
            average_annual=total_tco / years,
            cost_breakdown={
                **direct_costs,
                **indirect_costs,
                **hidden_costs,
            },
        )


# Usage-based pricing calculator
class UsagePricingCalculator:
    """Calculator for usage-based pricing model."""

    def __init__(self):
        self.base_fee = Decimal("500")  # Monthly base fee
        self.unit_prices = {
            "data_points": Decimal("0.001"),
            "api_calls": Decimal("0.00001"),
            "agent_invocations": Decimal("0.10"),
            "reports": Decimal("5.00"),
            "mau": Decimal("50"),
        }
        self.volume_tiers = [
            (Decimal("5000"), Decimal("0")),
            (Decimal("10000"), Decimal("0.10")),
            (Decimal("25000"), Decimal("0.15")),
            (Decimal("50000"), Decimal("0.20")),
            (Decimal("100000"), Decimal("0.25")),
        ]

    def calculate_monthly_cost(
        self,
        data_points: int = 0,
        api_calls: int = 0,
        agent_invocations: int = 0,
        reports: int = 0,
        mau: int = 0,
    ) -> Dict[str, Decimal]:
        """Calculate monthly usage-based cost."""
        usage_costs = {
            "data_points": Decimal(data_points) * self.unit_prices["data_points"],
            "api_calls": Decimal(api_calls) * self.unit_prices["api_calls"],
            "agent_invocations": Decimal(agent_invocations) * self.unit_prices["agent_invocations"],
            "reports": Decimal(reports) * self.unit_prices["reports"],
            "mau": Decimal(mau) * self.unit_prices["mau"],
        }

        usage_total = sum(usage_costs.values())

        # Apply volume discount
        discount_pct = Decimal("0")
        for threshold, discount in self.volume_tiers:
            if usage_total >= threshold:
                discount_pct = discount

        discount_amount = usage_total * discount_pct

        return {
            "base_fee": self.base_fee,
            **usage_costs,
            "usage_total": usage_total,
            "volume_discount": discount_amount,
            "monthly_total": self.base_fee + usage_total - discount_amount,
        }
