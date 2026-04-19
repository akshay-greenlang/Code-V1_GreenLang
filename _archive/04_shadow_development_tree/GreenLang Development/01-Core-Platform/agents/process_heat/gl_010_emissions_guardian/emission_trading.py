"""
Emission Trading Module - GL-010 EmissionsGuardian

This module implements carbon credit market integration for major emission trading
schemes including EU ETS, California Cap-and-Trade, RGGI, and voluntary markets.
Provides complete trading position management, allowance tracking, and compliance
forecasting.

Key Features:
    - EU ETS Phase 4 compliance and allowance tracking
    - California Cap-and-Trade (AB32) integration
    - RGGI (Regional Greenhouse Gas Initiative) support
    - Voluntary carbon markets (Verra VCS, Gold Standard)
    - Credit vintage and quality tracking
    - Position management and hedging analytics
    - Compliance forecasting and shortfall alerts
    - Price risk analysis with historical benchmarks

Regulatory References:
    - EU ETS Directive 2003/87/EC (as amended)
    - California AB32 / Cap-and-Trade Regulation
    - RGGI Model Rule
    - ICAO CORSIA (aviation offsets)

Example:
    >>> trading = EmissionTradingManager(
    ...     entity_id="FACILITY-001",
    ...     markets=[TradingMarket.EU_ETS, TradingMarket.VOLUNTARY]
    ... )
    >>> position = trading.calculate_compliance_position(year=2025)
    >>> print(f"Surplus/Deficit: {position.net_position:,.0f} allowances")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import statistics

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Carbon Market Reference Values
# =============================================================================

class MarketConstants:
    """Carbon market regulatory constants and thresholds."""

    # EU ETS Phase 4 (2021-2030) parameters
    EU_ETS_CAP_2024 = 1_386_000_000  # Total cap in tCO2e
    EU_ETS_LINEAR_REDUCTION_FACTOR = 0.0438  # 4.38% annual reduction
    EU_ETS_MSR_THRESHOLD_UPPER = 833_000_000  # Market Stability Reserve upper
    EU_ETS_MSR_THRESHOLD_LOWER = 400_000_000  # Market Stability Reserve lower

    # California Cap-and-Trade parameters
    CA_CT_CAP_2024 = 311_500_000  # California cap in tCO2e
    CA_CT_ALLOWANCE_PRICE_FLOOR = 23.20  # 2024 price floor (USD)
    CA_CT_ALLOWANCE_PRICE_CEILING = 76.86  # 2024 soft ceiling (USD)
    CA_CT_OFFSET_LIMIT_PCT = 4.0  # Max 4% offsets for compliance

    # RGGI parameters
    RGGI_CAP_2024 = 75_000_000  # RGGI cap in short tons
    RGGI_CCR_TRIGGER_PRICE = 15.39  # Cost Containment Reserve trigger (USD)
    RGGI_ECR_WITHHOLD_PCT = 10.0  # Emissions Containment Reserve

    # Vintage rules
    MAX_VINTAGE_AGE_YEARS = 8  # Maximum age for credit use
    VINTAGE_DISCOUNT_RATE = 0.02  # 2% discount per year of age

    # Quality scoring thresholds
    HIGH_QUALITY_SCORE = 80
    ACCEPTABLE_QUALITY_SCORE = 60


class TradingMarket(Enum):
    """Supported emission trading markets."""
    EU_ETS = "eu_ets"  # European Union Emissions Trading System
    CA_CAP_TRADE = "ca_cap_trade"  # California Cap-and-Trade
    RGGI = "rggi"  # Regional Greenhouse Gas Initiative
    UK_ETS = "uk_ets"  # UK Emissions Trading Scheme
    KOREA_ETS = "korea_ets"  # Korean Emissions Trading System
    CHINA_ETS = "china_ets"  # China National ETS
    VOLUNTARY = "voluntary"  # Voluntary carbon markets


class CreditRegistry(Enum):
    """Carbon credit registries."""
    EU_REGISTRY = "eu_registry"  # EU Transaction Log
    CITSS = "citss"  # California CITSS
    RGGI_COATS = "rggi_coats"  # RGGI CO2 Allowance Tracking System
    VERRA_VCS = "verra_vcs"  # Verified Carbon Standard
    GOLD_STANDARD = "gold_standard"  # Gold Standard Registry
    ACR = "acr"  # American Carbon Registry
    CAR = "car"  # Climate Action Reserve
    PURO_EARTH = "puro_earth"  # Puro.earth (removals)


class CreditType(Enum):
    """Types of carbon credits/allowances."""
    ALLOWANCE = "allowance"  # Compliance market allowance
    OFFSET = "offset"  # Project-based offset
    REMOVAL = "removal"  # Carbon removal credit
    AVOIDANCE = "avoidance"  # Avoided emissions credit


class ProjectType(Enum):
    """Carbon offset project types."""
    RENEWABLE_ENERGY = "renewable_energy"
    FOREST_CONSERVATION = "forest_conservation"
    AFFORESTATION = "afforestation"
    IMPROVED_COOKSTOVES = "improved_cookstoves"
    METHANE_CAPTURE = "methane_capture"
    INDUSTRIAL_GAS = "industrial_gas"
    ENERGY_EFFICIENCY = "energy_efficiency"
    DIRECT_AIR_CAPTURE = "direct_air_capture"
    BIOCHAR = "biochar"
    ENHANCED_WEATHERING = "enhanced_weathering"
    BLUE_CARBON = "blue_carbon"
    SOIL_CARBON = "soil_carbon"


class TransactionType(Enum):
    """Credit transaction types."""
    PURCHASE = "purchase"
    SALE = "sale"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"
    SURRENDER = "surrender"  # Compliance surrender
    RETIREMENT = "retirement"  # Voluntary retirement
    CANCELLATION = "cancellation"
    ALLOCATION = "allocation"  # Free allocation receipt


class ComplianceStatus(Enum):
    """Compliance position status."""
    SURPLUS = "surplus"
    BALANCED = "balanced"
    DEFICIT = "deficit"
    CRITICAL_DEFICIT = "critical_deficit"


# =============================================================================
# DATA MODELS
# =============================================================================

class CarbonCredit(BaseModel):
    """Individual carbon credit or allowance."""

    credit_id: str = Field(..., description="Unique credit identifier")
    registry: CreditRegistry = Field(..., description="Issuing registry")
    credit_type: CreditType = Field(..., description="Credit type")

    # Quantity and vintage
    quantity: float = Field(..., gt=0, description="Credit quantity (tCO2e)")
    vintage_year: int = Field(..., ge=2000, le=2100, description="Credit vintage year")

    # Market and pricing
    market: TradingMarket = Field(..., description="Applicable market")
    acquisition_price: Optional[float] = Field(
        default=None,
        ge=0,
        description="Acquisition price per credit"
    )
    acquisition_date: Optional[date] = Field(
        default=None,
        description="Acquisition date"
    )
    currency: str = Field(default="USD", description="Price currency")

    # Project details (for offsets)
    project_id: Optional[str] = Field(default=None, description="Project identifier")
    project_type: Optional[ProjectType] = Field(
        default=None,
        description="Offset project type"
    )
    project_country: Optional[str] = Field(
        default=None,
        description="Project host country"
    )

    # Quality metrics
    quality_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Quality score (0-100)"
    )
    additionality_verified: bool = Field(
        default=False,
        description="Additionality verification status"
    )
    permanence_years: Optional[int] = Field(
        default=None,
        description="Permanence commitment (years)"
    )

    # Status
    is_active: bool = Field(default=True, description="Credit active status")
    expiry_date: Optional[date] = Field(
        default=None,
        description="Credit expiry date"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    class Config:
        use_enum_values = True

    @property
    def age_years(self) -> int:
        """Calculate credit age in years."""
        return datetime.now().year - self.vintage_year

    @property
    def is_expired(self) -> bool:
        """Check if credit is expired."""
        if self.expiry_date:
            return date.today() > self.expiry_date
        return self.age_years > MarketConstants.MAX_VINTAGE_AGE_YEARS


class CreditPortfolio(BaseModel):
    """Portfolio of carbon credits/allowances."""

    entity_id: str = Field(..., description="Entity identifier")
    credits: List[CarbonCredit] = Field(
        default_factory=list,
        description="Portfolio credits"
    )

    # Summary metrics
    total_quantity: float = Field(default=0.0, description="Total credits (tCO2e)")
    total_value: float = Field(default=0.0, description="Total portfolio value")
    weighted_avg_price: float = Field(default=0.0, description="Weighted average price")
    avg_quality_score: float = Field(default=0.0, description="Average quality score")

    # By market breakdown
    by_market: Dict[str, float] = Field(
        default_factory=dict,
        description="Quantity by market"
    )
    by_vintage: Dict[int, float] = Field(
        default_factory=dict,
        description="Quantity by vintage year"
    )
    by_type: Dict[str, float] = Field(
        default_factory=dict,
        description="Quantity by credit type"
    )


class CreditTransaction(BaseModel):
    """Carbon credit transaction record."""

    transaction_id: str = Field(..., description="Transaction ID")
    transaction_type: TransactionType = Field(..., description="Transaction type")
    transaction_date: datetime = Field(..., description="Transaction timestamp")

    # Credit details
    credit_id: str = Field(..., description="Credit identifier")
    quantity: float = Field(..., gt=0, description="Transaction quantity")

    # Pricing
    price_per_unit: Optional[float] = Field(
        default=None,
        description="Price per credit"
    )
    total_value: Optional[float] = Field(
        default=None,
        description="Total transaction value"
    )
    currency: str = Field(default="USD", description="Currency")

    # Counterparty
    counterparty: Optional[str] = Field(
        default=None,
        description="Transaction counterparty"
    )

    # Compliance linkage
    compliance_period: Optional[str] = Field(
        default=None,
        description="Compliance period reference"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    class Config:
        use_enum_values = True


class CompliancePosition(BaseModel):
    """Compliance position analysis."""

    entity_id: str = Field(..., description="Entity identifier")
    market: TradingMarket = Field(..., description="Trading market")
    compliance_year: int = Field(..., description="Compliance year")

    # Emissions
    verified_emissions: float = Field(
        ...,
        ge=0,
        description="Verified emissions (tCO2e)"
    )
    projected_emissions: Optional[float] = Field(
        default=None,
        description="Projected full-year emissions"
    )

    # Allowances
    free_allocation: float = Field(
        default=0.0,
        ge=0,
        description="Free allocation received"
    )
    purchased_allowances: float = Field(
        default=0.0,
        ge=0,
        description="Purchased allowances"
    )
    banked_allowances: float = Field(
        default=0.0,
        ge=0,
        description="Banked from prior periods"
    )
    total_allowances: float = Field(
        default=0.0,
        ge=0,
        description="Total available allowances"
    )

    # Offsets
    eligible_offsets: float = Field(
        default=0.0,
        ge=0,
        description="Eligible offsets"
    )
    offset_limit: float = Field(
        default=0.0,
        ge=0,
        description="Maximum offset usage"
    )

    # Position
    net_position: float = Field(
        ...,
        description="Net position (surplus positive, deficit negative)"
    )
    status: ComplianceStatus = Field(..., description="Compliance status")

    # Financial
    shortfall_cost_estimate: Optional[float] = Field(
        default=None,
        description="Estimated cost to cover shortfall"
    )
    surplus_value_estimate: Optional[float] = Field(
        default=None,
        description="Estimated value of surplus"
    )

    # Risk metrics
    price_risk_exposure: Optional[float] = Field(
        default=None,
        description="Price risk exposure (VaR)"
    )

    # Timestamps
    calculation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    class Config:
        use_enum_values = True


class MarketPrice(BaseModel):
    """Carbon market price data."""

    market: TradingMarket = Field(..., description="Trading market")
    price_date: date = Field(..., description="Price date")
    settlement_price: float = Field(..., ge=0, description="Settlement price")
    currency: str = Field(default="EUR", description="Currency")

    # Price details
    bid_price: Optional[float] = Field(default=None, description="Bid price")
    ask_price: Optional[float] = Field(default=None, description="Ask price")
    volume: Optional[float] = Field(default=None, description="Trading volume")

    # Statistics
    day_high: Optional[float] = Field(default=None, description="Day high")
    day_low: Optional[float] = Field(default=None, description="Day low")
    ytd_change_pct: Optional[float] = Field(default=None, description="YTD change %")

    class Config:
        use_enum_values = True


class QualityAssessment(BaseModel):
    """Carbon credit quality assessment."""

    credit_id: str = Field(..., description="Credit identifier")
    assessment_date: datetime = Field(..., description="Assessment timestamp")

    # Core quality dimensions
    additionality_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Additionality score"
    )
    permanence_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Permanence score"
    )
    verification_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Verification quality score"
    )
    registry_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Registry credibility score"
    )
    co_benefits_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Co-benefits score"
    )

    # Composite score
    overall_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall quality score"
    )

    # Assessment details
    assessor: str = Field(..., description="Assessment entity")
    methodology: str = Field(..., description="Assessment methodology")
    confidence_level: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description="Confidence level"
    )

    # Flags
    concerns: List[str] = Field(
        default_factory=list,
        description="Quality concerns"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )


# =============================================================================
# EMISSION TRADING MANAGER
# =============================================================================

class EmissionTradingManager:
    """
    Carbon Emission Trading Manager for Multi-Market Compliance.

    Implements comprehensive carbon credit and allowance management for
    major emission trading schemes. Provides position tracking, compliance
    forecasting, and quality assessment for both compliance and voluntary
    markets.

    Features:
        - Multi-market support (EU ETS, CA C&T, RGGI, voluntary)
        - Portfolio management with vintage tracking
        - Compliance position analysis and forecasting
        - Credit quality scoring (additionality, permanence, verification)
        - Price risk analytics
        - Transaction audit trail with SHA-256 hashing

    Markets Supported:
        - EU ETS (European Union)
        - California Cap-and-Trade
        - RGGI (Northeastern US)
        - UK ETS
        - Voluntary markets (Verra VCS, Gold Standard)

    Example:
        >>> manager = EmissionTradingManager(
        ...     entity_id="FACILITY-001",
        ...     markets=[TradingMarket.EU_ETS]
        ... )
        >>> manager.add_credit(credit)
        >>> position = manager.calculate_compliance_position(2025)
    """

    # Quality scoring weights
    QUALITY_WEIGHTS = {
        "additionality": 0.30,
        "permanence": 0.25,
        "verification": 0.20,
        "registry": 0.15,
        "co_benefits": 0.10,
    }

    # Registry quality scores
    REGISTRY_SCORES = {
        CreditRegistry.EU_REGISTRY: 95,
        CreditRegistry.CITSS: 95,
        CreditRegistry.RGGI_COATS: 95,
        CreditRegistry.GOLD_STANDARD: 90,
        CreditRegistry.VERRA_VCS: 85,
        CreditRegistry.ACR: 80,
        CreditRegistry.CAR: 80,
        CreditRegistry.PURO_EARTH: 85,
    }

    def __init__(
        self,
        entity_id: str,
        markets: Optional[List[TradingMarket]] = None,
    ) -> None:
        """
        Initialize emission trading manager.

        Args:
            entity_id: Entity/facility identifier
            markets: List of active trading markets
        """
        self.entity_id = entity_id
        self.markets = markets or [TradingMarket.EU_ETS]

        # Portfolio management
        self._credits: List[CarbonCredit] = []
        self._transactions: List[CreditTransaction] = []

        # Price history
        self._price_history: Dict[str, List[MarketPrice]] = {}

        # Emissions tracking
        self._verified_emissions: Dict[int, float] = {}  # year -> tCO2e

        logger.info(
            f"EmissionTradingManager initialized for {entity_id} "
            f"with {len(self.markets)} market(s)"
        )

    def add_credit(
        self,
        registry: CreditRegistry,
        credit_type: CreditType,
        quantity: float,
        vintage_year: int,
        market: TradingMarket,
        acquisition_price: Optional[float] = None,
        project_id: Optional[str] = None,
        project_type: Optional[ProjectType] = None,
        project_country: Optional[str] = None,
    ) -> CarbonCredit:
        """
        Add carbon credit to portfolio.

        Args:
            registry: Credit registry
            credit_type: Credit type (allowance, offset, etc.)
            quantity: Credit quantity (tCO2e)
            vintage_year: Credit vintage year
            market: Applicable trading market
            acquisition_price: Price per credit (optional)
            project_id: Offset project ID (optional)
            project_type: Offset project type (optional)
            project_country: Project country (optional)

        Returns:
            Created CarbonCredit object

        Raises:
            ValueError: If credit parameters are invalid
        """
        logger.info(
            f"Adding {quantity:,.0f} credits from {registry.value}, "
            f"vintage {vintage_year}"
        )

        # Generate credit ID
        credit_id = f"{registry.value}_{vintage_year}_{len(self._credits) + 1:06d}"

        # Calculate provenance hash
        provenance_hash = self._hash_credit_data(
            credit_id=credit_id,
            registry=registry.value,
            quantity=quantity,
            vintage_year=vintage_year,
        )

        # Calculate quality score
        quality_score = None
        if credit_type == CreditType.OFFSET and project_type:
            quality_score = self._estimate_quality_score(
                registry=registry,
                project_type=project_type,
                vintage_year=vintage_year,
            )

        credit = CarbonCredit(
            credit_id=credit_id,
            registry=registry,
            credit_type=credit_type,
            quantity=quantity,
            vintage_year=vintage_year,
            market=market,
            acquisition_price=acquisition_price,
            acquisition_date=date.today() if acquisition_price else None,
            project_id=project_id,
            project_type=project_type,
            project_country=project_country,
            quality_score=quality_score,
            provenance_hash=provenance_hash,
        )

        self._credits.append(credit)

        # Record transaction
        if acquisition_price:
            self._record_transaction(
                credit_id=credit_id,
                transaction_type=TransactionType.PURCHASE,
                quantity=quantity,
                price_per_unit=acquisition_price,
            )

        logger.info(f"Credit {credit_id} added successfully")
        return credit

    def record_allocation(
        self,
        market: TradingMarket,
        quantity: float,
        vintage_year: int,
        allocation_type: str = "free",
    ) -> CarbonCredit:
        """
        Record free allocation receipt.

        Args:
            market: Trading market
            quantity: Allocation quantity (tCO2e)
            vintage_year: Allocation vintage year
            allocation_type: Allocation type (free, auction, etc.)

        Returns:
            Created CarbonCredit representing allocation
        """
        # Determine registry based on market
        registry_map = {
            TradingMarket.EU_ETS: CreditRegistry.EU_REGISTRY,
            TradingMarket.CA_CAP_TRADE: CreditRegistry.CITSS,
            TradingMarket.RGGI: CreditRegistry.RGGI_COATS,
            TradingMarket.UK_ETS: CreditRegistry.EU_REGISTRY,  # Simplified
        }
        registry = registry_map.get(market, CreditRegistry.EU_REGISTRY)

        credit = self.add_credit(
            registry=registry,
            credit_type=CreditType.ALLOWANCE,
            quantity=quantity,
            vintage_year=vintage_year,
            market=market,
            acquisition_price=0.0,  # Free allocation
        )

        # Record allocation transaction
        self._record_transaction(
            credit_id=credit.credit_id,
            transaction_type=TransactionType.ALLOCATION,
            quantity=quantity,
            price_per_unit=0.0,
        )

        return credit

    def surrender_credits(
        self,
        market: TradingMarket,
        quantity: float,
        compliance_year: int,
    ) -> List[str]:
        """
        Surrender credits for compliance.

        Implements FIFO (first-in-first-out) surrender with
        preference for oldest vintages within validity period.

        Args:
            market: Trading market
            quantity: Quantity to surrender
            compliance_year: Compliance year

        Returns:
            List of surrendered credit IDs

        Raises:
            ValueError: If insufficient credits available
        """
        logger.info(
            f"Surrendering {quantity:,.0f} credits for {market.value} "
            f"compliance year {compliance_year}"
        )

        # Get available credits for this market, sorted by vintage (oldest first)
        available = [
            c for c in self._credits
            if c.market == market.value and c.is_active and not c.is_expired
        ]
        available.sort(key=lambda c: c.vintage_year)

        total_available = sum(c.quantity for c in available)
        if total_available < quantity:
            raise ValueError(
                f"Insufficient credits: {total_available:,.0f} available, "
                f"{quantity:,.0f} required"
            )

        surrendered_ids = []
        remaining = quantity

        for credit in available:
            if remaining <= 0:
                break

            if credit.quantity <= remaining:
                # Surrender entire credit
                credit.is_active = False
                surrendered_ids.append(credit.credit_id)
                remaining -= credit.quantity

                self._record_transaction(
                    credit_id=credit.credit_id,
                    transaction_type=TransactionType.SURRENDER,
                    quantity=credit.quantity,
                    compliance_period=str(compliance_year),
                )
            else:
                # Partial surrender - split credit
                original_qty = credit.quantity
                credit.quantity = original_qty - remaining

                self._record_transaction(
                    credit_id=credit.credit_id,
                    transaction_type=TransactionType.SURRENDER,
                    quantity=remaining,
                    compliance_period=str(compliance_year),
                )

                surrendered_ids.append(f"{credit.credit_id}_partial")
                remaining = 0

        logger.info(f"Surrendered {len(surrendered_ids)} credit(s)")
        return surrendered_ids

    def calculate_compliance_position(
        self,
        compliance_year: int,
        market: Optional[TradingMarket] = None,
        projected_emissions: Optional[float] = None,
        market_price: Optional[float] = None,
    ) -> CompliancePosition:
        """
        Calculate compliance position for a given year.

        Analyzes current holdings against emissions obligations
        and provides surplus/deficit assessment with financial
        implications.

        Args:
            compliance_year: Year to analyze
            market: Specific market (default: first configured)
            projected_emissions: Projected emissions if not verified
            market_price: Current market price for valuation

        Returns:
            CompliancePosition with full analysis
        """
        target_market = market or self.markets[0]
        logger.info(
            f"Calculating compliance position for {self.entity_id}, "
            f"year {compliance_year}, market {target_market.value}"
        )

        # Get verified emissions
        verified = self._verified_emissions.get(compliance_year, 0.0)
        emissions = verified or projected_emissions or 0.0

        # Get available credits
        available_credits = [
            c for c in self._credits
            if c.market == target_market.value and c.is_active and not c.is_expired
        ]

        # Categorize credits
        allowances = [c for c in available_credits if c.credit_type == CreditType.ALLOWANCE.value]
        offsets = [c for c in available_credits if c.credit_type in [
            CreditType.OFFSET.value, CreditType.REMOVAL.value
        ]]

        total_allowances = sum(c.quantity for c in allowances)
        total_offsets = sum(c.quantity for c in offsets)

        # Calculate offset limit based on market rules
        if target_market == TradingMarket.CA_CAP_TRADE:
            offset_limit = emissions * (MarketConstants.CA_CT_OFFSET_LIMIT_PCT / 100)
        elif target_market == TradingMarket.EU_ETS:
            offset_limit = 0.0  # EU ETS no longer accepts offsets
        else:
            offset_limit = emissions * 0.05  # Default 5%

        eligible_offsets = min(total_offsets, offset_limit)

        # Calculate position
        total_coverage = total_allowances + eligible_offsets
        net_position = total_coverage - emissions

        # Determine status
        if net_position >= emissions * 0.1:
            status = ComplianceStatus.SURPLUS
        elif net_position >= 0:
            status = ComplianceStatus.BALANCED
        elif net_position >= -emissions * 0.1:
            status = ComplianceStatus.DEFICIT
        else:
            status = ComplianceStatus.CRITICAL_DEFICIT

        # Financial analysis
        shortfall_cost = None
        surplus_value = None
        price = market_price or self._get_default_price(target_market)

        if net_position < 0:
            shortfall_cost = abs(net_position) * price
        elif net_position > 0:
            surplus_value = net_position * price

        # Calculate price risk (simplified VaR at 95%)
        price_risk = abs(net_position) * price * 0.15  # 15% price volatility assumption

        # Provenance hash
        provenance_hash = hashlib.sha256(
            f"{self.entity_id}:{target_market.value}:{compliance_year}:"
            f"{emissions}:{net_position}:{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()

        position = CompliancePosition(
            entity_id=self.entity_id,
            market=target_market,
            compliance_year=compliance_year,
            verified_emissions=verified,
            projected_emissions=projected_emissions,
            free_allocation=sum(
                c.quantity for c in allowances
                if c.acquisition_price == 0.0
            ),
            purchased_allowances=sum(
                c.quantity for c in allowances
                if c.acquisition_price and c.acquisition_price > 0
            ),
            banked_allowances=sum(
                c.quantity for c in allowances
                if c.vintage_year < compliance_year
            ),
            total_allowances=total_allowances,
            eligible_offsets=eligible_offsets,
            offset_limit=offset_limit,
            net_position=net_position,
            status=status,
            shortfall_cost_estimate=shortfall_cost,
            surplus_value_estimate=surplus_value,
            price_risk_exposure=price_risk,
            provenance_hash=provenance_hash,
        )

        logger.info(
            f"Compliance position: {status.value}, "
            f"Net={net_position:,.0f} tCO2e"
        )

        return position

    def assess_credit_quality(
        self,
        credit_id: str,
        additionality_evidence: Optional[Dict[str, Any]] = None,
        permanence_mechanism: Optional[str] = None,
        third_party_verification: bool = False,
    ) -> QualityAssessment:
        """
        Assess carbon credit quality using multi-factor scoring.

        Evaluates credits against key quality dimensions:
        - Additionality (would project happen without carbon revenue?)
        - Permanence (will CO2 stay sequestered?)
        - Verification (independent third-party review?)
        - Registry credibility
        - Co-benefits (SDG alignment)

        Args:
            credit_id: Credit identifier
            additionality_evidence: Evidence supporting additionality
            permanence_mechanism: Permanence assurance mechanism
            third_party_verification: Third-party verification status

        Returns:
            QualityAssessment with scores and concerns

        Raises:
            ValueError: If credit not found
        """
        # Find credit
        credit = next(
            (c for c in self._credits if c.credit_id == credit_id),
            None
        )
        if credit is None:
            raise ValueError(f"Credit not found: {credit_id}")

        concerns = []
        recommendations = []

        # Additionality scoring
        additionality_score = 50.0  # Base score
        if additionality_evidence:
            if additionality_evidence.get("financial_additionality"):
                additionality_score += 20
            if additionality_evidence.get("regulatory_additionality"):
                additionality_score += 15
            if additionality_evidence.get("barrier_analysis"):
                additionality_score += 15
        else:
            concerns.append("Limited additionality documentation")
            recommendations.append("Obtain financial additionality analysis")

        # Permanence scoring
        permanence_score = 60.0  # Base score
        if credit.permanence_years:
            if credit.permanence_years >= 100:
                permanence_score = 95
            elif credit.permanence_years >= 40:
                permanence_score = 80
            elif credit.permanence_years >= 20:
                permanence_score = 65

        if permanence_mechanism == "buffer_pool":
            permanence_score += 10
        elif permanence_mechanism == "insurance":
            permanence_score += 15

        if credit.project_type in [
            ProjectType.DIRECT_AIR_CAPTURE.value,
            ProjectType.BIOCHAR.value,
            ProjectType.ENHANCED_WEATHERING.value,
        ]:
            permanence_score = min(permanence_score + 15, 100)  # Removal premium

        # Verification scoring
        verification_score = 70.0 if third_party_verification else 40.0
        if credit.additionality_verified:
            verification_score += 20

        # Registry scoring
        registry_score = self.REGISTRY_SCORES.get(
            CreditRegistry(credit.registry),
            60
        )

        # Co-benefits scoring (simplified)
        co_benefits_score = 50.0
        if credit.project_type in [
            ProjectType.FOREST_CONSERVATION.value,
            ProjectType.IMPROVED_COOKSTOVES.value,
            ProjectType.BLUE_CARBON.value,
        ]:
            co_benefits_score = 85.0  # High SDG alignment

        # Calculate overall score
        overall_score = (
            additionality_score * self.QUALITY_WEIGHTS["additionality"] +
            permanence_score * self.QUALITY_WEIGHTS["permanence"] +
            verification_score * self.QUALITY_WEIGHTS["verification"] +
            registry_score * self.QUALITY_WEIGHTS["registry"] +
            co_benefits_score * self.QUALITY_WEIGHTS["co_benefits"]
        )

        # Age penalty
        if credit.age_years > 3:
            age_penalty = (credit.age_years - 3) * 2
            overall_score -= age_penalty
            concerns.append(f"Vintage age {credit.age_years} years may reduce acceptance")

        overall_score = max(0, min(100, overall_score))

        # Quality tier warnings
        if overall_score < MarketConstants.ACCEPTABLE_QUALITY_SCORE:
            concerns.append("Credit quality below acceptable threshold")
            recommendations.append("Consider higher-quality credits for reputational risk")

        return QualityAssessment(
            credit_id=credit_id,
            assessment_date=datetime.now(timezone.utc),
            additionality_score=round(additionality_score, 1),
            permanence_score=round(permanence_score, 1),
            verification_score=round(verification_score, 1),
            registry_score=round(registry_score, 1),
            co_benefits_score=round(co_benefits_score, 1),
            overall_score=round(overall_score, 1),
            assessor="EmissionTradingManager",
            methodology="GreenLang Quality Framework v1.0",
            concerns=concerns,
            recommendations=recommendations,
        )

    def get_portfolio_summary(self) -> CreditPortfolio:
        """
        Get portfolio summary with aggregated metrics.

        Returns:
            CreditPortfolio with summary statistics
        """
        active_credits = [c for c in self._credits if c.is_active]

        if not active_credits:
            return CreditPortfolio(
                entity_id=self.entity_id,
                credits=active_credits,
            )

        total_quantity = sum(c.quantity for c in active_credits)

        # Calculate weighted average price
        total_value = 0.0
        priced_quantity = 0.0
        for credit in active_credits:
            if credit.acquisition_price:
                total_value += credit.quantity * credit.acquisition_price
                priced_quantity += credit.quantity

        weighted_avg_price = total_value / priced_quantity if priced_quantity > 0 else 0

        # Quality score average
        quality_credits = [c for c in active_credits if c.quality_score is not None]
        avg_quality = (
            statistics.mean(c.quality_score for c in quality_credits)
            if quality_credits else 0
        )

        # Breakdowns
        by_market: Dict[str, float] = {}
        by_vintage: Dict[int, float] = {}
        by_type: Dict[str, float] = {}

        for credit in active_credits:
            market_key = credit.market
            by_market[market_key] = by_market.get(market_key, 0) + credit.quantity

            by_vintage[credit.vintage_year] = (
                by_vintage.get(credit.vintage_year, 0) + credit.quantity
            )

            type_key = credit.credit_type
            by_type[type_key] = by_type.get(type_key, 0) + credit.quantity

        return CreditPortfolio(
            entity_id=self.entity_id,
            credits=active_credits,
            total_quantity=total_quantity,
            total_value=total_value,
            weighted_avg_price=round(weighted_avg_price, 2),
            avg_quality_score=round(avg_quality, 1),
            by_market=by_market,
            by_vintage=by_vintage,
            by_type=by_type,
        )

    def record_emissions(
        self,
        year: int,
        emissions: float,
        verified: bool = False,
    ) -> None:
        """
        Record emissions for compliance tracking.

        Args:
            year: Emission year
            emissions: Total emissions (tCO2e)
            verified: Whether emissions are third-party verified
        """
        self._verified_emissions[year] = emissions
        logger.info(
            f"Recorded {emissions:,.0f} tCO2e for {year} "
            f"({'verified' if verified else 'estimated'})"
        )

    def _record_transaction(
        self,
        credit_id: str,
        transaction_type: TransactionType,
        quantity: float,
        price_per_unit: Optional[float] = None,
        compliance_period: Optional[str] = None,
    ) -> CreditTransaction:
        """Record credit transaction."""
        transaction_id = f"TXN_{len(self._transactions) + 1:08d}"

        provenance_hash = hashlib.sha256(
            f"{transaction_id}:{credit_id}:{transaction_type.value}:"
            f"{quantity}:{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()

        total_value = quantity * price_per_unit if price_per_unit else None

        transaction = CreditTransaction(
            transaction_id=transaction_id,
            transaction_type=transaction_type,
            transaction_date=datetime.now(timezone.utc),
            credit_id=credit_id,
            quantity=quantity,
            price_per_unit=price_per_unit,
            total_value=total_value,
            compliance_period=compliance_period,
            provenance_hash=provenance_hash,
        )

        self._transactions.append(transaction)
        return transaction

    def _hash_credit_data(
        self,
        credit_id: str,
        registry: str,
        quantity: float,
        vintage_year: int,
    ) -> str:
        """Calculate SHA-256 hash for credit provenance."""
        import json

        hash_data = {
            "credit_id": credit_id,
            "registry": registry,
            "quantity": quantity,
            "vintage_year": vintage_year,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()

    def _estimate_quality_score(
        self,
        registry: CreditRegistry,
        project_type: ProjectType,
        vintage_year: int,
    ) -> float:
        """Estimate initial quality score based on basic attributes."""
        base_score = self.REGISTRY_SCORES.get(registry, 60)

        # Project type adjustments
        type_adjustments = {
            ProjectType.DIRECT_AIR_CAPTURE: 15,
            ProjectType.BIOCHAR: 10,
            ProjectType.ENHANCED_WEATHERING: 10,
            ProjectType.FOREST_CONSERVATION: 5,
            ProjectType.AFFORESTATION: 5,
            ProjectType.BLUE_CARBON: 5,
            ProjectType.RENEWABLE_ENERGY: -5,
            ProjectType.INDUSTRIAL_GAS: -10,
        }

        type_adj = type_adjustments.get(project_type, 0)

        # Vintage penalty
        age = datetime.now().year - vintage_year
        age_penalty = max(0, (age - 3) * 2)

        return max(0, min(100, base_score + type_adj - age_penalty))

    def _get_default_price(self, market: TradingMarket) -> float:
        """Get default market price for valuation."""
        # Representative prices as of 2024 (EUR/USD per tCO2e)
        default_prices = {
            TradingMarket.EU_ETS: 80.0,
            TradingMarket.CA_CAP_TRADE: 35.0,
            TradingMarket.RGGI: 15.0,
            TradingMarket.UK_ETS: 45.0,
            TradingMarket.VOLUNTARY: 15.0,
        }
        return default_prices.get(market, 50.0)

    def get_transaction_history(
        self,
        limit: int = 50,
        transaction_type: Optional[TransactionType] = None,
    ) -> List[CreditTransaction]:
        """
        Get transaction history.

        Args:
            limit: Maximum transactions to return
            transaction_type: Filter by type (optional)

        Returns:
            List of CreditTransaction objects
        """
        if transaction_type:
            filtered = [
                t for t in self._transactions
                if t.transaction_type == transaction_type.value
            ]
        else:
            filtered = self._transactions

        return list(reversed(filtered))[:limit]
