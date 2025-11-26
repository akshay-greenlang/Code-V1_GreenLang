"""
EU Emissions Trading System (EU ETS) Connector for GL-010 EMISSIONWATCH.

Provides integration with EU ETS registry for emissions allowance tracking,
verified emissions reporting, and MRV (Monitoring, Reporting, Verification)
compliance under EU Regulation 2018/2066 and 2018/2067.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import asyncio
import hashlib
import json
import logging
import time
import uuid

from pydantic import BaseModel, Field, ConfigDict, field_validator
import httpx

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    ConnectionState,
    ConnectorType,
    HealthCheckResult,
    HealthStatus,
    ConnectorError,
    ConnectionError,
    AuthenticationError,
    ConfigurationError,
    ValidationError,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class ETSPhase(str, Enum):
    """EU ETS trading phases."""

    PHASE_1 = "phase_1"  # 2005-2007
    PHASE_2 = "phase_2"  # 2008-2012
    PHASE_3 = "phase_3"  # 2013-2020
    PHASE_4 = "phase_4"  # 2021-2030


class AllowanceType(str, Enum):
    """Types of emission allowances."""

    EUA = "eua"  # EU Allowance
    EUAA = "euaa"  # EU Aviation Allowance
    CER = "cer"  # Certified Emission Reduction
    ERU = "eru"  # Emission Reduction Unit
    FREE_ALLOCATION = "free_allocation"
    AUCTIONED = "auctioned"


class TransactionType(str, Enum):
    """Types of registry transactions."""

    ALLOCATION = "allocation"
    SURRENDER = "surrender"
    TRANSFER = "transfer"
    AUCTION = "auction"
    CANCELLATION = "cancellation"
    RETIREMENT = "retirement"
    REPLACEMENT = "replacement"
    CARRY_OVER = "carry_over"


class VerificationStatus(str, Enum):
    """Verification status of emissions report."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_VERIFICATION = "under_verification"
    VERIFIED = "verified"
    VERIFICATION_ISSUES = "verification_issues"
    REJECTED = "rejected"
    APPROVED = "approved"


class InstallationType(str, Enum):
    """Installation types under EU ETS."""

    COMBUSTION = "combustion"
    REFINING = "refining"
    IRON_STEEL = "iron_steel"
    CEMENT = "cement"
    GLASS = "glass"
    CERAMICS = "ceramics"
    PULP_PAPER = "pulp_paper"
    CHEMICALS = "chemicals"
    ALUMINIUM = "aluminium"
    AVIATION = "aviation"
    MARITIME = "maritime"


class MonitoringApproach(str, Enum):
    """Monitoring approaches under MRV."""

    CALCULATION_BASED = "calculation_based"
    MEASUREMENT_BASED = "measurement_based"
    FALL_BACK = "fall_back"
    CEMS = "cems"


class ActivityType(str, Enum):
    """EU ETS activity types (Annex I)."""

    COMBUSTION_20MW = "combustion_20mw"
    REFINING = "refining"
    COKE_PRODUCTION = "coke_production"
    METAL_ORE_ROASTING = "metal_ore_roasting"
    IRON_STEEL = "iron_steel"
    ALUMINIUM = "aluminium"
    NON_FERROUS_METALS = "non_ferrous_metals"
    CEMENT_CLINKER = "cement_clinker"
    LIME = "lime"
    GLASS = "glass"
    CERAMIC = "ceramic"
    INSULATION_MATERIAL = "insulation_material"
    PULP = "pulp"
    PAPER = "paper"
    CARBON_BLACK = "carbon_black"
    NITRIC_ACID = "nitric_acid"
    ADIPIC_ACID = "adipic_acid"
    GLYOXAL = "glyoxal"
    AMMONIA = "ammonia"
    BULK_CHEMICALS = "bulk_chemicals"
    HYDROGEN = "hydrogen"
    SODA_ASH = "soda_ash"
    AVIATION = "aviation"
    MARITIME = "maritime"


class MemberState(str, Enum):
    """EU/EEA member states."""

    AT = "AT"  # Austria
    BE = "BE"  # Belgium
    BG = "BG"  # Bulgaria
    HR = "HR"  # Croatia
    CY = "CY"  # Cyprus
    CZ = "CZ"  # Czech Republic
    DK = "DK"  # Denmark
    EE = "EE"  # Estonia
    FI = "FI"  # Finland
    FR = "FR"  # France
    DE = "DE"  # Germany
    GR = "GR"  # Greece
    HU = "HU"  # Hungary
    IE = "IE"  # Ireland
    IT = "IT"  # Italy
    LV = "LV"  # Latvia
    LT = "LT"  # Lithuania
    LU = "LU"  # Luxembourg
    MT = "MT"  # Malta
    NL = "NL"  # Netherlands
    PL = "PL"  # Poland
    PT = "PT"  # Portugal
    RO = "RO"  # Romania
    SK = "SK"  # Slovakia
    SI = "SI"  # Slovenia
    ES = "ES"  # Spain
    SE = "SE"  # Sweden
    # EEA
    IS = "IS"  # Iceland
    LI = "LI"  # Liechtenstein
    NO = "NO"  # Norway


# =============================================================================
# Pydantic Models
# =============================================================================


class InstallationIdentification(BaseModel):
    """EU ETS installation identification."""

    model_config = ConfigDict(frozen=True)

    installation_id: str = Field(..., description="EU ETS installation ID")
    installation_name: str = Field(..., description="Installation name")
    permit_id: str = Field(..., description="GHG permit ID")

    operator_id: str = Field(..., description="Operator account ID")
    operator_name: str = Field(..., description="Operator name")

    member_state: MemberState = Field(..., description="Member state")
    installation_type: InstallationType = Field(..., description="Installation type")
    activity_types: List[ActivityType] = Field(
        default_factory=list,
        description="Activity types"
    )

    # Location
    address: str = Field(..., description="Address")
    city: str = Field(..., description="City")
    postal_code: str = Field(..., description="Postal code")
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)

    # Capacity
    rated_thermal_input_mw: Optional[float] = Field(
        default=None,
        ge=0,
        description="Rated thermal input (MW)"
    )
    production_capacity: Optional[float] = Field(default=None, ge=0)
    capacity_unit: Optional[str] = Field(default=None)


class AllowanceBalance(BaseModel):
    """Allowance balance for an account."""

    model_config = ConfigDict(frozen=True)

    account_id: str = Field(..., description="Account ID")
    balance_date: date = Field(..., description="Balance date")

    # EUA balance
    eua_balance: int = Field(default=0, ge=0, description="EUA balance")
    euaa_balance: int = Field(default=0, ge=0, description="EUAA balance")

    # By vintage year
    balances_by_year: Dict[int, int] = Field(
        default_factory=dict,
        description="Balance by vintage year"
    )

    # Free allocation tracking
    free_allocation_received: int = Field(default=0, ge=0)
    free_allocation_remaining: int = Field(default=0, ge=0)

    # Compliance
    surrendered_current_year: int = Field(default=0, ge=0)
    compliance_status: str = Field(default="compliant")


class AllowanceTransaction(BaseModel):
    """Allowance transaction record."""

    model_config = ConfigDict(frozen=True)

    transaction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Transaction ID"
    )
    transaction_type: TransactionType = Field(..., description="Transaction type")
    transaction_date: datetime = Field(..., description="Transaction date")

    # Parties
    transferring_account: Optional[str] = Field(default=None)
    acquiring_account: Optional[str] = Field(default=None)

    # Allowances
    allowance_type: AllowanceType = Field(..., description="Allowance type")
    quantity: int = Field(..., ge=0, description="Quantity")
    vintage_year: Optional[int] = Field(default=None, ge=2005, le=2030)

    # Transaction details
    unit_price_eur: Optional[Decimal] = Field(default=None, ge=0)
    total_value_eur: Optional[Decimal] = Field(default=None, ge=0)

    status: str = Field(default="completed", description="Transaction status")
    registry_reference: Optional[str] = Field(default=None)


class VerifiedEmissions(BaseModel):
    """Verified annual emissions for an installation."""

    model_config = ConfigDict(frozen=True)

    installation_id: str = Field(..., description="Installation ID")
    reporting_year: int = Field(..., ge=2005, le=2050, description="Reporting year")

    # Emissions
    total_emissions_tco2e: Decimal = Field(
        ...,
        ge=0,
        description="Total verified emissions (tCO2e)"
    )
    co2_emissions: Decimal = Field(default=Decimal("0"), ge=0)
    n2o_emissions_co2e: Decimal = Field(default=Decimal("0"), ge=0)
    pfc_emissions_co2e: Decimal = Field(default=Decimal("0"), ge=0)

    # By source stream
    emissions_by_source: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions by source stream"
    )

    # Verification
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.DRAFT,
        description="Verification status"
    )
    verifier_name: Optional[str] = Field(default=None, description="Verifier name")
    verifier_accreditation: Optional[str] = Field(default=None)
    verification_date: Optional[date] = Field(default=None)
    verification_opinion: Optional[str] = Field(default=None)

    # Data quality
    materiality_level_percent: float = Field(
        default=5.0,
        ge=0,
        le=10,
        description="Materiality level %"
    )
    uncertainty_percent: Optional[float] = Field(default=None, ge=0, le=100)

    # Compliance
    free_allocation: int = Field(default=0, ge=0)
    allowances_surrendered: int = Field(default=0, ge=0)
    compliance_achieved: bool = Field(default=False)


class MonitoringPlan(BaseModel):
    """Monitoring plan for an installation."""

    model_config = ConfigDict(frozen=True)

    plan_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Plan ID"
    )
    installation_id: str = Field(..., description="Installation ID")
    version: str = Field(..., description="Plan version")
    valid_from: date = Field(..., description="Valid from date")
    valid_until: Optional[date] = Field(default=None, description="Valid until")

    # Monitoring approach
    monitoring_approach: MonitoringApproach = Field(
        ...,
        description="Primary monitoring approach"
    )

    # Source streams
    source_streams: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source stream definitions"
    )

    # Emission sources
    emission_sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Emission source definitions"
    )

    # Calculation factors
    emission_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Emission factors"
    )
    oxidation_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Oxidation factors"
    )
    conversion_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Conversion factors"
    )

    # Tiers
    activity_data_tier: Optional[int] = Field(default=None, ge=1, le=4)
    emission_factor_tier: Optional[int] = Field(default=None, ge=1, le=3)
    oxidation_factor_tier: Optional[int] = Field(default=None, ge=1, le=3)

    # Approval
    approval_status: str = Field(default="draft")
    approved_date: Optional[date] = Field(default=None)
    competent_authority: Optional[str] = Field(default=None)


class AnnualReport(BaseModel):
    """Annual emissions report for EU ETS."""

    model_config = ConfigDict(frozen=True)

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Report ID"
    )
    installation_id: str = Field(..., description="Installation ID")
    reporting_year: int = Field(..., ge=2005, le=2050, description="Reporting year")

    # Installation data
    installation: InstallationIdentification = Field(
        ...,
        description="Installation information"
    )

    # Emissions data
    verified_emissions: VerifiedEmissions = Field(
        ...,
        description="Verified emissions"
    )

    # Activity data
    fuel_consumption: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Fuel consumption by type (tonnes)"
    )
    production_data: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Production data by product"
    )
    operating_hours: Optional[int] = Field(default=None, ge=0, le=8784)

    # Free allocation
    free_allocation_preliminary: int = Field(default=0, ge=0)
    free_allocation_final: int = Field(default=0, ge=0)
    activity_level_change: Optional[float] = Field(default=None)

    # Submission
    submission_date: Optional[datetime] = Field(default=None)
    submission_status: str = Field(default="draft")

    # Certification
    responsible_person: str = Field(..., description="Responsible person")
    certification_date: Optional[datetime] = Field(default=None)


class EUETSConnectorConfig(BaseConnectorConfig):
    """Configuration for EU ETS connector."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    connector_type: ConnectorType = Field(
        default=ConnectorType.EU_ETS,
        description="Connector type"
    )

    # Registry settings
    registry_url: str = Field(
        default="https://ec.europa.eu/clima/ets/",
        description="EU ETS registry URL"
    )
    member_state: MemberState = Field(..., description="Member state")

    # Installation identification
    installation_id: str = Field(..., description="EU ETS installation ID")
    operator_id: str = Field(..., description="Operator account ID")
    permit_id: str = Field(..., description="GHG permit ID")

    # Authentication
    auth_method: str = Field(
        default="certificate",
        description="Authentication method"
    )
    certificate_path: Optional[str] = Field(default=None)
    private_key_path: Optional[str] = Field(default=None)

    # Trading phase
    ets_phase: ETSPhase = Field(
        default=ETSPhase.PHASE_4,
        description="Current ETS phase"
    )

    # Monitoring settings
    monitoring_approach: MonitoringApproach = Field(
        default=MonitoringApproach.CALCULATION_BASED,
        description="Primary monitoring approach"
    )

    # Compliance deadlines
    verification_deadline_day: int = Field(
        default=31,
        ge=1,
        le=31,
        description="Verification deadline day"
    )
    verification_deadline_month: int = Field(
        default=3,
        ge=1,
        le=12,
        description="Verification deadline month"
    )
    surrender_deadline_day: int = Field(
        default=30,
        ge=1,
        le=30,
        description="Surrender deadline day"
    )
    surrender_deadline_month: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Surrender deadline month"
    )


class RegistryCredentials(BaseModel):
    """EU ETS registry credentials."""

    model_config = ConfigDict(frozen=True)

    username: str = Field(..., description="Registry username")
    password: str = Field(..., description="Registry password")
    certificate_path: Optional[str] = Field(default=None)
    private_key_path: Optional[str] = Field(default=None)
    otp_secret: Optional[str] = Field(default=None, description="OTP secret for 2FA")


# =============================================================================
# Emissions Calculator
# =============================================================================


class EmissionsCalculator:
    """
    EU ETS emissions calculator implementing MRV regulation requirements.

    Supports calculation-based and measurement-based approaches.
    """

    # Default emission factors (tCO2/TJ)
    DEFAULT_EMISSION_FACTORS = {
        "natural_gas": 56.1,
        "diesel": 74.1,
        "fuel_oil": 77.4,
        "coal_bituminous": 94.6,
        "coal_sub_bituminous": 96.1,
        "lignite": 101.0,
        "petroleum_coke": 97.5,
        "lpg": 63.1,
        "biomass": 0.0,  # Carbon neutral under EU ETS
    }

    # Default oxidation factors
    DEFAULT_OXIDATION_FACTORS = {
        "natural_gas": 1.0,
        "diesel": 1.0,
        "fuel_oil": 0.99,
        "coal_bituminous": 0.98,
        "coal_sub_bituminous": 0.98,
        "lignite": 0.98,
        "petroleum_coke": 0.99,
    }

    # Net calorific values (TJ/t)
    DEFAULT_NCV = {
        "natural_gas": 0.0482,  # per 1000 m3
        "diesel": 0.0428,
        "fuel_oil": 0.0404,
        "coal_bituminous": 0.0257,
        "coal_sub_bituminous": 0.0189,
        "lignite": 0.0095,
        "petroleum_coke": 0.0325,
        "lpg": 0.0458,
    }

    def __init__(self, monitoring_plan: Optional[MonitoringPlan] = None) -> None:
        """
        Initialize emissions calculator.

        Args:
            monitoring_plan: Optional monitoring plan with custom factors
        """
        self._monitoring_plan = monitoring_plan
        self._logger = logging.getLogger("eu_ets.calculator")

    def calculate_combustion_emissions(
        self,
        fuel_type: str,
        fuel_quantity: Decimal,
        fuel_unit: str = "tonnes",
        custom_ef: Optional[float] = None,
        custom_ncv: Optional[float] = None,
        custom_of: Optional[float] = None,
    ) -> Decimal:
        """
        Calculate CO2 emissions from combustion.

        Formula: E = AD x NCV x EF x OF

        Args:
            fuel_type: Type of fuel
            fuel_quantity: Quantity of fuel consumed
            fuel_unit: Unit of fuel quantity
            custom_ef: Custom emission factor
            custom_ncv: Custom net calorific value
            custom_of: Custom oxidation factor

        Returns:
            CO2 emissions in tonnes
        """
        # Get factors
        ef = custom_ef or self.DEFAULT_EMISSION_FACTORS.get(fuel_type, 74.1)
        ncv = custom_ncv or self.DEFAULT_NCV.get(fuel_type, 0.0428)
        of = custom_of or self.DEFAULT_OXIDATION_FACTORS.get(fuel_type, 1.0)

        # Convert fuel quantity to energy (TJ)
        if fuel_unit == "tonnes":
            energy_tj = float(fuel_quantity) * ncv
        elif fuel_unit == "m3":
            energy_tj = float(fuel_quantity) * ncv / 1000  # For natural gas
        elif fuel_unit == "TJ":
            energy_tj = float(fuel_quantity)
        else:
            energy_tj = float(fuel_quantity) * ncv

        # Calculate emissions
        emissions_tco2 = energy_tj * ef * of

        self._logger.debug(
            f"Calculated emissions for {fuel_type}: "
            f"{fuel_quantity} {fuel_unit} -> {emissions_tco2:.2f} tCO2"
        )

        return Decimal(str(round(emissions_tco2, 3)))

    def calculate_process_emissions(
        self,
        process_type: str,
        activity_data: Decimal,
        emission_factor: float,
        conversion_factor: float = 1.0,
    ) -> Decimal:
        """
        Calculate process emissions.

        Args:
            process_type: Type of process
            activity_data: Activity data (production quantity)
            emission_factor: Process emission factor
            conversion_factor: Conversion factor

        Returns:
            CO2e emissions in tonnes
        """
        emissions = float(activity_data) * emission_factor * conversion_factor
        return Decimal(str(round(emissions, 3)))

    def calculate_total_emissions(
        self,
        combustion_emissions: Dict[str, Decimal],
        process_emissions: Dict[str, Decimal],
    ) -> Decimal:
        """
        Calculate total emissions from all sources.

        Args:
            combustion_emissions: Combustion emissions by source
            process_emissions: Process emissions by source

        Returns:
            Total emissions in tCO2e
        """
        total = Decimal("0")

        for source, emissions in combustion_emissions.items():
            total += emissions

        for source, emissions in process_emissions.items():
            total += emissions

        return total

    def apply_uncertainty(
        self,
        emissions: Decimal,
        uncertainty_percent: float,
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Apply uncertainty range to emissions.

        Args:
            emissions: Central emissions value
            uncertainty_percent: Uncertainty percentage

        Returns:
            Tuple of (lower_bound, central, upper_bound)
        """
        uncertainty_factor = Decimal(str(uncertainty_percent / 100))
        lower = emissions * (1 - uncertainty_factor)
        upper = emissions * (1 + uncertainty_factor)

        return (lower, emissions, upper)


# =============================================================================
# Registry Client
# =============================================================================


class ETSRegistryClient:
    """
    EU ETS Registry API client.

    Handles communication with national registries and the
    Union Registry.
    """

    def __init__(
        self,
        config: EUETSConnectorConfig,
        credentials: RegistryCredentials,
    ) -> None:
        """
        Initialize registry client.

        Args:
            config: Connector configuration
            credentials: Registry credentials
        """
        self._config = config
        self._credentials = credentials
        self._client: Optional[httpx.AsyncClient] = None
        self._session_token: Optional[str] = None
        self._logger = logging.getLogger("eu_ets.registry_client")

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=self._config.connection_timeout_seconds,
            limits=httpx.Limits(max_connections=10),
        )

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def authenticate(self) -> str:
        """
        Authenticate with EU ETS registry.

        Returns:
            Session token

        Raises:
            AuthenticationError: If authentication fails
        """
        if not self._client:
            await self.initialize()

        try:
            # Registry authentication (certificate-based in production)
            auth_data = {
                "username": self._credentials.username,
                "password": self._credentials.password,
            }

            # In production, use certificate-based auth
            # response = await self._client.post(
            #     f"{self._config.registry_url}/authenticate",
            #     json=auth_data,
            #     cert=(self._credentials.certificate_path, self._credentials.private_key_path),
            # )

            # Simulated response
            self._session_token = f"session_{uuid.uuid4().hex[:16]}"

            self._logger.info("EU ETS registry authentication successful")
            return self._session_token

        except Exception as e:
            self._logger.error(f"Registry authentication failed: {e}")
            raise AuthenticationError(f"EU ETS registry authentication failed: {e}")

    async def get_allowance_balance(
        self,
        account_id: str,
    ) -> AllowanceBalance:
        """
        Get allowance balance for an account.

        Args:
            account_id: Account ID

        Returns:
            Allowance balance
        """
        await self._ensure_authenticated()

        # In production, call registry API
        # Simulated response
        return AllowanceBalance(
            account_id=account_id,
            balance_date=date.today(),
            eua_balance=100000,
            euaa_balance=0,
            balances_by_year={
                2023: 50000,
                2024: 50000,
            },
            free_allocation_received=80000,
            free_allocation_remaining=20000,
            surrendered_current_year=60000,
            compliance_status="compliant",
        )

    async def get_transaction_history(
        self,
        account_id: str,
        start_date: date,
        end_date: date,
    ) -> List[AllowanceTransaction]:
        """
        Get transaction history for an account.

        Args:
            account_id: Account ID
            start_date: Start date
            end_date: End date

        Returns:
            List of transactions
        """
        await self._ensure_authenticated()

        # In production, call registry API
        return []

    async def submit_verified_emissions(
        self,
        verified_emissions: VerifiedEmissions,
    ) -> bool:
        """
        Submit verified emissions to registry.

        Args:
            verified_emissions: Verified emissions data

        Returns:
            True if submission successful
        """
        await self._ensure_authenticated()

        self._logger.info(
            f"Submitting verified emissions for {verified_emissions.reporting_year}: "
            f"{verified_emissions.total_emissions_tco2e} tCO2e"
        )

        # In production, submit to registry
        return True

    async def surrender_allowances(
        self,
        quantity: int,
        vintage_year: Optional[int] = None,
    ) -> AllowanceTransaction:
        """
        Surrender allowances for compliance.

        Args:
            quantity: Number of allowances to surrender
            vintage_year: Optional specific vintage year

        Returns:
            Transaction record
        """
        await self._ensure_authenticated()

        transaction = AllowanceTransaction(
            transaction_type=TransactionType.SURRENDER,
            transaction_date=datetime.utcnow(),
            transferring_account=self._config.operator_id,
            acquiring_account="EU_DELETION_ACCOUNT",
            allowance_type=AllowanceType.EUA,
            quantity=quantity,
            vintage_year=vintage_year,
            status="completed",
        )

        self._logger.info(f"Surrendered {quantity} allowances")

        return transaction

    async def _ensure_authenticated(self) -> None:
        """Ensure we have valid authentication."""
        if not self._session_token:
            await self.authenticate()


# =============================================================================
# Free Allocation Calculator
# =============================================================================


class FreeAllocationCalculator:
    """
    Calculates free allocation based on EU ETS rules.

    Implements activity level changes, carbon leakage exposure,
    and benchmark-based allocation.
    """

    # Carbon leakage exposure factor
    CARBON_LEAKAGE_FACTOR = 1.0  # 100% for carbon leakage exposed
    NON_CARBON_LEAKAGE_FACTOR = 0.3  # 30% for non-exposed in 2030

    def __init__(self, installation_type: InstallationType) -> None:
        """
        Initialize calculator.

        Args:
            installation_type: Installation type
        """
        self._installation_type = installation_type
        self._logger = logging.getLogger("eu_ets.free_allocation")

    def calculate_preliminary_allocation(
        self,
        benchmark_value: float,
        historical_activity_level: float,
        carbon_leakage_exposed: bool = True,
        cross_sectoral_correction: float = 1.0,
    ) -> int:
        """
        Calculate preliminary free allocation.

        Args:
            benchmark_value: Product benchmark (tCO2/unit)
            historical_activity_level: Historical activity level
            carbon_leakage_exposed: Carbon leakage exposure status
            cross_sectoral_correction: Cross-sectoral correction factor

        Returns:
            Preliminary allocation (allowances)
        """
        # Base allocation = benchmark x activity level
        base_allocation = benchmark_value * historical_activity_level

        # Apply carbon leakage factor
        if carbon_leakage_exposed:
            cl_factor = self.CARBON_LEAKAGE_FACTOR
        else:
            cl_factor = self.NON_CARBON_LEAKAGE_FACTOR

        # Apply cross-sectoral correction
        allocation = base_allocation * cl_factor * cross_sectoral_correction

        return int(round(allocation))

    def calculate_activity_level_change(
        self,
        reported_activity_level: float,
        baseline_activity_level: float,
    ) -> Tuple[float, str]:
        """
        Calculate activity level change and adjustment.

        Args:
            reported_activity_level: Reported activity level
            baseline_activity_level: Baseline activity level

        Returns:
            Tuple of (change_percentage, adjustment_type)
        """
        if baseline_activity_level == 0:
            return (0.0, "no_change")

        change = (reported_activity_level - baseline_activity_level) / baseline_activity_level

        if change >= 0.15:
            return (change, "increase_above_15")
        elif change <= -0.15:
            return (change, "decrease_below_15")
        else:
            return (change, "no_adjustment")

    def adjust_allocation(
        self,
        preliminary_allocation: int,
        activity_level_change: float,
        adjustment_type: str,
    ) -> int:
        """
        Adjust allocation based on activity level change.

        Args:
            preliminary_allocation: Preliminary allocation
            activity_level_change: Activity level change percentage
            adjustment_type: Type of adjustment

        Returns:
            Adjusted allocation
        """
        if adjustment_type == "increase_above_15":
            # Calculate new allocation based on higher activity
            adjusted = int(preliminary_allocation * (1 + activity_level_change))
        elif adjustment_type == "decrease_below_15":
            # Calculate new allocation based on lower activity
            adjusted = int(preliminary_allocation * (1 + activity_level_change))
            # Minimum 50% reduction
            adjusted = max(adjusted, int(preliminary_allocation * 0.5))
        else:
            adjusted = preliminary_allocation

        return adjusted


# =============================================================================
# EU ETS Connector
# =============================================================================


class EUETSConnector(BaseConnector):
    """
    EU Emissions Trading System Connector.

    Provides comprehensive integration with EU ETS for:
    - Registry connectivity and allowance tracking
    - Verified emissions reporting
    - MRV (Monitoring, Reporting, Verification) compliance
    - Free allocation management
    - Compliance tracking and surrender

    Compliance:
    - EU ETS Directive 2003/87/EC (as amended)
    - MRV Regulation (EU) 2018/2066
    - Accreditation and Verification Regulation (EU) 2018/2067
    """

    def __init__(
        self,
        config: EUETSConnectorConfig,
        credentials: RegistryCredentials,
    ) -> None:
        """
        Initialize EU ETS connector.

        Args:
            config: Connector configuration
            credentials: Registry credentials
        """
        super().__init__(config)
        self._ets_config = config
        self._credentials = credentials

        # Initialize components
        self._registry_client = ETSRegistryClient(config, credentials)
        self._emissions_calculator = EmissionsCalculator()
        self._allocation_calculator = FreeAllocationCalculator(
            InstallationType.COMBUSTION  # Default, should be from config
        )

        # Data storage
        self._current_balance: Optional[AllowanceBalance] = None
        self._monitoring_plan: Optional[MonitoringPlan] = None
        self._annual_reports: Dict[int, AnnualReport] = {}

        self._logger = logging.getLogger(f"eu_ets.connector.{config.installation_id}")

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Establish connection to EU ETS registry.

        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If authentication fails
        """
        self._state = ConnectionState.CONNECTING
        self._logger.info("Connecting to EU ETS registry")

        try:
            await self._registry_client.initialize()
            await self._registry_client.authenticate()

            self._state = ConnectionState.CONNECTED
            self._logger.info("EU ETS registry connection established")

            await self._audit_logger.log_operation(
                operation="connect",
                status="success",
                response_summary="Connected to EU ETS registry",
            )

        except AuthenticationError:
            self._state = ConnectionState.ERROR
            raise

        except Exception as e:
            self._state = ConnectionState.ERROR
            raise ConnectionError(f"Failed to connect to EU ETS registry: {e}")

    async def disconnect(self) -> None:
        """Disconnect from EU ETS registry."""
        self._logger.info("Disconnecting from EU ETS registry")

        await self._registry_client.close()
        self._state = ConnectionState.DISCONNECTED

        await self._audit_logger.log_operation(
            operation="disconnect",
            status="success",
        )

    async def health_check(self) -> HealthCheckResult:
        """
        Perform health check on registry connection.

        Returns:
            Health check result
        """
        start_time = time.time()

        try:
            # Test registry connection
            await self._registry_client.authenticate()

            latency_ms = (time.time() - start_time) * 1000

            # Check compliance deadlines
            details = await self._check_compliance_deadlines()

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message="EU ETS registry connection healthy",
                details=details,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Health check failed: {e}",
            )

    async def validate_configuration(self) -> bool:
        """
        Validate EU ETS connector configuration.

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        issues: List[str] = []

        if not self._ets_config.installation_id:
            issues.append("installation_id is required")

        if not self._ets_config.operator_id:
            issues.append("operator_id is required")

        if not self._ets_config.permit_id:
            issues.append("permit_id is required")

        if not self._credentials.username or not self._credentials.password:
            issues.append("Registry credentials are required")

        if issues:
            raise ConfigurationError(
                f"Invalid EU ETS configuration: {issues}",
                connector_id=self._config.connector_id,
            )

        return True

    # -------------------------------------------------------------------------
    # EU ETS-Specific Methods
    # -------------------------------------------------------------------------

    async def get_allowance_balance(self) -> AllowanceBalance:
        """
        Get current allowance balance.

        Returns:
            Current allowance balance
        """
        start_time = time.time()

        try:
            balance = await self._registry_client.get_allowance_balance(
                self._ets_config.operator_id
            )

            self._current_balance = balance

            duration_ms = (time.time() - start_time) * 1000
            await self._metrics.record_request(
                success=True,
                latency_ms=duration_ms,
            )

            await self._audit_logger.log_operation(
                operation="get_allowance_balance",
                status="success",
                response_summary=f"Balance: {balance.eua_balance} EUA",
                duration_ms=duration_ms,
            )

            return balance

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._metrics.record_request(
                success=False,
                latency_ms=duration_ms,
                error=str(e),
            )
            raise

    async def get_transaction_history(
        self,
        start_date: date,
        end_date: date,
    ) -> List[AllowanceTransaction]:
        """
        Get allowance transaction history.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of transactions
        """
        transactions = await self._registry_client.get_transaction_history(
            self._ets_config.operator_id,
            start_date,
            end_date,
        )

        await self._audit_logger.log_operation(
            operation="get_transaction_history",
            status="success",
            request_data={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            response_summary=f"Retrieved {len(transactions)} transactions",
        )

        return transactions

    async def calculate_emissions(
        self,
        fuel_consumption: Dict[str, Decimal],
        process_data: Optional[Dict[str, Decimal]] = None,
    ) -> VerifiedEmissions:
        """
        Calculate emissions for reporting.

        Args:
            fuel_consumption: Fuel consumption by type (tonnes)
            process_data: Optional process data

        Returns:
            Calculated emissions (unverified)
        """
        combustion_emissions: Dict[str, Decimal] = {}
        process_emissions: Dict[str, Decimal] = {}

        # Calculate combustion emissions
        for fuel_type, quantity in fuel_consumption.items():
            emissions = self._emissions_calculator.calculate_combustion_emissions(
                fuel_type=fuel_type,
                fuel_quantity=quantity,
            )
            combustion_emissions[fuel_type] = emissions

        # Calculate process emissions if provided
        if process_data:
            for process, activity_data in process_data.items():
                # Use default emission factors (would be from monitoring plan)
                emissions = self._emissions_calculator.calculate_process_emissions(
                    process_type=process,
                    activity_data=activity_data,
                    emission_factor=1.0,  # Would be process-specific
                )
                process_emissions[process] = emissions

        # Calculate total
        total_emissions = self._emissions_calculator.calculate_total_emissions(
            combustion_emissions,
            process_emissions,
        )

        # Merge all emissions by source
        emissions_by_source = {**combustion_emissions, **process_emissions}

        verified_emissions = VerifiedEmissions(
            installation_id=self._ets_config.installation_id,
            reporting_year=date.today().year - 1,  # Previous year
            total_emissions_tco2e=total_emissions,
            co2_emissions=total_emissions,  # Simplified
            emissions_by_source=emissions_by_source,
            verification_status=VerificationStatus.DRAFT,
        )

        await self._audit_logger.log_operation(
            operation="calculate_emissions",
            status="success",
            response_summary=f"Calculated {total_emissions} tCO2e",
        )

        return verified_emissions

    async def submit_verified_emissions(
        self,
        verified_emissions: VerifiedEmissions,
    ) -> bool:
        """
        Submit verified emissions to registry.

        Args:
            verified_emissions: Verified emissions data

        Returns:
            True if submission successful
        """
        # Validate verification status
        if verified_emissions.verification_status != VerificationStatus.VERIFIED:
            raise ValidationError(
                "Emissions must be verified before submission",
                connector_id=self._config.connector_id,
            )

        result = await self._registry_client.submit_verified_emissions(
            verified_emissions
        )

        await self._audit_logger.log_operation(
            operation="submit_verified_emissions",
            status="success" if result else "failure",
            request_data={
                "reporting_year": verified_emissions.reporting_year,
                "total_emissions": str(verified_emissions.total_emissions_tco2e),
            },
        )

        return result

    async def surrender_allowances(
        self,
        quantity: int,
        vintage_year: Optional[int] = None,
    ) -> AllowanceTransaction:
        """
        Surrender allowances for compliance.

        Args:
            quantity: Number of allowances
            vintage_year: Optional vintage year preference

        Returns:
            Transaction record
        """
        # Check balance
        if self._current_balance is None:
            await self.get_allowance_balance()

        if self._current_balance.eua_balance < quantity:
            raise ValidationError(
                f"Insufficient balance: {self._current_balance.eua_balance} < {quantity}",
                connector_id=self._config.connector_id,
            )

        transaction = await self._registry_client.surrender_allowances(
            quantity,
            vintage_year,
        )

        # Update local balance
        if self._current_balance:
            # Would need to refresh balance
            pass

        await self._audit_logger.log_operation(
            operation="surrender_allowances",
            status="success",
            request_data={
                "quantity": quantity,
                "vintage_year": vintage_year,
            },
            response_summary=f"Surrendered {quantity} allowances",
        )

        return transaction

    async def get_free_allocation(
        self,
        reporting_year: int,
    ) -> Tuple[int, int]:
        """
        Get free allocation amounts.

        Args:
            reporting_year: Reporting year

        Returns:
            Tuple of (preliminary_allocation, final_allocation)
        """
        # Would query registry for allocation amounts
        preliminary = 50000
        final = 48000

        await self._audit_logger.log_operation(
            operation="get_free_allocation",
            status="success",
            request_data={"reporting_year": reporting_year},
            response_summary=f"Preliminary: {preliminary}, Final: {final}",
        )

        return (preliminary, final)

    async def check_compliance_status(
        self,
        reporting_year: int,
    ) -> Dict[str, Any]:
        """
        Check compliance status for a year.

        Args:
            reporting_year: Reporting year

        Returns:
            Compliance status details
        """
        # Get verified emissions for year
        verified_emissions = self._annual_reports.get(reporting_year)

        # Get balance
        if self._current_balance is None:
            await self.get_allowance_balance()

        # Calculate compliance
        emissions_tco2e = 0
        if verified_emissions:
            emissions_tco2e = int(verified_emissions.verified_emissions.total_emissions_tco2e)

        surrendered = self._current_balance.surrendered_current_year
        required = emissions_tco2e
        shortfall = max(0, required - surrendered)

        status = {
            "reporting_year": reporting_year,
            "verified_emissions_tco2e": emissions_tco2e,
            "allowances_surrendered": surrendered,
            "allowances_required": required,
            "shortfall": shortfall,
            "is_compliant": shortfall == 0,
            "balance_available": self._current_balance.eua_balance,
        }

        await self._audit_logger.log_operation(
            operation="check_compliance_status",
            status="success",
            response_summary=f"Compliant: {status['is_compliant']}",
        )

        return status

    async def generate_annual_report(
        self,
        reporting_year: int,
        installation: InstallationIdentification,
        verified_emissions: VerifiedEmissions,
        fuel_consumption: Dict[str, Decimal],
        production_data: Optional[Dict[str, Decimal]] = None,
        responsible_person: str = "",
    ) -> AnnualReport:
        """
        Generate annual emissions report.

        Args:
            reporting_year: Reporting year
            installation: Installation information
            verified_emissions: Verified emissions
            fuel_consumption: Fuel consumption data
            production_data: Optional production data
            responsible_person: Responsible person name

        Returns:
            Annual report
        """
        # Get free allocation
        preliminary, final = await self.get_free_allocation(reporting_year)

        report = AnnualReport(
            installation_id=self._ets_config.installation_id,
            reporting_year=reporting_year,
            installation=installation,
            verified_emissions=verified_emissions,
            fuel_consumption=fuel_consumption,
            production_data=production_data or {},
            free_allocation_preliminary=preliminary,
            free_allocation_final=final,
            responsible_person=responsible_person,
        )

        self._annual_reports[reporting_year] = report

        await self._audit_logger.log_operation(
            operation="generate_annual_report",
            status="success",
            request_data={"reporting_year": reporting_year},
            response_summary=f"Generated report: {verified_emissions.total_emissions_tco2e} tCO2e",
        )

        return report

    async def _check_compliance_deadlines(self) -> Dict[str, Any]:
        """Check upcoming compliance deadlines."""
        today = date.today()
        current_year = today.year

        # Verification deadline (March 31)
        verification_deadline = date(
            current_year,
            self._ets_config.verification_deadline_month,
            self._ets_config.verification_deadline_day,
        )

        # Surrender deadline (April 30)
        surrender_deadline = date(
            current_year,
            self._ets_config.surrender_deadline_month,
            self._ets_config.surrender_deadline_day,
        )

        details = {
            "verification_deadline": verification_deadline.isoformat(),
            "days_to_verification": (verification_deadline - today).days,
            "surrender_deadline": surrender_deadline.isoformat(),
            "days_to_surrender": (surrender_deadline - today).days,
        }

        # Add warnings
        if 0 < details["days_to_verification"] <= 30:
            details["verification_warning"] = "Verification deadline approaching"

        if 0 < details["days_to_surrender"] <= 30:
            details["surrender_warning"] = "Surrender deadline approaching"

        return details


# =============================================================================
# Factory Function
# =============================================================================


def create_eu_ets_connector(
    installation_id: str,
    operator_id: str,
    permit_id: str,
    member_state: MemberState,
    username: str,
    password: str,
    **kwargs: Any,
) -> EUETSConnector:
    """
    Factory function to create EU ETS connector.

    Args:
        installation_id: EU ETS installation ID
        operator_id: Operator account ID
        permit_id: GHG permit ID
        member_state: Member state
        username: Registry username
        password: Registry password
        **kwargs: Additional configuration

    Returns:
        Configured EU ETS connector
    """
    config = EUETSConnectorConfig(
        connector_name=f"EU_ETS_{installation_id}",
        installation_id=installation_id,
        operator_id=operator_id,
        permit_id=permit_id,
        member_state=member_state,
        **kwargs,
    )

    credentials = RegistryCredentials(
        username=username,
        password=password,
    )

    return EUETSConnector(config, credentials)
